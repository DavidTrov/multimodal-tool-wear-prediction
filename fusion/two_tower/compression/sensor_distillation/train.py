"""
CWT Sensor CNN — Compression Phase 2: Knowledge Distillation.

Trains the pruned student sensor CNN to imitate the original FP32 teacher,
recovering accuracy lost during aggressive channel pruning.

Loss (regression KD, Hinton et al. adapted for scalar regression)
------------------------------------------------------------------
  L = (1 - α) · Huber(student_pred, y, δ=20)      ← ground-truth
    +       α  · MSE(student_pred, teacher_pred)   ← teacher imitation

No temperature scaling — teacher output is already a soft continuous signal.

Pre-requisite
-------------
    python experiments/compression/cwt/phase1_pruning/train.py

Usage
-----
    python experiments/compression/cwt/phase2_distillation/train.py
    python experiments/compression/cwt/phase2_distillation/train.py \\
        --student-ckpt checkpoints/sensor_pruned_50.pt --output-suffix _50
    python experiments/compression/cwt/phase2_distillation/train.py --alpha 0.7 --epochs 50

Run from the thesis root.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from sensor.cnn.dataset import MATWISensorScalogramDataset
from sensor.multiscale.model import MultiScaleSensorCNN
from src.metrics import mae

DATA_ROOT     = ROOT / "data" / "raw"
SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
CKPT_DIR      = ROOT / "sensor" / "multiscale" / "checkpoints"
RESULTS_DIR   = Path(__file__).parent / "results"

DEFAULT_ALPHA         = 0.5
DEFAULT_EPOCHS        = 100
LR                    = 1e-4
WEIGHT_DECAY          = 1e-3
BATCH_SIZE            = 16
NUM_WORKERS           = 0
EARLY_STOP_PATIENCE   = 15


# ── Utilities ──────────────────────────────────────────────────────────────────

def model_size_report(model: nn.Module) -> dict:
    n = sum(p.numel() for p in model.parameters())
    return {
        "n_params": n,
        "fp32_kb":  round(n * 4   / 1024, 1),
        "int8_kb":  round(n       / 1024, 1),
        "int4_kb":  round(n * 0.5 / 1024, 1),
    }


def evaluate(model, loader, device):
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for scalograms, targets in loader:
            preds = model(scalograms.to(device)).squeeze(1).cpu()
            preds_all.append(preds)
            targets_all.append(targets)
    p, t = torch.cat(preds_all), torch.cat(targets_all)
    return mae(p, t), (p - t).abs().std().item()


# ── Main ───────────────────────────────────────────────────────────────────────

def run(args):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps"  if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device : {device}\n")

    student_ckpt = Path(args.student_ckpt)
    if not student_ckpt.exists():
        sys.exit(f"Student checkpoint not found: {student_ckpt}\nRun phase1_pruning/train.py first.")

    teacher_ckpt = CKPT_DIR / "phase4_multiscale_sgdm_best.pt"
    if not teacher_ckpt.exists():
        sys.exit(f"Teacher checkpoint not found: {teacher_ckpt}")

    suffix        = args.output_suffix
    out_ckpt_name = f"sensor_distilled{suffix}.pt"
    results_name  = f"distillation_results{suffix}.json"
    history_name  = f"distillation_history{suffix}.json"

    # ── Load teacher (fixed, frozen) ──────────────────────────────────────────
    teacher = MultiScaleSensorCNN().to(device)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device, weights_only=True))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher  : phase4_multiscale_sgdm_best.pt  "
          f"({sum(p.numel() for p in teacher.parameters()):,} params)")

    # ── Load student (full model object, pruned architecture) ─────────────────
    student = torch.load(student_ckpt, map_location=device, weights_only=False)
    student = student.to(device)
    n_student = sum(p.numel() for p in student.parameters())
    reduction = 1 - n_student / sum(p.numel() for p in teacher.parameters())
    print(f"Student  : {student_ckpt.name}  ({n_student:,} params, {reduction:.1%} smaller)")
    print(f"Alpha    : {args.alpha}  (teacher weight in distillation loss)")
    print(f"Epochs   : {args.epochs}  (max, early-stop patience={EARLY_STOP_PATIENCE})\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = MATWISensorScalogramDataset(SCALOGRAM_DIR, FEATURES_PATH, "train")
    val_ds   = MATWISensorScalogramDataset(SCALOGRAM_DIR, FEATURES_PATH, "val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print(f"Train : {len(train_ds)} samples    Val : {len(val_ds)} samples\n")

    # Baseline MAE before distillation
    val_mae_pre, _ = evaluate(student, val_loader, device)
    print(f"Student val MAE before distillation : {val_mae_pre:.2f} µm\n")

    # ── Optimisation ──────────────────────────────────────────────────────────
    optimizer   = torch.optim.Adam(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    huber = nn.HuberLoss(delta=20.0)
    mse   = nn.MSELoss()
    alpha = args.alpha

    best_val_mae  = val_mae_pre
    patience_ctr  = 0
    history       = []

    # Save pre-distillation checkpoint as initial best
    CKPT_DIR.mkdir(exist_ok=True)
    torch.save(student, CKPT_DIR / out_ckpt_name)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        student.train()
        train_loss = 0.0

        for scalograms, targets in train_loader:
            scalograms = scalograms.to(device)
            targets    = targets.to(device).unsqueeze(1)

            with torch.no_grad():
                teacher_pred = teacher(scalograms)

            student_pred = student(scalograms)
            loss = (1 - alpha) * huber(student_pred, targets) \
                 +       alpha  * mse(student_pred, teacher_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(scalograms)

        train_loss /= len(train_ds)
        val_mae_v, val_std = evaluate(student, val_loader, device)
        scheduler.step(val_mae_v)
        lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_mae": round(val_mae_v, 4), "val_mae_std": round(val_std, 4), "lr": lr,
        })
        print(
            f"Epoch {epoch:3d}  train_loss={train_loss:.2f}  "
            f"val_mae={val_mae_v:.2f} ± {val_std:.2f} µm  lr={lr:.2e}"
        )

        if val_mae_v < best_val_mae:
            best_val_mae = val_mae_v
            patience_ctr = 0
            torch.save(student, CKPT_DIR / out_ckpt_name)
            print(f"  ✓ New best: {best_val_mae:.2f} µm  (saved {out_ckpt_name})")
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # ── Summary ───────────────────────────────────────────────────────────────
    best_model = torch.load(CKPT_DIR / out_ckpt_name, map_location="cpu", weights_only=False)
    final_stats = model_size_report(best_model)

    print("\n" + "─" * 60)
    print("DISTILLATION SUMMARY")
    print("─" * 60)
    print(f"Teacher val MAE        : (from phase4 training, ~24.96 µm)")
    print(f"Student before KD      : {val_mae_pre:.2f} µm")
    print(f"Student after KD       : {best_val_mae:.2f} µm  "
          f"(Δ = {best_val_mae - val_mae_pre:+.2f} µm)")
    print(f"Params                 : {final_stats['n_params']:,}  "
          f"(INT8: {final_stats['int8_kb']:.1f} KB)")
    print(f"Checkpoint             : {CKPT_DIR / out_ckpt_name}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / results_name, "w") as f:
        json.dump({
            "alpha": alpha,
            "n_params": final_stats["n_params"],
            "int8_kb":  final_stats["int8_kb"],
            "val_mae_before_kd": round(val_mae_pre, 2),
            "val_mae_after_kd":  round(best_val_mae, 2),
            "checkpoint": str(CKPT_DIR / out_ckpt_name),
        }, f, indent=2)
    with open(RESULTS_DIR / history_name, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Results                : {RESULTS_DIR / results_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-ckpt",  default=str(CKPT_DIR / "sensor_pruned.pt"))
    parser.add_argument("--output-suffix", default="",
                        help="e.g. '_50' → sensor_distilled_50.pt")
    parser.add_argument("--alpha",   type=float, default=DEFAULT_ALPHA,
                        help="Teacher weight in loss (0=labels only, 1=teacher only)")
    parser.add_argument("--epochs",  type=int,   default=DEFAULT_EPOCHS)
    args = parser.parse_args()
    run(args)
