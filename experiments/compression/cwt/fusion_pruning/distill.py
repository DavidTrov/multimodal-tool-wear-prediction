"""
Joint Fusion — Post-Pruning Knowledge Distillation.

Distils the pruned fusion model (student) from the pre-pruning compressed
fusion model (teacher), recovering accuracy lost during channel pruning.

Teacher : phase5_compressed_fusion_best.pt  (17.64 µm test MAE, FP32)
          — same model the student was pruned from; same input/output interface.

Student : fusion_pruned.pt  (full model object, non-standard channel widths)
          — loaded directly, no architecture reinstall needed.

Loss
----
  L = (1 - α) · Huber(student, y, δ=20)      ← ground-truth supervision
    +       α  · MSE(student, teacher_pred)   ← soft teacher targets

No temperature scaling — teacher output is a scalar regression value, not
a probability distribution.

The distilled model is saved as a full object (fusion_distilled.pt) because
both encoder architectures have non-standard channel widths after pruning.

Pre-requisite
-------------
    python experiments/compression/cwt/fusion_pruning/train.py

Usage
-----
    python experiments/compression/cwt/fusion_pruning/distill.py
    python experiments/compression/cwt/fusion_pruning/distill.py \\
        --student-ckpt checkpoints/fusion_pruned_30.pt --output-suffix _30
    python experiments/compression/cwt/fusion_pruning/distill.py --alpha 0.7

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

from src.data.fusion_scalogram_dataset import MATWIFusionScalogramDataset
from src.models.multiscale_fusion_model import MultiScaleFusionModel
from src.utils.metrics import mae

DATA_ROOT     = ROOT / "data" / "raw"
SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
CKPT_DIR      = ROOT / "checkpoints"
RESULTS_DIR   = Path(__file__).parent / "results"

# Teacher checkpoint — pre-pruning compressed fusion (full state dict model)
COMPRESSED_IMG_CKPT = CKPT_DIR / "resnet_distilled_budget.pt"
PHASE4_SENSOR_CKPT  = CKPT_DIR / "phase4_multiscale_sgdm_best.pt"
TEACHER_FUSION_CKPT = CKPT_DIR / "phase5_compressed_fusion_best.pt"
IMAGE_FEAT_DIM      = 309

DEFAULT_ALPHA   = 0.5
DEFAULT_EPOCHS  = 40
AUX_LAMBDA      = 0.2
LR              = 1e-4
WEIGHT_DECAY    = 1e-3
BATCH_SIZE      = 16
NUM_WORKERS     = 0
EARLY_STOP_PATIENCE = 8


# ── Utilities ──────────────────────────────────────────────────────────────────

def model_size_report(model: nn.Module) -> dict:
    n = sum(p.numel() for p in model.parameters())
    return {"n_params": n, "int8_kb": round(n / 1024, 1)}


def validate(model, loader, device):
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for images, scalograms, targets in loader:
            preds = model(images.to(device), scalograms.to(device))[0].squeeze(1).cpu()
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
        sys.exit(f"Student checkpoint not found: {student_ckpt}\n"
                 "Run experiments/compression/cwt/fusion_pruning/train.py first.")
    for p in (COMPRESSED_IMG_CKPT, PHASE4_SENSOR_CKPT, TEACHER_FUSION_CKPT):
        if not p.exists():
            sys.exit(f"Required checkpoint not found: {p}")

    suffix        = args.output_suffix
    out_ckpt_name = f"fusion_distilled{suffix}.pt"
    results_name  = f"distillation_results{suffix}.json"
    history_name  = f"distillation_history{suffix}.json"

    # ── Load teacher ──────────────────────────────────────────────────────────
    teacher = MultiScaleFusionModel(image_feat_dim=IMAGE_FEAT_DIM)
    teacher.load_compressed_image_encoder(COMPRESSED_IMG_CKPT, device=device)
    teacher.load_phase4_weights(PHASE4_SENSOR_CKPT, device=device)
    teacher.load_state_dict(
        torch.load(TEACHER_FUSION_CKPT, map_location=device, weights_only=True)
    )
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher  : {TEACHER_FUSION_CKPT.name}  ({n_teacher:,} params)")

    # ── Load student (full pruned model object) ───────────────────────────────
    student = torch.load(student_ckpt, map_location=device, weights_only=False)
    student = student.to(device)
    for p in student.parameters():
        p.requires_grad = True
    n_student  = sum(p.numel() for p in student.parameters())
    n_img      = sum(p.numel() for p in student.image_encoder.parameters())
    n_sen      = sum(p.numel() for p in student.sensor_cnn.parameters())
    reduction  = 1 - n_student / n_teacher
    print(f"Student  : {student_ckpt.name}  ({n_student:,} params, {reduction:.1%} smaller)")
    print(f"  Image encoder : {n_img:,}  ({n_img/1024:.0f} KB INT8)")
    print(f"  Sensor CNN    : {n_sen:,}  ({n_sen/1024:.0f} KB INT8)")
    print(f"Alpha    : {args.alpha}  (teacher weight in distillation loss)")
    print(f"Epochs   : {args.epochs}  (max, early-stop patience={EARLY_STOP_PATIENCE})\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, "train")
    val_ds   = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, "val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print(f"Train : {len(train_ds)} samples    Val : {len(val_ds)} samples\n")

    val_mae_pre, _ = validate(student, val_loader, device)
    print(f"Student val MAE before distillation : {val_mae_pre:.2f} µm\n")

    # ── Optimisation ──────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    huber = nn.HuberLoss(delta=20.0)
    mse   = nn.MSELoss()
    alpha = args.alpha

    best_val_mae = val_mae_pre
    patience_ctr = 0
    history      = []

    # Save pre-distillation as initial best
    CKPT_DIR.mkdir(exist_ok=True)
    torch.save(student, CKPT_DIR / out_ckpt_name)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        student.train()
        train_loss = 0.0

        for images, scalograms, targets in train_loader:
            images     = images.to(device)
            scalograms = scalograms.to(device)
            targets    = targets.to(device).unsqueeze(1)

            with torch.no_grad():
                t_final, _ = teacher(images, scalograms)

            s_final, s_aux = student(images, scalograms)

            loss_gt = huber(s_final, targets) + AUX_LAMBDA * huber(s_aux, targets)
            loss_kd = mse(s_final, t_final)
            loss    = (1 - alpha) * loss_gt + alpha * loss_kd

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(images)

        train_loss /= len(train_ds)
        val_mae_v, val_std = validate(student, val_loader, device)
        scheduler.step()
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
    final = torch.load(CKPT_DIR / out_ckpt_name, map_location="cpu", weights_only=False)
    stats = model_size_report(final)

    print("\n" + "─" * 60)
    print("DISTILLATION SUMMARY")
    print("─" * 60)
    print(f"Student before KD : {val_mae_pre:.2f} µm")
    print(f"Student after KD  : {best_val_mae:.2f} µm  (Δ = {best_val_mae - val_mae_pre:+.2f} µm)")
    print(f"Params            : {stats['n_params']:,}  (INT8: {stats['int8_kb']:.0f} KB)")
    print(f"Checkpoint        : {CKPT_DIR / out_ckpt_name}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / results_name, "w") as f:
        json.dump({
            "alpha": alpha,
            "n_params": stats["n_params"],
            "int8_kb":  stats["int8_kb"],
            "val_mae_before_kd": round(val_mae_pre, 2),
            "val_mae_after_kd":  round(best_val_mae, 2),
            "checkpoint": str(CKPT_DIR / out_ckpt_name),
        }, f, indent=2)
    with open(RESULTS_DIR / history_name, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Results           : {RESULTS_DIR / results_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-ckpt",  default=str(CKPT_DIR / "fusion_pruned.pt"),
                        help="Path to pruned fusion model (full object)")
    parser.add_argument("--output-suffix", default="",
                        help="e.g. '_30' → fusion_distilled_30.pt")
    parser.add_argument("--alpha",   type=float, default=DEFAULT_ALPHA,
                        help="Teacher weight in loss (0=labels only, 1=teacher only)")
    parser.add_argument("--epochs",  type=int,   default=DEFAULT_EPOCHS)
    args = parser.parse_args()
    run(args)
