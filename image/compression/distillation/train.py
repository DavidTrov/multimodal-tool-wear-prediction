"""
Compression Phase 2 — Knowledge Distillation (regression adaptation).

Trains the pruned student model to imitate the original FP32 ResNet-18
teacher, recovering accuracy lost during aggressive channel pruning.

Loss (regression adaptation of Hinton et al.)
---------------------------------------------
  L = (1 - α) · MSE(student_pred, y)       ← ground-truth supervision
    +      α  · MSE(student_pred, teacher_pred)  ← teacher imitation

No temperature scaling is used — that is a classification concept for
softening probability distributions.  For scalar regression the teacher
output is already a soft continuous signal.

α = 0.5 by default (equal weight to labels and teacher).  Increase α if
the pruned student is already reasonable; decrease it if accuracy is very
poor after pruning.

Pre-requisite
-------------
    python experiments/compression/phase1_pruning/train.py

Usage
-----
    python experiments/compression/phase2_distillation/train.py
    python experiments/compression/phase2_distillation/train.py --alpha 0.7 --epochs 40

Run from the thesis root.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from image.baseline.dataset import MATWIDataset
from image.baseline.model import build_resnet18_regressor
from src.metrics import mae

DATA_ROOT   = ROOT / "data" / "raw"
CKPT_DIR    = ROOT / "checkpoints"
RESULTS_DIR = Path(__file__).parent / "results"

DEFAULT_ALPHA  = 0.5    # balance between ground-truth and teacher supervision
DEFAULT_EPOCHS = 40
LR          = 1e-4
BATCH_SIZE  = 16
NUM_WORKERS = 0


def model_size_report(model):
    n = sum(p.numel() for p in model.parameters())
    return {"n_params": n, "int8_kb": round(n / 1024, 1), "int4_kb": round(n * 0.5 / 1024, 1)}


def validate(model, loader, device):
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for images, targets in loader:
            preds_all.append(model(images.to(device)).squeeze(1).cpu())
            targets_all.append(targets)
    p, t = torch.cat(preds_all), torch.cat(targets_all)
    return mae(p, t), (p - t).abs().std().item()


def load_pruned_student(device, ckpt_path):
    """
    The pruned model has a different architecture than the original ResNet-18
    (fewer channels), so we cannot use build_resnet18_regressor() directly.
    Phase 1 saves the full model object via torch.save(model, ...), so we
    load it directly here.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        sys.exit(f"Pruned checkpoint not found: {ckpt_path}\nRun phase1_pruning/train.py first.")

    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        sys.exit(
            "Checkpoint contains only a state_dict — need the full model object.\n"
            "Re-run phase1_pruning/train.py (it saves torch.save(model, ...))."
        )
    model = obj.to(device)
    print(f"Loaded pruned model from {ckpt_path.name}")
    return model


def run(alpha: float, epochs: int, student_ckpt: str, suffix: str):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    ckpt_name    = f"resnet_distilled{suffix}.pt" if suffix else "distilled.pt"
    results_name = f"distillation_results{suffix}.json"
    history_name = f"distillation_history{suffix}.json"

    print(f"Device : {device}")
    print(f"α      : {alpha}  |  Epochs: {epochs}")
    print(f"Student: {student_ckpt}")
    print(f"Output : {ckpt_name}\n")

    # ── Load teacher (frozen original ResNet-18) ──────────────────────────────
    phase1_ckpt = CKPT_DIR / "phase1_best.pt"
    if not phase1_ckpt.exists():
        sys.exit(f"Teacher checkpoint not found: {phase1_ckpt}")

    teacher = build_resnet18_regressor().to(device)
    teacher.load_state_dict(torch.load(phase1_ckpt, map_location=device))
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    print("Teacher (phase1_best.pt) loaded and frozen")

    # ── Load pruned student ───────────────────────────────────────────────────
    student = load_pruned_student(device, student_ckpt)
    stats = model_size_report(student)
    print(f"Student params: {stats['n_params']:,}  |  INT8: {stats['int8_kb']} KB  |  INT4: {stats['int4_kb']} KB\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = MATWIDataset(DATA_ROOT, split="train")
    val_ds   = MATWIDataset(DATA_ROOT, split="val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    val_mae_before, _ = validate(student, val_loader, device)
    print(f"Val MAE before distillation: {val_mae_before:.2f} µm\n")

    # ── Training ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()
    train_ds_len = len(train_ds)

    best_val_mae = float("inf")
    history = []
    CKPT_DIR.mkdir(exist_ok=True)

    for epoch in range(1, epochs + 1):
        student.train()
        train_loss = 0.0

        for images, targets in train_loader:
            images  = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            student_pred = student(images)

            with torch.no_grad():
                teacher_pred = teacher(images)

            # Regression distillation loss
            loss_gt      = criterion(student_pred, targets)
            loss_distill = criterion(student_pred, teacher_pred)
            loss         = (1 - alpha) * loss_gt + alpha * loss_distill

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(images)

        train_loss /= train_ds_len
        val_mae_val, val_std = validate(student, val_loader, device)
        scheduler.step(val_mae_val)
        lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_mae": round(val_mae_val, 4), "val_mae_std": round(val_std, 4), "lr": lr,
        })
        print(
            f"Epoch {epoch:3d}  train_loss={train_loss:.2f}  "
            f"val_mae={val_mae_val:.2f} ± {val_std:.2f} µm  lr={lr:.2e}"
        )

        if val_mae_val < best_val_mae:
            best_val_mae = val_mae_val
            torch.save(student, CKPT_DIR / ckpt_name)
            print(f"  ✓ New best: {best_val_mae:.2f} µm  (saved {ckpt_name})")

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("DISTILLATION SUMMARY")
    print("─" * 60)
    print(f"Val MAE before : {val_mae_before:.2f} µm")
    print(f"Val MAE after  : {best_val_mae:.2f} µm")
    print(f"INT8 size      : {stats['int8_kb']} KB  (target: ≤2048 KB — NXP FRDM-MCXN947 flash)")
    print("─" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / history_name, "w") as f:
        json.dump(history, f, indent=2)
    results = {
        "alpha": alpha, "epochs": epochs,
        "val_mae_before": round(val_mae_before, 2),
        "val_mae_after": round(best_val_mae, 2),
        **stats,
    }
    with open(RESULTS_DIR / results_name, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha",  type=float, default=DEFAULT_ALPHA,
                        help="Distillation weight α (default: 0.5)")
    parser.add_argument("--epochs", type=int,   default=DEFAULT_EPOCHS,
                        help="Training epochs (default: 40)")
    parser.add_argument("--student-ckpt", type=str,
                        default=str(ROOT / "checkpoints" / "pruned.pt"),
                        help="Path to the pruned student checkpoint "
                             "(default: checkpoints/pruned.pt)")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Suffix appended to output filenames "
                             "(e.g. '_90' → resnet_distilled_90.pt). "
                             "Empty string preserves legacy name (distilled.pt).")
    args = parser.parse_args()
    run(alpha=args.alpha, epochs=args.epochs,
        student_ckpt=args.student_ckpt, suffix=args.output_suffix)
