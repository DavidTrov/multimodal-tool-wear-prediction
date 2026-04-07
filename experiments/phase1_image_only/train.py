"""
Phase 1 – image-only baseline (ResNet18).

Usage:
    python experiments/phase1_image_only/train.py           # fresh start
    python experiments/phase1_image_only/train.py --resume  # continue from checkpoint

Run from the thesis root so that `src/` is on the path.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow imports from src/
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import MATWIDataset
from src.models.image_model import build_resnet18_regressor
from src.utils.metrics import mae

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT   = ROOT / "data" / "raw"
CKPT_DIR    = ROOT / "checkpoints"
RESULTS_DIR = Path(__file__).parent / "results"

LR          = 1e-4
BATCH_SIZE  = 16
EPOCHS      = 60
NUM_WORKERS = 4
# ──────────────────────────────────────────────────────────────────────────────


def run(resume: bool):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Data
    train_ds = MATWIDataset(DATA_ROOT, split="train")
    val_ds   = MATWIDataset(DATA_ROOT, split="val")
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Model
    model     = build_resnet18_regressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    CKPT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Resume ──
    start_epoch  = 1
    best_val_mae = float("inf")
    history      = []

    ckpt_path = CKPT_DIR / "phase1_best.pt"
    if resume:
        if not ckpt_path.exists():
            print("No checkpoint found, starting fresh.")
        else:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"Resumed from {ckpt_path}")

        history_path = RESULTS_DIR / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            start_epoch  = history[-1]["epoch"] + 1
            best_val_mae = min(h["val_mae"] for h in history)
            print(f"Continuing from epoch {start_epoch}  |  Best so far: {best_val_mae:.2f} µm")

    for epoch in range(start_epoch, start_epoch + EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images  = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            optimizer.zero_grad()
            preds = model(images)
            loss  = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(images)

        train_loss /= len(train_ds)

        # ── Validate ──
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images  = images.to(device)
                targets = targets.to(device)
                preds   = model(images).squeeze(1)
                all_preds.append(preds)
                all_targets.append(targets)

        all_preds   = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        val_mae     = mae(all_preds, all_targets)
        val_mae_std = (all_preds - all_targets).abs().std().item()

        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch":       epoch,
            "train_loss":  round(train_loss, 4),
            "val_mae":     round(val_mae, 4),
            "val_mae_std": round(val_mae_std, 4),
            "lr":          current_lr,
        })
        print(f"Epoch {epoch:3d}  train_loss={train_loss:.2f}  val_mae={val_mae:.2f} ± {val_mae_std:.2f} µm  lr={current_lr:.2e}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ New best: {best_val_mae:.2f} µm")

        # Save history after every epoch so it's safe to interrupt
        with open(RESULTS_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nBest val MAE: {best_val_mae:.2f} µm")
    print(f"Checkpoint:  {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Continue from last checkpoint")
    args = parser.parse_args()
    run(resume=args.resume)
