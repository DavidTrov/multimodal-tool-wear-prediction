"""
Phase 1 – image-only baseline (ResNet18).

Usage:
    python experiments/phase1_image_only/train.py

Run from the thesis root so that `src/` is on the path.
"""

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
EPOCHS      = 30
NUM_WORKERS = 4
# ──────────────────────────────────────────────────────────────────────────────


def run():
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
    model = build_resnet18_regressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    CKPT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    history = []
    best_val_mae = float("inf")

    for epoch in range(1, EPOCHS + 1):
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

        history.append({
            "epoch":        epoch,
            "train_loss":   round(train_loss, 4),
            "val_mae":      round(val_mae, 4),
            "val_mae_std":  round(val_mae_std, 4),
        })
        print(f"Epoch {epoch:3d}/{EPOCHS}  train_loss={train_loss:.2f}  val_mae={val_mae:.2f} ± {val_mae_std:.2f} µm")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), CKPT_DIR / "phase1_best.pt")

    # Save history
    with open(RESULTS_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val MAE: {best_val_mae:.2f} µm")
    print(f"Checkpoint saved to: {CKPT_DIR / 'phase1_best.pt'}")
    print(f"History saved to:    {RESULTS_DIR / 'history.json'}")


if __name__ == "__main__":
    run()
