"""
Phase 1 — grid search over LR × weight_decay on the original train/val split.

Run from the thesis root:
    python experiments/phase1_image_only/grid_search.py

Saves results/grid_search.json with per-config val MAE and the best config.
"""

import json
import sys
from itertools import product
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import MATWIDataset
from src.models.image_model import build_resnet18_regressor
from src.utils.metrics import mae

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT   = ROOT / "data" / "raw"
CKPT_DIR    = ROOT / "checkpoints"
RESULTS_DIR = Path(__file__).parent / "results"

BATCH_SIZE  = 16
EPOCHS      = 60
NUM_WORKERS = 0

LR_VALUES           = [1e-4, 3e-4, 1e-3]
WEIGHT_DECAY_VALUES = [0.0,  1e-4, 1e-3]
# ─────────────────────────────────────────────────────────────────────────────


def train_and_eval(lr: float, weight_decay: float, device: str) -> float:
    """Train on original train split, return best val MAE."""
    train_ds = MATWIDataset(DATA_ROOT, split="train")
    val_ds   = MATWIDataset(DATA_ROOT, split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model     = build_resnet18_regressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_val_mae = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for images, targets in train_loader:
            images  = images.to(device)
            targets = targets.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        preds_list, targets_list = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                preds_list.append(model(images.to(device)).squeeze(1))
                targets_list.append(targets.to(device))

        val_mae_v = mae(torch.cat(preds_list), torch.cat(targets_list))
        scheduler.step(val_mae_v)

        if val_mae_v < best_val_mae:
            best_val_mae = val_mae_v

    return best_val_mae


def run():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device : {device}")
    RESULTS_DIR.mkdir(exist_ok=True)

    grid    = list(product(LR_VALUES, WEIGHT_DECAY_VALUES))
    results = []

    for i, (lr, wd) in enumerate(grid, 1):
        print(f"\n[{i}/{len(grid)}]  lr={lr:.0e}  weight_decay={wd:.0e}")
        val_mae_v = train_and_eval(lr, wd, device)
        results.append({"lr": lr, "weight_decay": wd, "val_mae": round(val_mae_v, 4)})
        print(f"  → val MAE = {val_mae_v:.2f} µm")

    best = min(results, key=lambda r: r["val_mae"])
    print(f"\nBest config: lr={best['lr']:.0e}  weight_decay={best['weight_decay']:.0e}  val_mae={best['val_mae']:.2f} µm")

    out = {"best": best, "all": results}
    with open(RESULTS_DIR / "grid_search.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {RESULTS_DIR / 'grid_search.json'}")


if __name__ == "__main__":
    run()
