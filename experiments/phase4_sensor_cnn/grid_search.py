"""
Phase 4 — grid search over LR × weight_decay on the original train/val split.
Uses MultiScaleSensorCNN + SGDM + HuberLoss(δ=20) (best known config).

Run from the thesis root:
    python experiments/phase4_sensor_cnn/grid_search.py

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

from src.data.sensor_scalogram_dataset import MATWISensorScalogramDataset
from src.models.multiscale_sensor_cnn import MultiScaleSensorCNN
from src.utils.metrics import mae

# ── Config ────────────────────────────────────────────────────────────────────
SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
CKPT_DIR      = ROOT / "checkpoints"
RESULTS_DIR   = Path(__file__).parent / "results"

BATCH_SIZE  = 16
EPOCHS      = 100
NUM_WORKERS = 0

LR_VALUES           = [5e-4, 1e-3, 3e-3]
WEIGHT_DECAY_VALUES = [1e-3, 5e-3, 1e-2]
# ─────────────────────────────────────────────────────────────────────────────


def train_and_eval(lr: float, weight_decay: float, device: str) -> float:
    train_ds = MATWISensorScalogramDataset(SCALOGRAM_DIR, FEATURES_PATH, split="train")
    val_ds   = MATWISensorScalogramDataset(SCALOGRAM_DIR, FEATURES_PATH, split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model     = MultiScaleSensorCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.HuberLoss(delta=20.0)

    best_val_mae = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for scalograms, targets in train_loader:
            scalograms = scalograms.to(device)
            targets    = targets.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(scalograms), targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        preds_list, targets_list = [], []
        with torch.no_grad():
            for scalograms, targets in val_loader:
                preds_list.append(model(scalograms.to(device)).squeeze(1))
                targets_list.append(targets.to(device))

        val_mae_v = mae(torch.cat(preds_list), torch.cat(targets_list))
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
