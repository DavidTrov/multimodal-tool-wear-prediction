"""
Phase 4 — grid search for MultiScaleSensorCNN (sensor-only).

Searches over LR × weight_decay × batch_size × momentum using SGDM.
Uses the original paper train/val/test split (sets unchanged).
The best-config model checkpoint is saved to checkpoints/.

Run from the thesis root:
    python experiments/phase4_sensor_cnn/grid_search.py

Saves:
    results/grid_search.json          — all configs + best config
    checkpoints/phase4_gs_best.pt     — weights for the best config
"""

import json
import sys
import time
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

# ── Paths ─────────────────────────────────────────────────────────────────────
SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
CKPT_DIR      = ROOT / "checkpoints"
RESULTS_DIR   = Path(__file__).parent / "results"

BEST_CKPT = CKPT_DIR / "phase4_gs_best.pt"

# ── Search space ──────────────────────────────────────────────────────────────
LR_VALUES           = [5e-4, 1e-3, 3e-3]
WEIGHT_DECAY_VALUES = [1e-3, 5e-3, 1e-2]
MOMENTUM_VALUES     = [0.85, 0.90, 0.95]

# ── Fixed training config ─────────────────────────────────────────────────────
BATCH_SIZE  = 16     # fixed — least impactful param on this dataset size
EPOCHS      = 40     # shortened vs full 100 for grid-search speed
NUM_WORKERS = 0
# ─────────────────────────────────────────────────────────────────────────────


def train_and_eval(
    lr:           float,
    weight_decay: float,
    momentum:     float,
    device:       str,
) -> tuple[float, dict]:
    """
    Train on paper train split, evaluate on paper val split.
    Returns (best_val_mae, state_dict_of_best_epoch).
    """
    train_ds = MATWISensorScalogramDataset(SCALOGRAM_DIR, FEATURES_PATH, split="train")
    val_ds   = MATWISensorScalogramDataset(SCALOGRAM_DIR, FEATURES_PATH, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    model = MultiScaleSensorCNN().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )
    criterion = nn.HuberLoss(delta=20.0)

    best_val_mae   = float("inf")
    best_state     = None

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
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_val_mae, best_state


def run():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    CKPT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    grid = list(product(LR_VALUES, WEIGHT_DECAY_VALUES, MOMENTUM_VALUES))
    n    = len(grid)
    print(f"Device  : {device}")
    print(f"Configs : {n}  ({len(LR_VALUES)} LR × {len(WEIGHT_DECAY_VALUES)} WD × "
          f"{len(MOMENTUM_VALUES)} momentum)  batch_size={BATCH_SIZE} fixed")
    print(f"Epochs  : {EPOCHS} per config  →  {n * EPOCHS:,} total epochs\n")

    results            = []
    overall_best_mae   = float("inf")
    overall_best_cfg   = None
    overall_best_state = None
    t_start            = time.time()

    for i, (lr, wd, mom) in enumerate(grid, 1):
        t0 = time.time()
        print(f"[{i:2d}/{n}]  lr={lr:.0e}  wd={wd:.0e}  mom={mom:.2f}", end="  ", flush=True)

        val_mae_v, state = train_and_eval(lr, wd, mom, device)

        elapsed = time.time() - t0
        remaining = (time.time() - t_start) / i * (n - i)
        print(f"val_mae={val_mae_v:.2f} µm  ({elapsed:.0f}s,  ~{remaining/60:.0f} min remaining)")

        entry = {
            "lr": lr, "weight_decay": wd, "momentum": mom,
            "val_mae": round(val_mae_v, 4),
        }
        results.append(entry)

        if val_mae_v < overall_best_mae:
            overall_best_mae   = val_mae_v
            overall_best_cfg   = entry
            overall_best_state = state
            torch.save(overall_best_state, BEST_CKPT)
            print(f"        ✓ New best overall: {overall_best_mae:.2f} µm  → saved {BEST_CKPT.name}")

    # Sort for readability
    results.sort(key=lambda r: r["val_mae"])

    print(f"\n{'='*60}")
    print(f"Best config:")
    print(f"  lr           = {overall_best_cfg['lr']:.0e}")
    print(f"  weight_decay = {overall_best_cfg['weight_decay']:.0e}")
    print(f"  batch_size   = {overall_best_cfg['batch_size']}")
    print(f"  momentum     = {overall_best_cfg['momentum']}")
    print(f"  val MAE      = {overall_best_cfg['val_mae']:.2f} µm")
    print(f"  checkpoint   → {BEST_CKPT}")
    print(f"{'='*60}")

    output = {
        "best":   overall_best_cfg,
        "all":    results,
        "epochs": EPOCHS,
        "optimizer": "sgdm",
    }
    out_path = RESULTS_DIR / "grid_search.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results → {out_path}")


if __name__ == "__main__":
    run()
