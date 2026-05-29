"""
Phase 4 — grid search for MultiScaleSensorCNN (sensor-only).

Searches over LR × weight_decay × momentum × head_architecture using SGDM.
Val MAE is recorded at every epoch so the best epoch is found per config.
Uses the original paper train/val/test split (sets unchanged).

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

from sensor.cnn.dataset import MATWISensorScalogramDataset
from sensor.multiscale.model import MultiScaleSensorCNN
from src.metrics import mae

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

# Head architectures: list of hidden dims between the 96-dim features and the
# final output.  [] means no hidden layer (current baseline).
HEAD_CONFIGS = {
    "linear":   [],          # Dropout → Linear(96→1)                  97 params
    "small":    [48],        # Dropout → Linear(96→48) → ReLU → Linear(48→1)   4,705 params
    "medium":   [64, 32],    # Dropout → 96→64 → ReLU → 64→32 → ReLU → 32→1   8,353 params
}

# ── Fixed training config ─────────────────────────────────────────────────────
BATCH_SIZE  = 16
MAX_EPOCHS  = 20     # val MAE recorded every epoch; best epoch reported per config
DROPOUT     = 0.3
NUM_WORKERS = 0
# ─────────────────────────────────────────────────────────────────────────────


def build_head(hidden_dims: list[int], dropout: float = DROPOUT) -> nn.Sequential:
    """Build regression head: Dropout → [Linear→ReLU]* → Linear(→1)."""
    dims   = [96] + hidden_dims + [1]
    layers = [nn.Dropout(dropout)]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:          # ReLU between hidden layers, not after last
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def train_and_eval(
    lr:           float,
    weight_decay: float,
    momentum:     float,
    head_name:    str,
    device:       str,
) -> tuple[float, int, dict]:
    """
    Train on paper train split, evaluate on paper val split every epoch.
    Returns (best_val_mae, best_epoch, state_dict_of_best_epoch).
    """
    train_ds = MATWISensorScalogramDataset(SCALOGRAM_DIR, FEATURES_PATH, split="train")
    val_ds   = MATWISensorScalogramDataset(SCALOGRAM_DIR, FEATURES_PATH, split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model      = MultiScaleSensorCNN().to(device)
    model.head = build_head(HEAD_CONFIGS[head_name]).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-5,
    )
    criterion = nn.HuberLoss(delta=20.0)

    best_val_mae = float("inf")
    best_epoch   = 0
    best_state   = None

    for epoch in range(1, MAX_EPOCHS + 1):
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
            best_epoch   = epoch
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_val_mae, best_epoch, best_state


def run():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    CKPT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    head_names = list(HEAD_CONFIGS.keys())
    grid = list(product(LR_VALUES, WEIGHT_DECAY_VALUES, MOMENTUM_VALUES, head_names))
    n    = len(grid)

    print(f"Device     : {device}")
    print(f"Configs    : {n}  ({len(LR_VALUES)} LR × {len(WEIGHT_DECAY_VALUES)} WD × "
          f"{len(MOMENTUM_VALUES)} mom × {len(head_names)} heads)")
    print(f"Max epochs : {MAX_EPOCHS} per config  →  {n * MAX_EPOCHS:,} total epochs")
    print(f"Heads      : {head_names}\n")

    results            = []
    overall_best_mae   = float("inf")
    overall_best_cfg   = None
    overall_best_state = None
    t_start            = time.time()

    for i, (lr, wd, mom, head) in enumerate(grid, 1):
        t0 = time.time()
        print(f"[{i:2d}/{n}]  lr={lr:.0e}  wd={wd:.0e}  mom={mom:.2f}  head={head:8s}", end="  ", flush=True)

        val_mae_v, best_ep, state = train_and_eval(lr, wd, mom, head, device)

        elapsed   = time.time() - t0
        remaining = (time.time() - t_start) / i * (n - i)
        print(f"val_mae={val_mae_v:.2f} µm  best_ep={best_ep:2d}  "
              f"({elapsed:.0f}s,  ~{remaining/60:.0f} min left)")

        entry = {
            "lr": lr, "weight_decay": wd, "momentum": mom, "head": head,
            "best_epoch": best_ep, "val_mae": round(val_mae_v, 4),
        }
        results.append(entry)

        if val_mae_v < overall_best_mae:
            overall_best_mae   = val_mae_v
            overall_best_cfg   = entry
            overall_best_state = state
            torch.save(overall_best_state, BEST_CKPT)
            print(f"         ✓ New best: {overall_best_mae:.2f} µm  → saved {BEST_CKPT.name}")

    results.sort(key=lambda r: r["val_mae"])

    print(f"\n{'='*60}")
    print(f"Best config:")
    for k, v in overall_best_cfg.items():
        print(f"  {k:15s} = {v}")
    print(f"  checkpoint    → {BEST_CKPT}")
    print(f"{'='*60}")

    output = {
        "best":       overall_best_cfg,
        "all":        results,
        "max_epochs": MAX_EPOCHS,
        "optimizer":  "sgdm",
        "head_configs": {k: v for k, v in HEAD_CONFIGS.items()},
    }
    out_path = RESULTS_DIR / "grid_search.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results → {out_path}")


if __name__ == "__main__":
    run()
