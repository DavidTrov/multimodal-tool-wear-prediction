"""
Phase 4 — sensor-only CNN on CWT scalograms.

Trains a small CNN directly on (5, 64, 64) CWT scalograms to predict tool
wear in µm.  This is the deployment-track model: small enough to fit on
STM32F401RC (256 KB flash, 64 KB SRAM) and compatible with X-CUBE-AI
(neural-network-only runtime).

Architectures
-------------
  --arch simple      : 4-block VGG stack (~61 K params)
  --arch multiscale  : Inception-style multiscale feature pyramid
                       adapted from Zhang et al. 2023 (~28 K params, default)

Augmentation
------------
SpecAugment-style augmentation is applied to the training split: cyclic
time shift, frequency masking, time masking, and occasional channel drop.
See src/data/sensor_scalogram_dataset.py for parameters.

Pre-requisite
-------------
    python experiments/phase3_fusion/precompute_scalograms.py

Usage
-----
    python experiments/phase4_sensor_cnn/train.py                          # multiscale + aug (default)
    python experiments/phase4_sensor_cnn/train.py --arch simple            # simple stack
    python experiments/phase4_sensor_cnn/train.py --no-augment             # disable augmentation
    python experiments/phase4_sensor_cnn/train.py --resume

Run from the thesis root.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.sensor_scalogram_dataset import MATWISensorScalogramDataset
from src.models.sensor_cnn_model import SensorCNNRegressor, MultiScaleSensorCNN
from src.utils.metrics import mae

# ── Config ────────────────────────────────────────────────────────────────────
SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
CKPT_DIR      = ROOT / "checkpoints"
RESULTS_DIR   = Path(__file__).parent / "results"

LR          = 1e-3
BATCH_SIZE  = 16
EPOCHS      = 120
NUM_WORKERS = 0

ARCH_FACTORY = {
    "simple":      SensorCNNRegressor,
    "multiscale":  MultiScaleSensorCNN,
}
# ──────────────────────────────────────────────────────────────────────────────


def run(arch: str, augment: bool, resume: bool):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device      : {device}")
    print(f"Architecture: {arch}")
    print(f"Augmentation: {augment}\n")

    if not SCALOGRAM_DIR.exists() or not any(SCALOGRAM_DIR.glob("*.pt")):
        sys.exit(
            "Scalogram directory is empty.\n"
            "Run first:  python experiments/phase3_fusion/precompute_scalograms.py"
        )

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = MATWISensorScalogramDataset(
        SCALOGRAM_DIR, FEATURES_PATH, split="train",
        augment=augment,   # True/False; dataset handles default
    )
    val_ds   = MATWISensorScalogramDataset(
        SCALOGRAM_DIR, FEATURES_PATH, split="val", augment=False,
    )
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ARCH_FACTORY[arch]().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {n_params:,}")
    print(f"INT8 size  : {n_params / 1024:.1f} KB")
    print(f"INT4 size  : {n_params * 0.5 / 1024:.1f} KB\n")

    # ── Optimisation ──────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    criterion = nn.MSELoss()

    CKPT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    ckpt_path    = CKPT_DIR / f"phase4_{arch}_best.pt"
    history_path = RESULTS_DIR / f"history_{arch}.json"
    start_epoch  = 1
    best_val_mae = float("inf")
    history      = []

    if resume and ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            start_epoch  = history[-1]["epoch"] + 1
            best_val_mae = min(h["val_mae"] for h in history)
        print(f"Resumed from epoch {start_epoch - 1}  |  Best so far: {best_val_mae:.2f} µm")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, start_epoch + EPOCHS):

        model.train()
        train_loss = 0.0
        for scalograms, targets in train_loader:
            scalograms = scalograms.to(device)
            targets    = targets.to(device).unsqueeze(1)

            optimizer.zero_grad()
            preds = model(scalograms)
            loss  = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(scalograms)

        train_loss /= len(train_ds)

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for scalograms, targets in val_loader:
                scalograms = scalograms.to(device)
                targets    = targets.to(device)
                preds      = model(scalograms).squeeze(1)
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
        print(
            f"Epoch {epoch:3d}  "
            f"train_loss={train_loss:.2f}  "
            f"val_mae={val_mae:.2f} ± {val_mae_std:.2f} µm  "
            f"lr={current_lr:.2e}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ New best: {best_val_mae:.2f} µm")

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nBest val MAE : {best_val_mae:.2f} µm")
    print(f"Checkpoint   : {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",        type=str, default="multiscale",
                        choices=list(ARCH_FACTORY.keys()),
                        help="Architecture (default: multiscale)")
    parser.add_argument("--no-augment",  action="store_true",
                        help="Disable scalogram augmentation")
    parser.add_argument("--resume",      action="store_true")
    args = parser.parse_args()
    run(arch=args.arch, augment=not args.no_augment, resume=args.resume)
