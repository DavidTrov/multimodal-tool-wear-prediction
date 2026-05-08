"""
Phase 4 — sensor-only CNN on CWT scalograms.

Trains a small CNN directly on (5, 64, 64) CWT scalograms to predict tool
wear in µm.  Supports two architectures and two optimizers for A/B testing:

Architectures
-------------
  baseline    : 4-block VGG-style CNN, 61K params  [default]
  multiscale  : inception-style multiscale pyramid, 169K params [Zhang 2023]

Optimizers
----------
  adam  : Adam, lr=1e-3  [default]
  sgdm  : SGD + momentum=0.9 + weight_decay=5e-3, lr=1e-3  [Zhang 2023 §4.1]

Checkpoints are namespaced so both architectures can coexist:
    checkpoints/phase4_baseline_best.pt
    checkpoints/phase4_multiscale_best.pt

Pre-requisite
-------------
    python experiments/phase3_fusion/precompute_scalograms.py --force

Usage
-----
    python experiments/phase4_sensor_cnn/train.py
    python experiments/phase4_sensor_cnn/train.py --arch multiscale --optim sgdm
    python experiments/phase4_sensor_cnn/train.py --arch multiscale --optim sgdm --resume

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
from src.models.sensor_cnn_model import SensorCNNRegressor
from src.models.multiscale_sensor_cnn import MultiScaleSensorCNN
from src.utils.metrics import mae

# ── Config ────────────────────────────────────────────────────────────────────
SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
CKPT_DIR      = ROOT / "checkpoints"
RESULTS_DIR   = Path(__file__).parent / "results"

LR          = 1e-3   # 0.001 for SGDM; 0.01 caused explosion on this 647-sample dataset
BATCH_SIZE  = 16
EPOCHS      = 100
NUM_WORKERS = 0

ARCH_REGISTRY = {
    "baseline":   SensorCNNRegressor,
    "multiscale": MultiScaleSensorCNN,
}
# ──────────────────────────────────────────────────────────────────────────────


def build_optimizer(name: str, model: nn.Module) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=LR)
    if name == "sgdm":
        # Zhang 2023 §4.1: lr=1e-3, momentum=0.9, lr_decay=0.005 (L2)
        return torch.optim.SGD(
            model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-3
        )
    raise ValueError(f"Unknown optimizer: {name!r}. Choose 'adam' or 'sgdm'.")


def run(arch: str, optim_name: str, resume: bool):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device    : {device}")
    print(f"Arch      : {arch}")
    print(f"Optimizer : {optim_name}\n")

    if not SCALOGRAM_DIR.exists() or not any(SCALOGRAM_DIR.glob("*.pt")):
        sys.exit(
            "Scalogram directory is empty.\n"
            "Run first:  python experiments/phase3_fusion/precompute_scalograms.py --force"
        )

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = MATWISensorScalogramDataset(SCALOGRAM_DIR, FEATURES_PATH, split="train")
    val_ds   = MATWISensorScalogramDataset(SCALOGRAM_DIR, FEATURES_PATH, split="val")
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cls = ARCH_REGISTRY[arch]
    model     = model_cls().to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {n_params:,}")
    print(f"INT8 size  : {n_params / 1024:.1f} KB\n")

    # ── Optimisation ──────────────────────────────────────────────────────────
    optimizer = build_optimizer(optim_name, model)
    # Cosine annealing: decays LR smoothly from peak to eta_min over all epochs.
    # Better than ReduceLROnPlateau for SGDM — avoids premature LR collapse
    # before the model has explored the flat loss basin.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )
    criterion = nn.MSELoss()

    CKPT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    ckpt_path    = CKPT_DIR / f"phase4_{arch}_{optim_name}_best.pt"
    history_path = RESULTS_DIR / f"history_{arch}_{optim_name}.json"
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

        scheduler.step()
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
    parser.add_argument(
        "--arch", choices=list(ARCH_REGISTRY), default="baseline",
        help="Model architecture (default: baseline)",
    )
    parser.add_argument(
        "--optim", choices=["adam", "sgdm"], default="adam",
        help="Optimizer (default: adam)",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(arch=args.arch, optim_name=args.optim, resume=args.resume)
