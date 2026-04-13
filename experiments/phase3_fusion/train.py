"""
Phase 3 – multimodal fusion (ResNet18 image + tsfresh sensor features).

Usage:
    python experiments/phase3_fusion/train.py           # fresh start
    python experiments/phase3_fusion/train.py --resume  # continue from checkpoint

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

from src.data.fusion_dataset import MATWIFusionDataset
from src.models.fusion_model import FusionModel
from src.utils.metrics import mae

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT     = ROOT / "data" / "raw"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features.parquet"
CKPT_DIR      = ROOT / "checkpoints"
RESULTS_DIR   = Path(__file__).parent / "results"

LR          = 1e-4
BATCH_SIZE  = 16
EPOCHS      = 60
NUM_WORKERS = 0
# ──────────────────────────────────────────────────────────────────────────────


def run(resume: bool):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Data
    train_ds = MATWIFusionDataset(DATA_ROOT, FEATURES_PATH, split="train")
    val_ds   = MATWIFusionDataset(DATA_ROOT, FEATURES_PATH, split="val")
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Compute sensor feature statistics from training set (used for runtime normalisation)
    all_sensor = torch.stack([train_ds[i][1] for i in range(len(train_ds))])
    sensor_mean = all_sensor.mean(dim=0)
    sensor_std  = all_sensor.std(dim=0)
    print(f"Sensor stats computed from {len(train_ds)} training samples")

    # Model
    model = FusionModel(sensor_mean=sensor_mean, sensor_std=sensor_std).to(device)

    # Bootstrap image encoder from Phase 1 fine-tuned weights
    phase1_ckpt = CKPT_DIR / "phase1_best.pt"
    if phase1_ckpt.exists():
        phase1_state = torch.load(phase1_ckpt, map_location=device)
        # Phase 1 model has an fc head (fc.weight, fc.bias) — skip those,
        # copy everything else (all conv/bn layers) into the image encoder
        encoder_state = model.image_encoder.state_dict()
        transferred = {
            k: v for k, v in phase1_state.items()
            if k in encoder_state and v.shape == encoder_state[k].shape
        }
        encoder_state.update(transferred)
        model.image_encoder.load_state_dict(encoder_state)
        print(f"Transferred {len(transferred)}/{len(phase1_state)} layers from Phase 1 checkpoint")
    else:
        print("Phase 1 checkpoint not found — using ImageNet weights only")

    # Freeze the image encoder — only train sensor branch + fusion head
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    print("Image encoder frozen")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    CKPT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    ckpt_path    = CKPT_DIR / "phase3_best.pt"
    start_epoch  = 1
    best_val_mae = float("inf")
    history      = []

    if resume and ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        history_path = RESULTS_DIR / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            start_epoch  = history[-1]["epoch"] + 1
            best_val_mae = min(h["val_mae"] for h in history)
        print(f"Resumed from {ckpt_path}  |  Best so far: {best_val_mae:.2f} µm")

    for epoch in range(start_epoch, start_epoch + EPOCHS):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for images, sensors, targets in train_loader:
            images  = images.to(device)
            sensors = sensors.to(device)
            targets = targets.to(device).unsqueeze(1)

            optimizer.zero_grad()
            preds = model(images, sensors)
            loss  = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(images)

        train_loss /= len(train_ds)

        # ── Validate ──
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, sensors, targets in val_loader:
                images  = images.to(device)
                sensors = sensors.to(device)
                targets = targets.to(device)
                preds   = model(images, sensors).squeeze(1)
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

        with open(RESULTS_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nBest val MAE: {best_val_mae:.2f} µm")
    print(f"Checkpoint:  {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(resume=args.resume)
