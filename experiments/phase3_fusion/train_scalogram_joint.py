"""
Phase 3c – joint multimodal training from ImageNet weights (no frozen branches).

Both the image branch (ResNet18) and the sensor branch (SensorCNN) are trained
simultaneously from scratch on the MATWI dataset.  No Phase-1 weights are
loaded — the image branch starts from ImageNet pre-training and fine-tunes
jointly with the sensor branch, exactly replicating the Phase-1 setup for the
image side.

Image branch training matches Phase-1 exactly:
  - ImageNet pre-trained weights (ResNet18_Weights.IMAGENET1K_V1)
  - Adam, lr = 1e-4
  - ReduceLROnPlateau factor=0.5, patience=5
  - MSELoss
  - Training augmentation: RandomHorizontalFlip, RandomVerticalFlip, ColorJitter

Sensor branch uses a higher lr (1e-3) via a separate param group, since it
trains from random initialisation on a different signal type.

Architecture (same as train_scalogram.py — mid-level fusion):
  Image encoder  : ResNet18 (ImageNet init, fully trainable) → 512-d features
  Sensor encoder : SensorCNN on (5, 64, 64) CWT scalogram   →  64-d features
  Fusion         : LayerNorm × 2 → concat(576) → Linear(576 → 1)

Usage
-----
    python experiments/phase3_fusion/train_scalogram_joint.py [--resume]

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

from src.data.scalogram_dataset import MATWIScalogramDataset
from src.models.scalogram_fusion_model import ScalogramFusionModel
from src.utils.metrics import mae

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT      = ROOT / "data" / "raw"
SCALOGRAM_DIR  = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH  = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
CKPT_DIR       = ROOT / "checkpoints"
RESULTS_DIR    = Path(__file__).parent / "results_scalogram_joint"

LR_IMAGE    = 1e-4   # matches Phase-1 exactly
LR_SENSOR   = 1e-3   # higher — sensor branch starts from random init
BATCH_SIZE  = 16
EPOCHS      = 80
NUM_WORKERS = 0
# ──────────────────────────────────────────────────────────────────────────────


def run(resume: bool):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # ── Check scalograms exist ────────────────────────────────────────────────
    if not SCALOGRAM_DIR.exists() or not any(SCALOGRAM_DIR.glob("*.pt")):
        sys.exit(
            "Scalogram directory is empty.\n"
            "Run first:  python experiments/phase3_fusion/precompute_scalograms.py"
        )

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = MATWIScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, split="train")
    val_ds   = MATWIScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, split="val")
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    if len(train_ds) == 0 or len(val_ds) == 0:
        sys.exit("Dataset is empty — check scalogram_dir and features_path.")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # ── Model — no Phase-1 weights, no frozen layers ──────────────────────────
    model = ScalogramFusionModel().to(device)

    # Replace fc with Identity: image encoder outputs 512-d features, not a scalar
    model.image_encoder.fc = nn.Identity()
    print("Image encoder fc replaced with Identity (512-d feature extractor)")
    print("All parameters trainable — joint training from ImageNet init")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable:,}")

    # ── Two param groups: image branch (Phase-1 LR), sensor+head (higher LR) ──
    image_params  = list(model.image_encoder.parameters())
    sensor_params = (
        list(model.sensor_cnn.parameters())
        + list(model.image_norm.parameters())
        + list(model.sensor_norm.parameters())
        + list(model.head.parameters())
    )
    optimizer = torch.optim.Adam([
        {"params": image_params,  "lr": LR_IMAGE},
        {"params": sensor_params, "lr": LR_SENSOR},
    ])

    # Scheduler monitors val MAE — matches Phase-1 (factor=0.5, patience=5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    CKPT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    ckpt_path    = CKPT_DIR / "phase3c_scalogram_joint_best.pt"
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
        print(f"Resumed from epoch {start_epoch - 1}  |  Best so far: {best_val_mae:.2f} µm")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, start_epoch + EPOCHS):

        # Train
        model.train()
        train_loss = 0.0
        for images, scalograms, targets in train_loader:
            images     = images.to(device)
            scalograms = scalograms.to(device)
            targets    = targets.to(device).unsqueeze(1)

            optimizer.zero_grad()
            preds = model(images, scalograms)
            loss  = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(images)

        train_loss /= len(train_ds)

        # Validate
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, scalograms, targets in val_loader:
                images     = images.to(device)
                scalograms = scalograms.to(device)
                targets    = targets.to(device)
                preds      = model(images, scalograms).squeeze(1)
                all_preds.append(preds)
                all_targets.append(targets)

        all_preds   = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        val_mae     = mae(all_preds, all_targets)
        val_mae_std = (all_preds - all_targets).abs().std().item()

        scheduler.step(val_mae)
        lr_image  = optimizer.param_groups[0]["lr"]
        lr_sensor = optimizer.param_groups[1]["lr"]

        history.append({
            "epoch":       epoch,
            "train_loss":  round(train_loss, 4),
            "val_mae":     round(val_mae, 4),
            "val_mae_std": round(val_mae_std, 4),
            "lr_image":    lr_image,
            "lr_sensor":   lr_sensor,
        })
        print(
            f"Epoch {epoch:3d}  "
            f"train_loss={train_loss:.2f}  "
            f"val_mae={val_mae:.2f} ± {val_mae_std:.2f} µm  "
            f"lr_img={lr_image:.2e}  lr_sen={lr_sensor:.2e}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ New best: {best_val_mae:.2f} µm")

        with open(RESULTS_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nBest val MAE : {best_val_mae:.2f} µm")
    print(f"Checkpoint   : {ckpt_path}")

    W = model.head.weight.data.cpu().squeeze()
    print(f"\nHead weight norms — image(512d): {W[:512].norm():.4f}  "
          f"sensor(64d): {W[512:].norm():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(resume=args.resume)
