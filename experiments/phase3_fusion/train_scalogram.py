"""
Phase 3b – multimodal fusion with CWT scalograms  (mid-level fusion).

Architecture
    Image encoder  : frozen ResNet18 conv layers (Phase-1 weights) → 512-d
    Sensor encoder : SensorCNN on (5, 64, 64) CWT scalogram        →  64-d
    Head           : LayerNorm × 2 → concat(576) → Linear(576 → 1)

Both feature vectors enter the shared head from the very first step, so the
sensor encoder receives a gradient immediately and does not have to overcome
an initialisation bias toward the image branch.

Steps before running
--------------------
1. python experiments/phase3_fusion/precompute_scalograms.py
2. python experiments/phase3_fusion/train_scalogram.py [--resume]

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
RESULTS_DIR    = Path(__file__).parent / "results_scalogram"

LR          = 1e-3
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

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ScalogramFusionModel().to(device)

    # Load Phase-1 weights into image encoder, then swap fc → Identity
    phase1_ckpt = CKPT_DIR / "phase1_best.pt"
    if phase1_ckpt.exists():
        model.load_phase1_weights(phase1_ckpt, device=device)
        print(f"Loaded Phase-1 weights from {phase1_ckpt}")
        print("Image encoder fc replaced with Identity (512-d feature extractor)")
    else:
        # No Phase-1 checkpoint: still remove the fc head
        model.image_encoder.fc = nn.Identity()
        print("Phase-1 checkpoint not found — using ImageNet weights, fc → Identity")

    # Freeze image encoder conv layers
    model.freeze_image_encoder()
    print("Image encoder frozen")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable:,}")

    # ── Optimisation ──────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )
    criterion = nn.MSELoss()

    CKPT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    ckpt_path    = CKPT_DIR / "phase3b_scalogram_best.pt"
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

        with open(RESULTS_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nBest val MAE : {best_val_mae:.2f} µm")
    print(f"Checkpoint   : {ckpt_path}")

    # Show head weights split by modality to understand their contributions
    W = model.head.weight.data.cpu().squeeze()   # (576,)
    w_img_norm    = W[:512].norm().item()
    w_sensor_norm = W[512:].norm().item()
    print(f"\nHead weight norms — image(512d): {w_img_norm:.4f}  "
          f"sensor(64d): {w_sensor_norm:.4f}")
    print("(Higher sensor norm → sensor features are actively used)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(resume=args.resume)
