"""
Phase 5 — improved scalogram fusion (ResNet18 + MultiScaleSensorCNN).

Key differences from Phase 3b:
  - Sensor branch: MultiScaleSensorCNN (CBAM, ResBlock, GroupNorm, ~244K params)
    vs old SensorCNN (BatchNorm, plain VGG, 61K params)
  - Sensor branch warm-started from Phase-4 checkpoint (not random init)
  - Two SGDM param groups: sensor encoder at LR=2e-4, head at LR=1e-3
  - HuberLoss(δ=20) matches Phase-4 success
  - Scalograms include aircut gating + force HPF (recomputed)

Usage
-----
    python experiments/phase5_scalogram_fusion/train.py
    python experiments/phase5_scalogram_fusion/train.py --resume

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

from src.data.fusion_scalogram_dataset import MATWIFusionScalogramDataset
from src.models.multiscale_fusion_model import MultiScaleFusionModel
from src.utils.metrics import mae

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT     = ROOT / "data" / "raw"
SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
CKPT_DIR      = ROOT / "checkpoints"
RESULTS_DIR   = Path(__file__).parent / "results"

PHASE1_CKPT = CKPT_DIR / "phase1_best.pt"
PHASE4_CKPT = CKPT_DIR / "phase4_multiscale_sgdm_best.pt"
CKPT_PATH   = CKPT_DIR / "phase5_multiscale_fusion_best.pt"

LR           = 3e-3   # fusion head + LayerNorms (only trainable params)
WEIGHT_DECAY = 5e-3
BATCH_SIZE   = 16
EPOCHS       = 60
AUX_LAMBDA   = 0.2    # weight of auxiliary sensor loss
NUM_WORKERS  = 0
# ─────────────────────────────────────────────────────────────────────────────


def run(resume: bool = False):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device : {device}\n")

    for p in (PHASE1_CKPT, PHASE4_CKPT):
        if not p.exists():
            sys.exit(f"Required checkpoint not found: {p}")

    CKPT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, "train")
    val_ds   = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, "val")
    print(f"Train samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MultiScaleFusionModel()
    model.load_phase1_weights(PHASE1_CKPT, device=device)
    model.load_phase4_weights(PHASE4_CKPT, device=device)
    model.freeze_image_encoder()
    model.freeze_sensor_encoder()
    model = model.to(device)

    n_frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen params    : {n_frozen:,}  (image encoder + sensor encoder)")
    print(f"Trainable params : {n_trainable:,}  (fusion head + aux head + LayerNorms)\n")

    # ── Optimisation — only head + LayerNorms ──────────────────────────────────
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )
    criterion = nn.HuberLoss(delta=20.0)

    history_path = RESULTS_DIR / "history.json"
    start_epoch  = 1
    best_val_mae = float("inf")
    history      = []

    if resume and CKPT_PATH.exists():
        model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            start_epoch  = history[-1]["epoch"] + 1
            best_val_mae = min(h["val_mae"] for h in history)
        print(f"Resumed from epoch {start_epoch - 1}  |  Best so far: {best_val_mae:.2f} µm\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, start_epoch + EPOCHS):

        model.train()
        train_loss = 0.0
        for images, scalograms, targets in train_loader:
            images     = images.to(device)
            scalograms = scalograms.to(device)
            targets    = targets.to(device).unsqueeze(1)

            optimizer.zero_grad()
            p_final, p_aux = model(images, scalograms)
            loss = criterion(p_final, targets) + AUX_LAMBDA * criterion(p_aux, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(images)

        train_loss /= len(train_ds)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, scalograms, targets in val_loader:
                images     = images.to(device)
                scalograms = scalograms.to(device)
                targets    = targets.to(device)
                preds      = model(images, scalograms)[0].squeeze(1)
                all_preds.append(preds)
                all_targets.append(targets)

        all_preds   = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        val_mae_v   = mae(all_preds, all_targets)
        val_mae_std = (all_preds - all_targets).abs().std().item()

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch":       epoch,
            "train_loss":  round(train_loss, 4),
            "val_mae":     round(val_mae_v, 4),
            "val_mae_std": round(val_mae_std, 4),
            "lr":          current_lr,
        })
        print(
            f"Epoch {epoch:3d}  "
            f"train_loss={train_loss:.2f}  "
            f"val_mae={val_mae_v:.2f} ± {val_mae_std:.2f} µm  "
            f"lr={current_lr:.2e}"
        )

        if val_mae_v < best_val_mae:
            best_val_mae = val_mae_v
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  ✓ New best: {best_val_mae:.2f} µm")

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nBest val MAE : {best_val_mae:.2f} µm")
    print(f"Checkpoint   : {CKPT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(resume=args.resume)
