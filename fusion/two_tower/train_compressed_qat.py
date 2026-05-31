"""
Phase 5 — fusion with QAT INT8 compressed image encoder (2M-param budget ResNet).

Identical to train_compressed.py except:
  - Image encoder : QAT INT8 quantized pruned ResNet (resnet_qat_int8_2m.pt)
                    Standalone test MAE: 19.06 um (INT8, CPU-only)
  - Device        : always CPU — QAT INT8 models cannot run on MPS/CUDA
  - Checkpoint    : fusion/two_tower/checkpoints/phase5_compressed_qat_fusion_best.pt
  - Results       : fusion/two_tower/results_compressed_qat/

The sensor encoder and fusion head are identical to train_compressed.py.
Only the image encoder changes. The image encoder is frozen, so INT8
quantization only affects the frozen feature extraction, not training.

Usage
-----
    python fusion/two_tower/train_compressed_qat.py
    python fusion/two_tower/train_compressed_qat.py --resume

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

from fusion.two_tower.dataset import MATWIFusionScalogramDataset
from fusion.two_tower.model import MultiScaleFusionModel
from src.metrics import mae

# QAT INT8 models require CPU + qnnpack
torch.backends.quantized.engine = "qnnpack"

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT       = ROOT / "data" / "raw"
SCALOGRAM_DIR   = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH   = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
RESULTS_DIR     = Path(__file__).parent / "results_compressed_qat"

COMPRESSED_CKPT = ROOT / "image" / "compression" / "checkpoints" / "resnet_qat_int8_2m.pt"
PHASE4_CKPT     = ROOT / "sensor" / "multiscale" / "checkpoints" / "phase4_multiscale_sgdm_best.pt"
CKPT_PATH       = Path(__file__).parent / "checkpoints" / "phase5_compressed_qat_fusion_best.pt"

# Compressed ResNet avgpool output dimensionality (same architecture as distilled 2m)
IMAGE_FEAT_DIM = 309

LR           = 1e-3
WEIGHT_DECAY = 5e-3
BATCH_SIZE   = 16
EPOCHS       = 40
AUX_LAMBDA   = 0.2
NUM_WORKERS  = 0
# ─────────────────────────────────────────────────────────────────────────────


def run(resume: bool = False):
    # QAT INT8 models are CPU-only — quantized ops require qnnpack (CPU)
    device = "cpu"
    print(f"Device : {device}  (forced — QAT INT8 image encoder is CPU-only)\n")

    for p in (COMPRESSED_CKPT, PHASE4_CKPT):
        if not p.exists():
            sys.exit(f"Required checkpoint not found: {p}")

    CKPT_PATH.parent.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, "train")
    val_ds   = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, "val")
    print(f"Train samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MultiScaleFusionModel(image_feat_dim=IMAGE_FEAT_DIM)
    model.load_compressed_image_encoder(COMPRESSED_CKPT, device=device)
    model.load_phase4_weights(PHASE4_CKPT, device=device)
    model.freeze_image_encoder()
    model.freeze_sensor_encoder()
    model = model.to(device)

    n_frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen params    : {n_frozen:,}  (QAT INT8 image encoder + sensor encoder)")
    print(f"Trainable params : {n_trainable:,}  (fusion head + aux head + LayerNorms)")
    print(f"Image feat dim   : {IMAGE_FEAT_DIM}  (compressed, vs 512 standard)\n")

    # ── Optimisation ──────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
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
        print(f"Resumed from epoch {start_epoch - 1}  |  Best so far: {best_val_mae:.2f} um\n")

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
            f"val_mae={val_mae_v:.2f} +/- {val_mae_std:.2f} um  "
            f"lr={current_lr:.2e}"
        )

        if val_mae_v < best_val_mae:
            best_val_mae = val_mae_v
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  New best: {best_val_mae:.2f} um")

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nBest val MAE : {best_val_mae:.2f} um")
    print(f"Checkpoint   : {CKPT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(resume=args.resume)
