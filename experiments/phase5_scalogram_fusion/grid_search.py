"""
Phase 5 — grid search over LR × weight_decay on the original train/val split.
Uses MultiScaleFusionModel with frozen encoders + SGDM + HuberLoss(δ=20).

Run from the thesis root:
    python experiments/phase5_scalogram_fusion/grid_search.py

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

BATCH_SIZE  = 16
EPOCHS      = 60
AUX_LAMBDA  = 0.2
NUM_WORKERS = 0

LR_VALUES           = [1e-3, 3e-3, 5e-3]
WEIGHT_DECAY_VALUES = [1e-3, 5e-3, 1e-2]
# ─────────────────────────────────────────────────────────────────────────────


def build_model(device: str) -> MultiScaleFusionModel:
    model = MultiScaleFusionModel()
    model.load_phase1_weights(PHASE1_CKPT, device=device)
    model.load_phase4_weights(PHASE4_CKPT, device=device)
    model.freeze_image_encoder()
    model.freeze_sensor_encoder()
    return model.to(device)


def train_and_eval(lr: float, weight_decay: float, device: str) -> float:
    train_ds = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, split="train")
    val_ds   = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model     = build_model(device)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, momentum=0.9, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.HuberLoss(delta=20.0)

    best_val_mae = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
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
        scheduler.step()

        model.eval()
        preds_list, targets_list = [], []
        with torch.no_grad():
            for images, scalograms, targets in val_loader:
                preds_list.append(model(images.to(device), scalograms.to(device))[0].squeeze(1))
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

    for p in (PHASE1_CKPT, PHASE4_CKPT):
        if not p.exists():
            sys.exit(f"Required checkpoint not found: {p}")

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
