"""
Phase 3b – test-set evaluation for the CWT scalogram fusion model.

Usage:
    python experiments/phase3_fusion/evaluate_scalogram.py
"""

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

DATA_ROOT     = ROOT / "data" / "raw"
SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
CKPT_PATH     = ROOT / "checkpoints" / "phase3b_scalogram_best.pt"
RESULTS_DIR   = Path(__file__).parent / "results_scalogram"
BATCH_SIZE    = 16
NUM_WORKERS   = 0


def load_model(device):
    model = ScalogramFusionModel().to(device)
    # Swap fc → Identity (same as in training) before loading checkpoint
    model.image_encoder.fc = nn.Identity()
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    return model


def evaluate(split: str, model, device):
    ds     = MATWIScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, split=split)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for images, scalograms, targets in loader:
            images     = images.to(device)
            scalograms = scalograms.to(device)
            preds      = model(images, scalograms).squeeze(1).cpu()
            all_preds.append(preds)
            all_targets.append(targets)

    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    errors  = (preds - targets).abs()

    return {
        "split":     split,
        "n_samples": len(ds),
        "mae":       round(errors.mean().item(), 2),
        "mae_std":   round(errors.std().item(), 2),
        "mae_min":   round(errors.min().item(), 2),
        "mae_max":   round(errors.max().item(), 2),
    }


def run():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    if not CKPT_PATH.exists():
        sys.exit(f"Checkpoint not found: {CKPT_PATH}\nRun train_scalogram.py first.")

    model = load_model(device)
    print(f"Loaded: {CKPT_PATH}\n")

    # Show head weight norms per modality
    W = model.head.weight.data.cpu().squeeze()
    print(f"Head weight norms — image(512d): {W[:512].norm():.4f}  "
          f"sensor(64d): {W[512:].norm():.4f}\n")

    results = {}
    for split in ("train", "val", "test"):
        r = evaluate(split, model, device)
        results[split] = r
        print(
            f"{split:5s}  n={r['n_samples']:4d}  "
            f"MAE={r['mae']:.2f} ± {r['mae_std']:.2f} µm  "
            f"(min={r['mae_min']:.2f}, max={r['mae_max']:.2f})"
        )

    print(f"\n─── Baselines ───────────────────────────────")
    print(f"Phase 1 image-only  test MAE : 23.17 µm")
    print(f"Phase 2 sensor-only test MAE : 28.25 µm")
    print(f"Paper baseline               : 19.00 µm")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    run()
