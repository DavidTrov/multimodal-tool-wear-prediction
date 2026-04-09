"""
Phase 3 – test set evaluation for fusion model.

Usage:
    python experiments/phase3_fusion/evaluate.py
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.fusion_dataset import MATWIFusionDataset
from src.models.fusion_model import FusionModel

DATA_ROOT     = ROOT / "data" / "raw"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features.parquet"
CKPT_PATH     = ROOT / "checkpoints" / "phase3_best.pt"
RESULTS_DIR   = Path(__file__).parent / "results"
BATCH_SIZE    = 16
NUM_WORKERS   = 4


def evaluate(split: str, model, device):
    ds     = MATWIFusionDataset(DATA_ROOT, FEATURES_PATH, split=split)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for images, sensors, targets in loader:
            images  = images.to(device)
            sensors = sensors.to(device)
            preds   = model(images, sensors).squeeze(1).cpu()
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

    model = FusionModel().to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    print(f"Loaded checkpoint: {CKPT_PATH}\n")

    results = {}
    for split in ("train", "val", "test"):
        r = evaluate(split, model, device)
        results[split] = r
        print(f"{split:5s}  n={r['n_samples']:4d}  MAE={r['mae']:.2f} ± {r['mae_std']:.2f} µm  (min={r['mae_min']:.2f}, max={r['mae_max']:.2f})")

    print(f"\nPhase 1 image-only  test MAE: 23.17 µm")
    print(f"Phase 2 sensor-only test MAE: 28.25 µm")
    print(f"Paper baseline:               19.00 µm")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    run()
