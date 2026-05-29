"""
Phase 4 — test-set evaluation for the sensor-only CWT scalogram CNN.

Usage:
    python experiments/phase4_sensor_cnn/evaluate.py
    python experiments/phase4_sensor_cnn/evaluate.py --arch multiscale --optim sgdm

Run from the thesis root.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sensor.cnn.dataset import MATWISensorScalogramDataset
from sensor.cnn.model import SensorCNNRegressor
from sensor.multiscale.model import MultiScaleSensorCNN

SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
CKPT_DIR      = ROOT / "checkpoints"
RESULTS_DIR   = Path(__file__).parent / "results"
BATCH_SIZE    = 16
NUM_WORKERS   = 0

ARCH_REGISTRY = {
    "baseline":   SensorCNNRegressor,
    "multiscale": MultiScaleSensorCNN,
}


def evaluate(split: str, model, device):
    ds = MATWISensorScalogramDataset(SCALOGRAM_DIR, FEATURES_PATH, split=split)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for scalograms, targets in loader:
            preds = model(scalograms.to(device)).squeeze(1).cpu()
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


def run(arch: str, optim: str):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device : {device}")
    print(f"Arch   : {arch}")
    print(f"Optim  : {optim}")

    ckpt_path = CKPT_DIR / f"phase4_{arch}_{optim}_best.pt"
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}\nRun train.py first.")

    model_cls = ARCH_REGISTRY[arch]
    model     = model_cls().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded : {ckpt_path}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {n_params:,}")
    print(f"INT8 size  : {n_params / 1024:.1f} KB\n")

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
    print(f"Phase 1 image-only  test MAE : 23.17 µm  (cannot deploy)")
    print(f"Phase 2 sensor-only test MAE : 28.25 µm  (XGBoost, cannot deploy)")
    print(f"Paper baseline               : 19.00 µm")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"eval_results_{arch}_{optim}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch", choices=list(ARCH_REGISTRY), default="baseline",
        help="Architecture to evaluate (default: baseline)",
    )
    parser.add_argument(
        "--optim", choices=["adam", "sgdm"], default="adam",
        help="Optimizer used during training (default: adam)",
    )
    args = parser.parse_args()
    run(arch=args.arch, optim=args.optim)
