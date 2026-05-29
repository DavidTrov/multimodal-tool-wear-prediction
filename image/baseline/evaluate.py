"""
Phase 1 – test set evaluation.

Usage:
    python experiments/phase1_image_only/evaluate.py

Run from the thesis root so that `src/` is on the path.
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from image.baseline.dataset import MATWIDataset
from image.baseline.model import build_resnet18_regressor

DATA_ROOT   = ROOT / "data" / "raw"
CKPT_PATH   = ROOT / "checkpoints" / "phase1_best.pt"
RESULTS_DIR = Path(__file__).parent / "results"
BATCH_SIZE  = 16
NUM_WORKERS = 4


def evaluate(split: str, model, device):
    ds     = MATWIDataset(DATA_ROOT, split=split)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images  = images.to(device)
            preds   = model(images).squeeze(1).cpu()
            all_preds.append(preds)
            all_targets.append(targets)

    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    errors  = (preds - targets).abs()

    return {
        "split":        split,
        "n_samples":    len(ds),
        "mae":          round(errors.mean().item(), 2),
        "mae_std":      round(errors.std().item(), 2),
        "mae_min":      round(errors.min().item(), 2),
        "mae_max":      round(errors.max().item(), 2),
    }


def run():
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = build_resnet18_regressor().to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    print(f"Loaded checkpoint: {CKPT_PATH}\n")

    results = {}
    for split in ("train", "val", "test"):
        r = evaluate(split, model, device)
        results[split] = r
        print(f"{split:5s}  n={r['n_samples']:4d}  MAE={r['mae']:.2f} ± {r['mae_std']:.2f} µm  (min={r['mae_min']:.2f}, max={r['mae_max']:.2f})")

    print(f"\nPaper baseline: 19 µm")

    out = RESULTS_DIR / "eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    run()
