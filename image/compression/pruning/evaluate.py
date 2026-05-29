"""
Compression Phase 1 – test set evaluation for the pruned model.

Usage:
    python experiments/compression/phase1_pruning/evaluate.py

Run from the thesis root.
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from image.baseline.dataset import MATWIDataset

DATA_ROOT   = ROOT / "data" / "raw"
CKPT_PATH   = ROOT / "checkpoints" / "distilled.pt"
RESULTS_DIR = Path(__file__).parent / "results"
BATCH_SIZE  = 16
NUM_WORKERS = 0


def evaluate(split: str, model, device):
    ds     = MATWIDataset(DATA_ROOT, split=split)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            preds  = model(images).squeeze(1).cpu()
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

    if not CKPT_PATH.exists():
        sys.exit(f"Checkpoint not found: {CKPT_PATH}\nRun phase1_pruning/train.py first.")

    # pruned.pt is saved as a full model object (not state_dict)
    # weights_only=False required because the checkpoint contains a full model object
    model = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded checkpoint : {CKPT_PATH}")
    print(f"Parameters        : {n_params:,}")
    print(f"INT8 size         : {n_params / 1024:.1f} KB")
    print(f"INT4 size (est)   : {n_params * 0.5 / 1024:.1f} KB\n")

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
    print(f"Phase 1 image-only (unpruned)  : 23.17 µm")
    print(f"Paper baseline                 : 19.00 µm")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    run()
