"""
Phase 5 — evaluation of the compressed-encoder fusion model.

Loads the best checkpoint from train_compressed.py and evaluates on all
three splits, comparing against the standard fusion and single-modality
baselines.

Usage
-----
    python experiments/phase5_scalogram_fusion/evaluate_compressed.py

Run from the thesis root.
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fusion.two_tower.dataset import MATWIFusionScalogramDataset
from fusion.two_tower.model import MultiScaleFusionModel

DATA_ROOT       = ROOT / "data" / "raw"
SCALOGRAM_DIR   = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH   = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
COMPRESSED_CKPT = ROOT / "image" / "compression" / "checkpoints" / "resnet_distilled_2m.pt"
CKPT_PATH       = ROOT / "fusion" / "two_tower" / "checkpoints" / "phase5_compressed_fusion_best.pt"
RESULTS_DIR     = Path(__file__).parent / "results_compressed"

IMAGE_FEAT_DIM = 309
NUM_WORKERS    = 0


def evaluate(split: str, model: torch.nn.Module, device: str) -> dict:
    ds     = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, split)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=NUM_WORKERS)

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, scalograms, targets in loader:
            images     = images.to(device)
            scalograms = scalograms.to(device)
            targets    = targets.to(device)
            preds      = model(images, scalograms)[0].squeeze(1)
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
        sys.exit(f"Checkpoint not found: {CKPT_PATH}\nRun train_compressed.py first.")

    RESULTS_DIR.mkdir(exist_ok=True)

    model = MultiScaleFusionModel(image_feat_dim=IMAGE_FEAT_DIM)
    # Must install the compressed (pruned) backbone BEFORE loading the state dict,
    # because the checkpoint contains non-standard channel widths [13, 26, 77, ..., 309].
    # A plain ResNet18 has [64, 128, 256, 512] and will fail shape checks.
    model.load_compressed_image_encoder(COMPRESSED_CKPT, device=device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    model = model.to(device)

    n_total    = sum(p.numel() for p in model.parameters())
    n_img_enc  = sum(p.numel() for p in model.image_encoder.parameters())
    n_sen_enc  = sum(p.numel() for p in model.sensor_cnn.parameters())
    print(f"Total parameters      : {n_total:,}")
    print(f"  Image encoder       : {n_img_enc:,}  (compressed ResNet, 309-d)")
    print(f"  Sensor encoder      : {n_sen_enc:,}  (MultiScaleSensorCNN, 96-d)")
    print(f"  Fusion head + norms : {n_total - n_img_enc - n_sen_enc:,}\n")

    results = {}
    for split in ("train", "val", "test"):
        r = evaluate(split, model, device)
        results[split] = r
        print(
            f"{split:5s}  n={r['n_samples']:4d}  "
            f"MAE={r['mae']:.2f} ± {r['mae_std']:.2f} µm  "
            f"(min={r['mae_min']:.2f}, max={r['mae_max']:.2f})"
        )

    print()
    print("── Baselines ──────────────────────────────────────────────────────────")
    print(f"Paper baseline (ResNet50, image-only)              test MAE : 19.00 µm")
    print(f"Phase 5 fusion — standard ResNet18  (n=225)        test MAE : 22.57 µm")
    print(f"Phase 1 image-only — standard ResNet18  (n=247)    test MAE : 23.17 µm")
    print(f"Compressed ResNet (budget, standalone)  (n=247)    test MAE : 20.80 µm")
    print(f"Phase 4 sensor-only (MultiScaleCNN)  (n=247)       test MAE : 24.96 µm")

    out_path = RESULTS_DIR / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    run()
