"""
Phase 5 — evaluation of the QAT INT8 compressed-encoder fusion model.

Loads the best checkpoint from train_compressed_qat.py and evaluates on
all three splits. Counterpart to evaluate_compressed.py but with the
QAT INT8 image encoder (resnet_qat_int8_2m.pt) instead of FP32 distilled.

Device is forced to CPU — QAT INT8 models cannot run on MPS/CUDA.

Usage
-----
    python fusion/two_tower/evaluate_compressed_qat.py

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

# QAT INT8 models require CPU + qnnpack
torch.backends.quantized.engine = "qnnpack"

DATA_ROOT       = ROOT / "data" / "raw"
SCALOGRAM_DIR   = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH   = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
COMPRESSED_CKPT = ROOT / "image" / "compression" / "checkpoints" / "resnet_qat_int8_2m.pt"
CKPT_PATH       = Path(__file__).parent / "checkpoints" / "phase5_compressed_qat_fusion_best.pt"
RESULTS_DIR     = Path(__file__).parent / "results_compressed_qat"

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
    # QAT INT8 models are CPU-only
    device = "cpu"
    print(f"Device : {device}  (forced — QAT INT8 image encoder is CPU-only)\n")

    if not CKPT_PATH.exists():
        sys.exit(f"Checkpoint not found: {CKPT_PATH}\nRun train_compressed_qat.py first.")

    if not COMPRESSED_CKPT.exists():
        sys.exit(f"QAT checkpoint not found: {COMPRESSED_CKPT}")

    RESULTS_DIR.mkdir(exist_ok=True)

    model = MultiScaleFusionModel(image_feat_dim=IMAGE_FEAT_DIM)
    # Install the QAT INT8 backbone BEFORE loading the fusion state dict —
    # same reason as evaluate_compressed.py: non-standard channel widths.
    model.load_compressed_image_encoder(COMPRESSED_CKPT, device=device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    model = model.to(device)

    n_total    = sum(p.numel() for p in model.parameters())
    n_img_enc  = sum(p.numel() for p in model.image_encoder.parameters())
    n_sen_enc  = sum(p.numel() for p in model.sensor_cnn.parameters())
    print(f"Total parameters      : {n_total:,}")
    print(f"  Image encoder       : {n_img_enc:,}  (QAT INT8 ResNet, 309-d)")
    print(f"  Sensor encoder      : {n_sen_enc:,}  (MultiScaleSensorCNN, 96-d)")
    print(f"  Fusion head + norms : {n_total - n_img_enc - n_sen_enc:,}\n")

    results = {}
    for split in ("train", "val", "test"):
        r = evaluate(split, model, device)
        results[split] = r
        print(
            f"{split:5s}  n={r['n_samples']:4d}  "
            f"MAE={r['mae']:.2f} +/- {r['mae_std']:.2f} um  "
            f"(min={r['mae_min']:.2f}, max={r['mae_max']:.2f})"
        )

    print()
    print("── Baselines ──────────────────────────────────────────────────────────")
    print(f"Phase 5 compressed FP32 (resnet_distilled_2m)       test MAE : 15.55 um")
    print(f"Phase 1 image-only — standard ResNet18  (n=247)     test MAE : 23.17 um")
    print(f"Compressed ResNet 2M FP32 (standalone)  (n=247)     test MAE : 20.80 um")
    print(f"Compressed ResNet 2M QAT  (standalone)  (n=247)     test MAE : 19.06 um")
    print(f"Phase 4 sensor-only (MultiScaleCNN SE)  (n=247)     test MAE : 29.27 um")

    out_path = RESULTS_DIR / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    run()
