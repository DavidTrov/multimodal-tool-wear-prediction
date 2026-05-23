"""
CWT Compression Phase 3 — INT8 Quantization of the Full Fusion Pipeline.

Applies dynamic INT8 quantization (weight-only) to all Conv2d and Linear
layers in the complete fusion model:

    compressed ResNet (309-d)  +  pruned sensor CNN (96-d)  +  fusion head

Dynamic quantization quantizes layer weights to INT8 at save time while
computing activations in FP32 at inference. This is the standard approach
for deploying PyTorch models on MCUs (TFLite-style INT8 weight compression).

Measures actual INT8 MAE on val and test splits — not just projected size.

Pre-requisites
--------------
    # 1. Prune sensor CNN
    python experiments/compression/cwt/phase1_pruning/train.py

    # 2. Distill sensor CNN
    python experiments/compression/cwt/phase2_distillation/train.py

    # 3. Retrain fusion head with pruned sensor encoder
    python experiments/phase5_scalogram_fusion/train_fully_compressed.py

Usage
-----
    python experiments/compression/cwt/phase3_quantization/quantize.py
    python experiments/compression/cwt/phase3_quantization/quantize.py \\
        --sensor-ckpt checkpoints/sensor_distilled_50.pt \\
        --fusion-ckpt checkpoints/phase5_fully_compressed_best.pt \\
        --output-suffix _50

Run from the thesis root.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.data.fusion_scalogram_dataset import MATWIFusionScalogramDataset
from src.models.multiscale_fusion_model import MultiScaleFusionModel
from src.utils.metrics import mae

DATA_ROOT     = ROOT / "data" / "raw"
SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
CKPT_DIR      = ROOT / "checkpoints"
RESULTS_DIR   = Path(__file__).parent / "results"

IMAGE_FEAT_DIM = 309
NUM_WORKERS    = 0


# ── Utilities ──────────────────────────────────────────────────────────────────

def count_by_dtype(model: nn.Module):
    """Count parameters in quantized (INT8) vs regular (FP32) layers."""
    n_int8, n_fp32 = 0, 0
    for module in model.modules():
        cls_name = type(module).__name__
        if "Quantized" in cls_name or "Dynamic" in cls_name:
            n_int8 += sum(p.numel() for p in module.parameters())
        else:
            for p in module.parameters(recurse=False):
                n_fp32 += p.numel()
    return n_int8, n_fp32


def theoretical_size_kb(n_int8: int, n_fp32: int) -> float:
    """Compute theoretical storage in KB: INT8 = 1 byte, FP32 = 4 bytes."""
    return (n_int8 * 1 + n_fp32 * 4) / 1024


def evaluate_split(model: nn.Module, split: str, device: str) -> dict:
    ds     = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, split)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=NUM_WORKERS)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, scalograms, targets in loader:
            preds = model(images.to(device), scalograms.to(device))[0].squeeze(1)
            all_preds.append(preds.cpu())
            all_targets.append(targets)
    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    errors  = (preds - targets).abs()
    return {
        "split":     split,
        "n":         len(ds),
        "mae":       round(errors.mean().item(), 2),
        "mae_std":   round(errors.std().item(), 2),
        "mae_min":   round(errors.min().item(), 2),
        "mae_max":   round(errors.max().item(), 2),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run(args):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device : {device}\n")

    img_ckpt    = CKPT_DIR / "resnet_distilled_budget.pt"
    sensor_ckpt = Path(args.sensor_ckpt)
    fusion_ckpt = Path(args.fusion_ckpt)

    for p in (img_ckpt, sensor_ckpt, fusion_ckpt):
        if not p.exists():
            sys.exit(f"Checkpoint not found: {p}")

    suffix       = args.output_suffix
    results_name = f"quantization_results{suffix}.json"

    # ── Load full fusion model ────────────────────────────────────────────────
    model = MultiScaleFusionModel(image_feat_dim=IMAGE_FEAT_DIM)
    model.load_compressed_image_encoder(img_ckpt, device=device)
    model.load_pruned_sensor_encoder(sensor_ckpt, device=device)
    model.load_state_dict(torch.load(fusion_ckpt, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    n_total = sum(p.numel() for p in model.parameters())
    n_img   = sum(p.numel() for p in model.image_encoder.parameters())
    n_sen   = sum(p.numel() for p in model.sensor_cnn.parameters())
    n_head  = n_total - n_img - n_sen

    print("FP32 model breakdown:")
    print(f"  Image encoder (compressed ResNet) : {n_img:>9,} params  "
          f"({n_img * 4 / 1024:.0f} KB FP32 / {n_img / 1024:.0f} KB INT8)")
    print(f"  Sensor encoder (pruned CNN)       : {n_sen:>9,} params  "
          f"({n_sen * 4 / 1024:.0f} KB FP32 / {n_sen / 1024:.0f} KB INT8)")
    print(f"  Fusion head + norms               : {n_head:>9,} params  "
          f"({n_head * 4 / 1024:.0f} KB FP32 / {n_head / 1024:.0f} KB INT8)")
    print(f"  Total                             : {n_total:>9,} params  "
          f"({n_total * 4 / 1024:.0f} KB FP32 / {n_total / 1024:.0f} KB INT8)")

    # ── FP32 baseline MAE ─────────────────────────────────────────────────────
    print("\n── FP32 evaluation ──────────────────────────────────────────────")
    fp32_results = {}
    for split in ("val", "test"):
        r = evaluate_split(model, split, device)
        fp32_results[split] = r
        print(f"  {split:5s}  n={r['n']:4d}  MAE={r['mae']:.2f} ± {r['mae_std']:.2f} µm")

    # ── Apply dynamic INT8 quantization ───────────────────────────────────────
    # Weight-only INT8: Conv2d + Linear weights quantized to INT8.
    # Activations remain FP32. No calibration data required.
    # Note: LayerNorm, GroupNorm, GELU, Dropout stay FP32 (not quantized ops).
    print("\n── Applying dynamic INT8 quantization ───────────────────────────")
    model_int8 = torch.quantization.quantize_dynamic(
        model.cpu(),
        qconfig_spec={nn.Linear, nn.Conv2d},
        dtype=torch.qint8,
    )
    model_int8.eval()

    # Move to device for inference (dynamic quantization runs on CPU only)
    # For MPS/CUDA, we run on CPU which is correct for dynamic quantization
    infer_device = "cpu"
    print(f"  Quantized model runs on CPU (dynamic quantization requirement)")

    # Reload model on correct device for INT8 eval
    model_eval = MultiScaleFusionModel(image_feat_dim=IMAGE_FEAT_DIM)
    model_eval.load_compressed_image_encoder(img_ckpt, device="cpu")
    model_eval.load_pruned_sensor_encoder(sensor_ckpt, device="cpu")
    model_eval.load_state_dict(torch.load(fusion_ckpt, map_location="cpu", weights_only=True))
    model_int8 = torch.quantization.quantize_dynamic(
        model_eval.cpu(),
        qconfig_spec={nn.Linear, nn.Conv2d},
        dtype=torch.qint8,
    )
    model_int8.eval()

    # ── INT8 MAE ──────────────────────────────────────────────────────────────
    print("\n── INT8 evaluation ──────────────────────────────────────────────")
    int8_results = {}
    for split in ("val", "test"):
        r = evaluate_split(model_int8, split, infer_device)
        int8_results[split] = r
        fp32_mae = fp32_results[split]["mae"]
        print(
            f"  {split:5s}  n={r['n']:4d}  MAE={r['mae']:.2f} ± {r['mae_std']:.2f} µm  "
            f"(FP32: {fp32_mae:.2f}, Δ={r['mae'] - fp32_mae:+.2f})"
        )

    # ── Size report ───────────────────────────────────────────────────────────
    theoretical_int8_kb = n_total / 1024          # 1 byte per param (weight-only)
    theoretical_fp32_kb = n_total * 4 / 1024

    print("\n── Model size ───────────────────────────────────────────────────")
    print(f"  FP32  : {theoretical_fp32_kb:.0f} KB  ({theoretical_fp32_kb/1024:.2f} MB)")
    print(f"  INT8  : {theoretical_int8_kb:.0f} KB  ({theoretical_int8_kb/1024:.2f} MB)")
    print(f"  Target: 2048 KB (2.00 MB flash)")
    gap = theoretical_int8_kb - 2048
    if gap > 0:
        print(f"  Gap   : {gap:.0f} KB over target  ← INT4 fusion head would save "
              f"{n_head * 0.5 / 1024:.0f} KB")
    else:
        print(f"  Gap   : {abs(gap):.0f} KB under target  ✓ fits in flash")

    # Baselines comparison
    print("\n── Baselines ────────────────────────────────────────────────────")
    print(f"  Paper ResNet50 (image-only)                test MAE : 19.00 µm")
    print(f"  Phase 5c-ii compressed fusion Adam         test MAE : 17.64 µm")
    print(f"  Phase 5 standard fusion (ResNet18)         test MAE : 22.57 µm")
    print(f"  Phase 1 image-only (ResNet18)              test MAE : 23.17 µm")
    print(f"  Phase 4 sensor-only (MultiScaleCNN)        test MAE : 24.96 µm")
    print(f"  Fully compressed fusion (INT8, this run)   test MAE : {int8_results['test']['mae']:.2f} µm")

    # ── Save results ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "sensor_ckpt":            str(sensor_ckpt),
        "fusion_ckpt":            str(fusion_ckpt),
        "n_params_total":         n_total,
        "n_params_image_encoder": n_img,
        "n_params_sensor_cnn":    n_sen,
        "n_params_fusion_head":   n_head,
        "theoretical_fp32_kb":    round(theoretical_fp32_kb, 1),
        "theoretical_int8_kb":    round(theoretical_int8_kb, 1),
        "target_kb":              2048,
        "fits_in_flash":          theoretical_int8_kb <= 2048,
        "fp32": {
            "val_mae":  fp32_results["val"]["mae"],
            "test_mae": fp32_results["test"]["mae"],
        },
        "int8": {
            "val_mae":  int8_results["val"]["mae"],
            "test_mae": int8_results["test"]["mae"],
        },
        "int8_accuracy_drop": {
            "val":  round(int8_results["val"]["mae"]  - fp32_results["val"]["mae"],  2),
            "test": round(int8_results["test"]["mae"] - fp32_results["test"]["mae"], 2),
        },
    }
    out_path = RESULTS_DIR / results_name
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor-ckpt",
                        default=str(CKPT_DIR / "sensor_distilled.pt"))
    parser.add_argument("--fusion-ckpt",
                        default=str(CKPT_DIR / "phase5_fully_compressed_best.pt"))
    parser.add_argument("--output-suffix", default="",
                        help="e.g. '_50' → quantization_results_50.json")
    args = parser.parse_args()
    run(args)
