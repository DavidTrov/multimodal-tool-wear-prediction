"""
Compression — Dynamic INT8 Quantization of the Distilled Fusion Model.

Applies dynamic INT8 quantization (weight-only) to the distilled fusion model
produced by fusion/two_tower/compression/pruning/distill.py.

Dynamic quantization: Conv2d + Linear weights → INT8 at save time.
Activations remain FP32 at inference. No calibration data required.

Note: this measures the accuracy cost of weight-only INT8 compression. For a
fully deployable INT8 model with calibrated activations, use
static_quant/export_onnx.py instead (ONNX QDQ format, lower RAM).

Pre-requisite
-------------
    python fusion/two_tower/compression/pruning/distill.py

Usage
-----
    python fusion/two_tower/compression/quantization/quantize.py
    python fusion/two_tower/compression/quantization/quantize.py \\
        --model-ckpt fusion/two_tower/compression/pruning/checkpoints/fusion_distilled_qat.pt \\
        --output-suffix _qat

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

from fusion.two_tower.dataset import MATWIFusionScalogramDataset
from src.metrics import mae

DATA_ROOT     = ROOT / "data" / "raw"
SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
PRUNING_CKPT_DIR = Path(__file__).parents[1] / "pruning" / "checkpoints"
RESULTS_DIR      = Path(__file__).parent / "results"

NUM_WORKERS = 0


# ── Utilities ──────────────────────────────────────────────────────────────────

def evaluate_split(model: nn.Module, split: str, device: str) -> dict:
    ds     = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, split)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=NUM_WORKERS)
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for images, scalograms, targets in loader:
            preds = model(images.to(device), scalograms.to(device))[0].squeeze(1)
            preds_all.append(preds.cpu())
            targets_all.append(targets)
    p, t   = torch.cat(preds_all), torch.cat(targets_all)
    errors = (p - t).abs()
    return {
        "split":   split,
        "n":       len(ds),
        "mae":     round(errors.mean().item(), 2),
        "mae_std": round(errors.std().item(), 2),
        "mae_min": round(errors.min().item(), 2),
        "mae_max": round(errors.max().item(), 2),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run(args):
    model_ckpt = Path(args.model_ckpt)
    if not model_ckpt.exists():
        sys.exit(f"Checkpoint not found: {model_ckpt}\n"
                 "Run experiments/compression/cwt/fusion_pruning/train.py first.")

    suffix       = args.output_suffix
    results_name = f"quantization_results{suffix}.json"

    # ── Load pruned model (full object) ───────────────────────────────────────
    print(f"Loading : {model_ckpt}")
    model = torch.load(model_ckpt, map_location="cpu", weights_only=False)
    model.eval()

    n_total = sum(p.numel() for p in model.parameters())
    n_img   = sum(p.numel() for p in model.image_encoder.parameters())
    n_sen   = sum(p.numel() for p in model.sensor_cnn.parameters())
    n_head  = n_total - n_img - n_sen

    print("\nModel breakdown (pruned):")
    print(f"  Image encoder  : {n_img:>8,} params  ({n_img * 4 / 1024:.0f} KB FP32 / {n_img / 1024:.0f} KB INT8)")
    print(f"  Sensor CNN     : {n_sen:>8,} params  ({n_sen * 4 / 1024:.0f} KB FP32 / {n_sen / 1024:.0f} KB INT8)")
    print(f"  Fusion head    : {n_head:>8,} params  ({n_head * 4 / 1024:.0f} KB FP32 / {n_head / 1024:.0f} KB INT8)")
    print(f"  Total          : {n_total:>8,} params  ({n_total * 4 / 1024:.0f} KB FP32 / {n_total / 1024:.0f} KB INT8)")

    # ── FP32 baseline ─────────────────────────────────────────────────────────
    print("\n── FP32 evaluation ──────────────────────────────────────────────────")
    fp32_results = {}
    for split in ("val", "test"):
        r = evaluate_split(model, split, "cpu")
        fp32_results[split] = r
        print(f"  {split:5s}  n={r['n']:4d}  MAE={r['mae']:.2f} ± {r['mae_std']:.2f} µm")

    # ── Dynamic INT8 quantization ─────────────────────────────────────────────
    # Weight-only INT8: Conv2d + Linear weights stored as INT8.
    # Activations remain FP32. Runs on CPU (dynamic quant requirement).
    # LayerNorm, GroupNorm, GELU, Dropout remain FP32.
    # qnnpack is required on Apple Silicon / ARM; fbgemm on x86.
    print("\n── Applying dynamic INT8 quantization ───────────────────────────────")
    import warnings
    for engine in ("qnnpack", "fbgemm"):
        try:
            torch.backends.quantized.engine = engine
            break
        except RuntimeError:
            continue
    print(f"  Quantization engine : {torch.backends.quantized.engine}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model_int8 = torch.quantization.quantize_dynamic(
            model,
            qconfig_spec={nn.Linear, nn.Conv2d},
            dtype=torch.qint8,
        )
    model_int8.eval()
    print("  Done. Running INT8 evaluation on CPU.")

    # ── INT8 evaluation ───────────────────────────────────────────────────────
    print("\n── INT8 evaluation ──────────────────────────────────────────────────")
    int8_results = {}
    for split in ("val", "test"):
        r = evaluate_split(model_int8, split, "cpu")
        int8_results[split] = r
        fp32_mae = fp32_results[split]["mae"]
        print(
            f"  {split:5s}  n={r['n']:4d}  MAE={r['mae']:.2f} ± {r['mae_std']:.2f} µm  "
            f"(FP32: {fp32_mae:.2f}  Δ={r['mae'] - fp32_mae:+.2f})"
        )

    # ── Size report ───────────────────────────────────────────────────────────
    int8_kb  = n_total / 1024
    fp32_kb  = n_total * 4 / 1024
    gap_kb   = int8_kb - 2048

    print("\n── Size report ──────────────────────────────────────────────────────")
    print(f"  FP32  : {fp32_kb:>7.0f} KB  ({fp32_kb/1024:.2f} MB)")
    print(f"  INT8  : {int8_kb:>7.0f} KB  ({int8_kb/1024:.2f} MB)")
    print(f"  Target: {2048:>7d} KB  (2.00 MB NXP flash)")
    if gap_kb > 0:
        print(f"  Gap   : {gap_kb:>+7.0f} KB over target")
        head_int4_saving = n_head * 0.5 / 1024
        print(f"  Note  : INT4 fusion head would save {head_int4_saving:.0f} KB")
    else:
        print(f"  Gap   : {gap_kb:>+7.0f} KB  ✓ fits in flash")

    print("\n── Baselines ────────────────────────────────────────────────────────")
    print(f"  Paper ResNet50 (image-only)                    test MAE : 19.00 µm")
    print(f"  Compressed QAT fusion FP32 (phase5)            val  MAE : 38.74 µm")
    print(f"  Compressed FP32 fusion (phase5, non-QAT)       test MAE : 15.55 µm")
    print(f"  Phase 1   image-only (ResNet18)                test MAE : 23.17 µm")
    print(f"  Phase 4   sensor-only (MultiScaleCNN)          test MAE : 29.27 µm")
    print(f"  Distilled+dynamic-INT8 (this run)              test MAE : {int8_results['test']['mae']:.2f} µm")

    # ── Save ──────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "model_ckpt":             str(model_ckpt),
        "n_params_total":         n_total,
        "n_params_image_encoder": n_img,
        "n_params_sensor_cnn":    n_sen,
        "n_params_fusion_head":   n_head,
        "fp32_kb":                round(fp32_kb, 1),
        "int8_kb":                round(int8_kb, 1),
        "target_kb":              2048,
        "fits_in_flash":          int8_kb <= 2048,
        "fp32": {"val_mae": fp32_results["val"]["mae"], "test_mae": fp32_results["test"]["mae"]},
        "int8": {"val_mae": int8_results["val"]["mae"], "test_mae": int8_results["test"]["mae"]},
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
    parser.add_argument("--model-ckpt",    default=str(PRUNING_CKPT_DIR / "fusion_distilled.pt"),
                        help="Full-object distilled fusion model (use fusion_distilled_qat.pt for QAT pipeline)")
    parser.add_argument("--output-suffix", default="",
                        help="Appended to results filename, e.g. '_qat'")
    args = parser.parse_args()
    run(args)
