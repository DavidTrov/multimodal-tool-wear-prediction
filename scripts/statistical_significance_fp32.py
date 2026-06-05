"""
RQ3 — statistical significance of the FP32 fusion model vs the FP32 image model.

This is the FP32 counterpart of scripts/statistical_significance.py (which
compared the *deployable INT8* artifacts). RQ3 asks whether multimodal fusion
helps *before* quantization is applied — i.e. comparing the models at full
precision, removing INT8 rounding as a confound.

Two FP32 models, evaluated in PyTorch on the same test samples:
    fusion : compressed-encoder two-tower (image_feat_dim=309)         ≈15.55 µm
               image encoder : resnet_distilled_2m.pt   (pruned ResNet, 309-d)
               weights       : phase5_compressed_fusion_best.pt
    image  : 2M QAT checkpoint, fc dequantized to pure FP32             ≈19.07 µm
               resnet_qat_int8_2m.pt → dequantize_fc → plain FP32 ResNet

Per-sample absolute error and paired difference:
    e_fusion[i] = |pred_fusion[i] - true[i]|
    e_image[i]  = |pred_image[i]  - true[i]|
    d[i]        = e_fusion[i] - e_image[i]      (negative ⇒ fusion better)

VALID PAIRING. The image dataset (MATWIDataset) lists samples positionally
after reset_index; the fusion dataset (MATWIFusionScalogramDataset) keys on
`labels_idx` over the image∩scalogram intersection. We key BOTH error vectors by
`labels_idx` (original labels.csv row index), intersect, sort, and assert the
ground-truth wear values agree element-wise before any test — so the two arrays
come from the identical samples in the identical order.

Tests (identical to the INT8 analysis, so the two are directly comparable):
    1. Wilcoxon signed-rank (two-sided)
    2. Paired bootstrap (B=10,000) on the MAE difference — 95% CI + p-value
    3. Cohen's d on the paired differences

Usage
-----
    python scripts/statistical_significance_fp32.py
    python scripts/statistical_significance_fp32.py --bootstrap 10000 --seed 42

Run from the thesis root.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))   # import sibling helper module

from image.baseline.dataset import MATWIDataset
from image.compression.quantization.export_onnx_static import dequantize_fc
from fusion.two_tower.dataset import MATWIFusionScalogramDataset
from fusion.two_tower.model import MultiScaleFusionModel

# Reuse the alignment + test machinery from the INT8 analysis so the FP32 and
# INT8 comparisons are methodologically identical by construction.
from statistical_significance import (
    image_labels_idx,
    build_paired_errors,
    test_wilcoxon,
    test_bootstrap,
    test_cohens_d,
)

DATA_ROOT     = ROOT / "data" / "raw"
SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"

# Image FP32 = the 2M QAT checkpoint with its fc dequantized to plain FP32
# (this is the exact model that reports 19.07 µm in the thesis).
IMAGE_FP32_CKPT = ROOT / "image" / "compression" / "checkpoints" / "resnet_qat_int8_2m.pt"

# Fusion FP32 = compressed-encoder two-tower (309-d pruned ResNet + sensor CNN).
FUSION_IMG_ENC_CKPT = ROOT / "image" / "compression" / "checkpoints" / "resnet_distilled_2m.pt"
FUSION_FP32_CKPT    = ROOT / "fusion" / "two_tower" / "checkpoints" / "phase5_compressed_fusion_best.pt"
FUSION_IMAGE_FEAT_DIM = 309

RESULTS_DIR = Path(__file__).parent / "results"


def run_image_fp32(split):
    """Return {labels_idx: (pred_um, true_um)} for the FP32 image model (QAT-dequantized)."""
    torch.backends.quantized.engine = "qnnpack"
    model = torch.load(IMAGE_FP32_CKPT, map_location="cpu", weights_only=False)
    model = dequantize_fc(model).to("cpu").eval()

    ds = MATWIDataset(DATA_ROOT, split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    idxs = image_labels_idx(split)
    assert len(idxs) == len(ds), (
        f"image labels_idx recovery mismatch: {len(idxs)} vs dataset {len(ds)}")

    result = {}
    with torch.no_grad():
        for i, (img, tgt) in enumerate(loader):
            pred = float(model(img).reshape(-1)[0])
            result[int(idxs[i])] = (pred, float(tgt.reshape(-1)[0]))
    return result


def run_fusion_fp32(split):
    """Return {labels_idx: (pred_um, true_um)} for the FP32 compressed two-tower fusion."""
    model = MultiScaleFusionModel(image_feat_dim=FUSION_IMAGE_FEAT_DIM)
    # Install the pruned backbone BEFORE loading the fusion state dict — the
    # checkpoint carries the non-standard channel widths of the compressed ResNet.
    model.load_compressed_image_encoder(FUSION_IMG_ENC_CKPT, device="cpu")
    model.load_state_dict(torch.load(FUSION_FP32_CKPT, map_location="cpu", weights_only=True))
    model = model.to("cpu").eval()

    ds = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    idxs = ds.meta["labels_idx"].to_numpy()
    assert len(idxs) == len(ds), "fusion labels_idx mismatch"

    result = {}
    with torch.no_grad():
        for i, (img, scal, tgt) in enumerate(loader):
            pred = float(model(img, scal)[0].reshape(-1)[0])
            result[int(idxs[i])] = (pred, float(tgt.reshape(-1)[0]))
    return result


def main():
    ap = argparse.ArgumentParser(
        description="RQ3 paired significance: FP32 fusion vs FP32 image")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for p in (IMAGE_FP32_CKPT, FUSION_IMG_ENC_CKPT, FUSION_FP32_CKPT):
        if not p.exists():
            sys.exit(f"Missing checkpoint: {p}")

    print("Running FP32 inference (PyTorch) for both models…")
    print(f"  image  : {IMAGE_FP32_CKPT.name}  (fc dequantized → FP32)")
    print(f"  fusion : {FUSION_FP32_CKPT.name}  (309-d compressed two-tower)\n")

    image_map  = run_image_fp32(args.split)
    fusion_map = run_fusion_fp32(args.split)

    common, e_fusion, e_image = build_paired_errors(fusion_map, image_map)
    n = len(common)

    mae_fusion = float(e_fusion.mean())
    mae_image  = float(e_image.mean())

    print(f"{'═'*70}")
    print(f"  RQ3 — FP32 FUSION vs FP32 IMAGE  ({args.split})")
    print(f"{'═'*70}")
    print(f"  image  samples : {len(image_map)}")
    print(f"  fusion samples : {len(fusion_map)}")
    print(f"  common (paired): {n}   ← unit of analysis")
    if n != len(image_map) or n != len(fusion_map):
        print(f"  NOTE: restricted to the {n}-sample intersection (valid pairing).")
    print()
    print(f"  MAE fusion : {mae_fusion:.4f} µm")
    print(f"  MAE image  : {mae_image:.4f} µm")
    print(f"  difference : {mae_fusion - mae_image:+.4f} µm  "
          f"({'fusion better' if mae_fusion < mae_image else 'image better'})")

    wil  = test_wilcoxon(e_fusion, e_image)
    boot = test_bootstrap(e_fusion, e_image, args.bootstrap, args.seed)
    coh  = test_cohens_d(e_fusion, e_image)

    print(f"\n{'─'*70}")
    print("  TEST 1 — Wilcoxon signed-rank (two-sided)")
    print(f"{'─'*70}")
    print(f"  W = {wil['W']:.1f}   p = {wil['p_value']:.4e}")
    print(f"  → {'SIGNIFICANT (p<0.05)' if wil['p_value'] < 0.05 else 'NOT significant (p≥0.05)'}")

    print(f"\n{'─'*70}")
    print(f"  TEST 2 — Paired bootstrap on MAE difference (B={boot['B']}, seed={boot['seed']})")
    print(f"{'─'*70}")
    print(f"  observed ΔMAE   : {boot['observed_diff']:+.4f} µm")
    print(f"  95% CI          : [{boot['ci95_low']:+.4f}, {boot['ci95_high']:+.4f}] µm")
    print(f"  CI excludes 0   : {boot['ci_excludes_zero']}")
    print(f"  bootstrap p     : {boot['p_value']:.4f}")
    print(f"  → {'SIGNIFICANT (CI excludes 0)' if boot['ci_excludes_zero'] else 'NOT significant (CI includes 0 — gap within evaluation noise)'}")

    print(f"\n{'─'*70}")
    print("  TEST 3 — Effect size (Cohen's d on paired differences)")
    print(f"{'─'*70}")
    print(f"  mean(d) = {coh['mean_diff']:+.4f} µm   std(d) = {coh['std_diff']:.4f} µm")
    print(f"  Cohen's d = {coh['cohens_d']:+.4f}")
    mag = abs(coh["cohens_d"])
    label = ("negligible" if mag < 0.2 else "small" if mag < 0.5
             else "medium" if mag < 0.8 else "large")
    print(f"  → effect size: {label}")

    results = {
        "split": args.split,
        "precision": "fp32",
        "research_question": "RQ3",
        "models": {
            "fusion": {"weights": FUSION_FP32_CKPT.name,
                       "image_encoder": FUSION_IMG_ENC_CKPT.name,
                       "image_feat_dim": FUSION_IMAGE_FEAT_DIM},
            "image": {"weights": IMAGE_FP32_CKPT.name, "note": "fc dequantized to FP32"},
        },
        "n_image": len(image_map),
        "n_fusion": len(fusion_map),
        "n_paired": n,
        "mae_fusion": mae_fusion,
        "mae_image": mae_image,
        "mae_difference": mae_fusion - mae_image,
        "wilcoxon": wil,
        "bootstrap": boot,
        "cohens_d": coh,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"statistical_significance_fp32_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
