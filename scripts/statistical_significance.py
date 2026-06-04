"""
Statistical significance of the headline claim:
the deployable INT8 fusion model beats the deployable INT8 image model on the
held-out test set.

Both models are evaluated on the SAME test samples, so this is a *paired*
comparison: the unit of analysis is the per-sample absolute error, and the two
error vectors are correlated (a hard sample is hard for both models). A paired
test exploits that correlation for statistical power.

Models compared (deployable INT8, full-integer INT8-I/O TFLite — the artifacts
that ship to the MCU, NOT the FP32 versions):
    fusion : fusion/deployment/checkpoints/fusion_int8_qat_nxp_io.tflite   (1.13M, 20.33 µm)
    image  : image/compression/checkpoints/resnet_2m_qat_int8_nxp_io.tflite (2M QAT, 21.97 µm)

Per-sample absolute error:
    e_fusion[i] = |pred_fusion[i] - true[i]|
    e_image[i]  = |pred_image[i]  - true[i]|
    d[i]        = e_fusion[i] - e_image[i]      (negative ⇒ fusion better)

CRITICAL — valid pairing. The image dataset (MATWIDataset) lists samples in
positional order after reset_index; the fusion dataset
(MATWIFusionScalogramDataset) keys on `labels_idx` (the original labels.csv row
index) and only includes the image∩scalogram intersection. We therefore key
BOTH error vectors by `labels_idx`, intersect, sort, and assert the ground-truth
wear values agree element-wise before running any test. This guarantees the two
arrays come from the identical set of samples in the identical order.

Tests
-----
1. Wilcoxon signed-rank (two-sided) — non-parametric paired test; does not
   assume normality (absolute errors are right-skewed, bounded at zero).
2. Paired bootstrap (B=10,000) on the MAE *difference* — tests the exact
   statistic we report (a mean), gives a 95% CI and a two-sided p-value.
3. Cohen's d on the paired differences — effect-size context.

Usage
-----
    python scripts/statistical_significance.py
    python scripts/statistical_significance.py --bootstrap 10000 --seed 42

Run from the thesis root.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import SPLIT_MAP
from image.baseline.dataset import MATWIDataset
from fusion.two_tower.dataset import MATWIFusionScalogramDataset

DATA_ROOT     = ROOT / "data" / "raw"
SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"

IMAGE_TFLITE  = ROOT / "image" / "compression" / "checkpoints" / "resnet_2m_qat_int8_nxp_io.tflite"
FUSION_TFLITE = ROOT / "fusion" / "deployment" / "checkpoints" / "fusion_int8_qat_nxp_io.tflite"

RESULTS_DIR = Path(__file__).parent / "results"


# ── INT8-I/O quant helpers ────────────────────────────────────────────────────

def _quantize(x, scale, zp):
    return np.clip(np.round(x / scale) + zp, -128, 127).astype(np.int8)


# ── Per-sample inference, keyed by labels_idx ─────────────────────────────────

def image_labels_idx(split):
    """Recover the original labels.csv row index for each MATWIDataset sample.

    MATWIDataset applies exactly these three masks (with reset_index between,
    which renumbers but never reorders or drops extra rows), so applying the
    same masks while preserving the original index yields labels_idx[i] aligned
    1:1 with MATWIDataset[i].
    """
    labels = pd.read_csv(DATA_ROOT / "labels.csv")
    labels["labels_idx"] = labels.index
    labels = labels[labels["Set"].isin(SPLIT_MAP[split])]
    labels = labels[labels["ImageFile"].notna()]
    labels = labels[labels["wear"].notna()]
    return labels["labels_idx"].to_numpy()


def run_image_int8(split):
    """Return {labels_idx: (pred_um, true_um)} for the deployable image INT8 model."""
    import tensorflow as tf

    ds = MATWIDataset(DATA_ROOT, split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    idxs = image_labels_idx(split)
    assert len(idxs) == len(ds), (
        f"image labels_idx recovery mismatch: {len(idxs)} vs dataset {len(ds)}")

    interp = tf.lite.Interpreter(model_path=str(IMAGE_TFLITE))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    i_s, i_z = inp["quantization"]
    o_s, o_z = out["quantization"]

    result = {}
    for i, (img, tgt) in enumerate(loader):
        x = _quantize(img.numpy().astype(np.float32), i_s, i_z)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        raw = interp.get_tensor(out["index"]).reshape(-1)[0]
        pred = (float(raw) - o_z) * o_s
        result[int(idxs[i])] = (pred, float(tgt.reshape(-1)[0]))
    return result


def run_fusion_int8(split):
    """Return {labels_idx: (pred_um, true_um)} for the deployable fusion INT8 model."""
    import tensorflow as tf

    ds = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    idxs = ds.meta["labels_idx"].to_numpy()
    assert len(idxs) == len(ds), "fusion labels_idx mismatch"

    interp = tf.lite.Interpreter(model_path=str(FUSION_TFLITE))
    interp.allocate_tensors()
    ins = interp.get_input_details()
    img_in  = next(d for d in ins if int(np.prod(d["shape"])) == 3 * 224 * 224)
    scal_in = next(d for d in ins if int(np.prod(d["shape"])) == 5 * 64 * 64)
    out_det = interp.get_output_details()[0]
    o_s, o_z = out_det["quantization"]

    result = {}
    for i, (img, scal, tgt) in enumerate(loader):
        interp.set_tensor(img_in["index"],
                          _quantize(img.numpy(),  *img_in["quantization"]))
        interp.set_tensor(scal_in["index"],
                          _quantize(scal.numpy(), *scal_in["quantization"]))
        interp.invoke()
        raw = interp.get_tensor(out_det["index"]).reshape(-1)[0]
        pred = (float(raw) - o_z) * o_s
        result[int(idxs[i])] = (pred, float(tgt.reshape(-1)[0]))
    return result


# ── Aligned error vectors ─────────────────────────────────────────────────────

def build_paired_errors(fusion_map, image_map):
    """Intersect on labels_idx, sort, assert truths agree, return aligned arrays."""
    common = sorted(set(fusion_map) & set(image_map))
    if not common:
        sys.exit("No common test samples between the two models — pairing impossible.")

    e_fusion, e_image, truths = [], [], []
    for k in common:
        pf, tf_ = fusion_map[k]
        pi, ti  = image_map[k]
        # Same sample ⇒ identical ground truth (to float tolerance).
        assert abs(tf_ - ti) < 1e-3, (
            f"ground-truth mismatch at labels_idx={k}: fusion {tf_} vs image {ti}")
        e_fusion.append(abs(pf - tf_))
        e_image.append(abs(pi - ti))
        truths.append(tf_)

    e_fusion = np.asarray(e_fusion, dtype=np.float64)
    e_image  = np.asarray(e_image,  dtype=np.float64)
    assert e_fusion.shape == e_image.shape, "aligned arrays differ in length"
    return np.array(common), e_fusion, e_image


# ── Statistical tests ─────────────────────────────────────────────────────────

def test_wilcoxon(e_fusion, e_image):
    res = stats.wilcoxon(e_fusion, e_image, alternative="two-sided")
    return {"W": float(res.statistic), "p_value": float(res.pvalue)}


def test_bootstrap(e_fusion, e_image, B, seed):
    rng = np.random.default_rng(seed)
    n = len(e_fusion)
    diffs = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=n)              # same indices ⇒ pairing preserved
        diffs[b] = e_fusion[idx].mean() - e_image[idx].mean()

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    frac_ge0 = float(np.mean(diffs >= 0.0))
    frac_le0 = float(np.mean(diffs <= 0.0))
    p_value = min(1.0, 2.0 * min(frac_ge0, frac_le0))
    return {
        "B": B,
        "seed": seed,
        "observed_diff": float(e_fusion.mean() - e_image.mean()),
        "bootstrap_mean_diff": float(diffs.mean()),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
        "p_value": p_value,
    }


def test_cohens_d(e_fusion, e_image):
    d = e_fusion - e_image
    sd = d.std(ddof=1)
    cohens_d = float(d.mean() / sd) if sd > 0 else float("nan")
    return {
        "mean_diff": float(d.mean()),
        "std_diff": float(sd),
        "cohens_d": cohens_d,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Paired significance tests: deployable fusion INT8 vs image INT8")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--bootstrap", type=int, default=10000, help="bootstrap iterations B")
    ap.add_argument("--seed", type=int, default=42, help="seed for bootstrap RNG")
    args = ap.parse_args()

    for p in (IMAGE_TFLITE, FUSION_TFLITE):
        if not p.exists():
            sys.exit(f"Missing deployable model: {p}")

    print("Running deployable INT8 inference (this evaluates both TFLite models)…")
    print(f"  image  : {IMAGE_TFLITE.name}")
    print(f"  fusion : {FUSION_TFLITE.name}\n")

    image_map  = run_image_int8(args.split)
    fusion_map = run_fusion_int8(args.split)

    common, e_fusion, e_image = build_paired_errors(fusion_map, image_map)
    n = len(common)

    mae_fusion = float(e_fusion.mean())
    mae_image  = float(e_image.mean())

    print(f"{'═'*70}")
    print(f"  PAIRED TEST SET ({args.split})")
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
    print(f"  H0: paired differences symmetric about zero (no model difference)")
    print(f"  W = {wil['W']:.1f}   p = {wil['p_value']:.4f}")
    print(f"  → {'SIGNIFICANT (p<0.05)' if wil['p_value'] < 0.05 else 'NOT significant (p≥0.05)'}")

    print(f"\n{'─'*70}")
    print(f"  TEST 2 — Paired bootstrap on MAE difference (B={boot['B']}, seed={boot['seed']})")
    print(f"{'─'*70}")
    print(f"  observed ΔMAE   : {boot['observed_diff']:+.4f} µm")
    print(f"  bootstrap ΔMAE  : {boot['bootstrap_mean_diff']:+.4f} µm")
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
        "models": {"fusion": FUSION_TFLITE.name, "image": IMAGE_TFLITE.name},
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
    out_path = RESULTS_DIR / f"statistical_significance_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
