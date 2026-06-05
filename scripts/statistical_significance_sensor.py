"""
Statistical significance of the image and fusion models against the sensor model.

This extends scripts/statistical_significance.py (fusion vs image) with the
deployable INT8 sensor model, answering: do the image-only and fusion models
each beat the sensor-only model by a statistically significant margin?

Three deployable INT8 models (full-integer INT8-I/O TFLite — the artifacts that
ship to the MCU, NOT the FP32 versions):
    fusion : fusion/deployment/checkpoints/fusion_int8_qat_nxp_io.tflite        (≈20.8 µm)
    image  : image/compression/checkpoints/resnet_2m_qat_int8_nxp_io.tflite     (≈22.0 µm)
    sensor : sensor/deployment/checkpoints/sensor_multiscale_int8_nxp_io.tflite (≈28.9 µm)

Two paired comparisons are reported, each negative ⇒ the first model is better:
    fusion vs sensor :  d[i] = e_fusion[i] - e_sensor[i]
    image  vs sensor :  d[i] = e_image[i]  - e_sensor[i]

VALID PAIRING. The three datasets enumerate samples differently (the image
dataset orders by position after reset_index; the fusion dataset uses the
image∩scalogram intersection; the sensor dataset uses every sample with a
precomputed scalogram). We therefore key every model's per-sample error by
`labels_idx` (the original labels.csv row index), restrict ALL THREE models to
their common intersection, sort, and assert the ground-truth wear values agree
element-wise before any test. Both pairwise comparisons then run on the identical
sample set, so they are mutually comparable and the pairing is exact.

Tests per comparison (same as the fusion-vs-image analysis):
    1. Wilcoxon signed-rank (two-sided)  — non-parametric paired test.
    2. Paired bootstrap (B=10,000)       — 95% CI + p-value on the MAE difference.
    3. Cohen's d on the paired differences — effect size.

Usage
-----
    python scripts/statistical_significance_sensor.py
    python scripts/statistical_significance_sensor.py --bootstrap 10000 --seed 42

Run from the thesis root.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))   # import sibling helper module

from sensor.cnn.dataset import MATWISensorScalogramDataset

# Reuse the inference + test machinery already written for fusion-vs-image so the
# two analyses are guaranteed methodologically identical.
from statistical_significance import (
    _quantize,
    run_image_int8,
    run_fusion_int8,
    test_wilcoxon,
    test_bootstrap,
    test_cohens_d,
)

SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
SENSOR_TFLITE = ROOT / "sensor" / "deployment" / "checkpoints" / "sensor_multiscale_int8_nxp_io.tflite"

IMAGE_TFLITE  = ROOT / "image" / "compression" / "checkpoints" / "resnet_2m_qat_int8_nxp_io.tflite"
FUSION_TFLITE = ROOT / "fusion" / "deployment" / "checkpoints" / "fusion_int8_qat_nxp_io.tflite"

RESULTS_DIR = Path(__file__).parent / "results"


def run_sensor_int8(split):
    """Return {labels_idx: (pred_um, true_um)} for the deployable sensor INT8 model."""
    import tensorflow as tf

    ds = MATWISensorScalogramDataset(SCALOGRAM_DIR, FEATURES_PATH, split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    idxs = ds.meta["labels_idx"].to_numpy()
    assert len(idxs) == len(ds), "sensor labels_idx mismatch"

    interp = tf.lite.Interpreter(model_path=str(SENSOR_TFLITE))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    i_s, i_z = inp["quantization"]
    o_s, o_z = out["quantization"]

    result = {}
    for i, (scal, tgt) in enumerate(loader):
        x = _quantize(scal.numpy().astype(np.float32), i_s, i_z)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        raw = interp.get_tensor(out["index"]).reshape(-1)[0]
        pred = (float(raw) - o_z) * o_s
        result[int(idxs[i])] = (pred, float(tgt.reshape(-1)[0]))
    return result


def aligned_errors(map_a, map_b, common):
    """Build index-aligned absolute-error arrays for two models over `common`."""
    e_a, e_b = [], []
    for k in common:
        pa, ta = map_a[k]
        pb, tb = map_b[k]
        assert abs(ta - tb) < 1e-3, (
            f"ground-truth mismatch at labels_idx={k}: {ta} vs {tb}")
        e_a.append(abs(pa - ta))
        e_b.append(abs(pb - tb))
    return np.asarray(e_a, dtype=np.float64), np.asarray(e_b, dtype=np.float64)


def pairwise(name_a, map_a, name_b, map_b, common, B, seed):
    """Run the three paired tests for model A vs model B over the common set."""
    e_a, e_b = aligned_errors(map_a, map_b, common)
    assert e_a.shape == e_b.shape == (len(common),)

    mae_a, mae_b = float(e_a.mean()), float(e_b.mean())
    wil  = test_wilcoxon(e_a, e_b)
    boot = test_bootstrap(e_a, e_b, B, seed)          # diff = MAE_a - MAE_b
    coh  = test_cohens_d(e_a, e_b)

    better = name_a if mae_a < mae_b else name_b

    print(f"\n{'═'*72}")
    print(f"  {name_a.upper()} vs {name_b.upper()}   (n={len(common)} paired)")
    print(f"{'═'*72}")
    print(f"  MAE {name_a:<7}: {mae_a:.4f} µm")
    print(f"  MAE {name_b:<7}: {mae_b:.4f} µm")
    print(f"  ΔMAE ({name_a}-{name_b}): {mae_a - mae_b:+.4f} µm   "
          f"({better} better)")

    print(f"\n  Test 1 — Wilcoxon signed-rank (two-sided)")
    print(f"    W = {wil['W']:.1f}   p = {wil['p_value']:.2e}   "
          f"→ {'SIGNIFICANT' if wil['p_value'] < 0.05 else 'NOT significant'}")

    print(f"\n  Test 2 — Paired bootstrap (B={boot['B']}, seed={boot['seed']})")
    print(f"    ΔMAE = {boot['observed_diff']:+.4f} µm   "
          f"95% CI = [{boot['ci95_low']:+.4f}, {boot['ci95_high']:+.4f}] µm")
    print(f"    bootstrap p = {boot['p_value']:.4f}   "
          f"→ {'SIGNIFICANT (CI excludes 0)' if boot['ci_excludes_zero'] else 'NOT significant (CI includes 0)'}")

    print(f"\n  Test 3 — Cohen's d")
    mag = abs(coh["cohens_d"])
    label = ("negligible" if mag < 0.2 else "small" if mag < 0.5
             else "medium" if mag < 0.8 else "large")
    print(f"    d = {coh['cohens_d']:+.4f}  ({label})")

    return {
        "comparison": f"{name_a}_vs_{name_b}",
        "n_paired": len(common),
        f"mae_{name_a}": mae_a,
        f"mae_{name_b}": mae_b,
        "mae_difference": mae_a - mae_b,
        "better": better,
        "wilcoxon": wil,
        "bootstrap": boot,
        "cohens_d": coh,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Paired tests: deployable image & fusion INT8 vs sensor INT8")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for p in (IMAGE_TFLITE, FUSION_TFLITE, SENSOR_TFLITE):
        if not p.exists():
            sys.exit(f"Missing deployable model: {p}")

    print("Running deployable INT8 inference for all three models…")
    print(f"  image  : {IMAGE_TFLITE.name}")
    print(f"  fusion : {FUSION_TFLITE.name}")
    print(f"  sensor : {SENSOR_TFLITE.name}\n")

    image_map  = run_image_int8(args.split)
    fusion_map = run_fusion_int8(args.split)
    sensor_map = run_sensor_int8(args.split)

    # Common three-way intersection ⇒ all comparisons use the identical samples.
    common = sorted(set(image_map) & set(fusion_map) & set(sensor_map))
    if not common:
        sys.exit("No samples common to all three models — pairing impossible.")

    print(f"{'═'*72}")
    print(f"  SAMPLE COUNTS ({args.split})")
    print(f"{'═'*72}")
    print(f"  image  : {len(image_map)}")
    print(f"  fusion : {len(fusion_map)}")
    print(f"  sensor : {len(sensor_map)}")
    print(f"  common (used for all comparisons): {len(common)}")
    if len({len(image_map), len(fusion_map), len(sensor_map), len(common)}) > 1:
        print(f"  NOTE: restricted to the {len(common)}-sample three-way "
              f"intersection so pairing is valid.")

    out = {
        "split": args.split,
        "models": {"image": IMAGE_TFLITE.name,
                   "fusion": FUSION_TFLITE.name,
                   "sensor": SENSOR_TFLITE.name},
        "n_common": len(common),
        "comparisons": {},
    }

    r1 = pairwise("fusion", fusion_map, "sensor", sensor_map, common, args.bootstrap, args.seed)
    r2 = pairwise("image",  image_map,  "sensor", sensor_map, common, args.bootstrap, args.seed)
    out["comparisons"]["fusion_vs_sensor"] = r1
    out["comparisons"]["image_vs_sensor"]  = r2

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"statistical_significance_sensor_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
