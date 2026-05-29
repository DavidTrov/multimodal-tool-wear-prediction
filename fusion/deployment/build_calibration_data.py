"""
build_calibration_data.py — Build INT8 calibration dataset for NXP ONNX2Quant.

Usage:
    python experiments/phase6_deployment/build_calibration_data.py
    python experiments/phase6_deployment/build_calibration_data.py --n 200

Output:
    experiments/phase6_deployment/calibration_data/
        0000.npy, 0001.npy, ..., 0099.npy     (shape: (1, 5, 64, 64), float32)

Format required by ONNX2Quant:
    onnx2quant model.onnx \
        -c scalogram;experiments/phase6_deployment/calibration_data/ \
        -o model_int8.onnx

Why stratified sampling?
    INT8 PTQ works by observing the activation range at each layer on the
    calibration set and choosing scale/zero-point to cover that range.
    If calibration samples cluster at low wear (e.g. only fresh tools), the
    quantiser misses the high-wear activation range, causing clipping and
    accuracy loss on worn tools. Stratified sampling across wear quantile
    bins ensures all operating conditions are represented.

    100 samples is sufficient for stable statistics on this 64-channel model.
    The calibration set is drawn exclusively from the TRAIN split — never val
    or test — to avoid any information leak into the quantisation step.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.sensor_scalogram_dataset import MATWISensorScalogramDataset

SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
OUT_DIR       = Path(__file__).parent / "calibration_data"

N_BINS    = 10   # wear quantile bins for stratification
N_DEFAULT = 100  # total calibration samples


def build(n_samples: int = N_DEFAULT, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    ds = MATWISensorScalogramDataset(SCALOGRAM_DIR, FEATURES_PATH, split="train")
    print(f"Training set: {len(ds)} samples")

    # Collect all wear values up front for stratification
    targets = np.array([ds[i][1].item() for i in range(len(ds))])
    print(f"Wear range: {targets.min():.1f} – {targets.max():.1f} µm  "
          f"(median {np.median(targets):.1f} µm)")

    # Stratify: divide wear range into N_BINS equal-count bins, sample evenly
    n_per_bin = max(1, n_samples // N_BINS)
    bin_edges = np.percentile(targets, np.linspace(0, 100, N_BINS + 1))

    selected_indices = []
    for b in range(N_BINS):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        # Include upper edge in last bin
        if b < N_BINS - 1:
            mask = (targets >= lo) & (targets < hi)
        else:
            mask = (targets >= lo) & (targets <= hi)
        bucket = np.where(mask)[0]
        if len(bucket) == 0:
            continue
        pick = rng.choice(bucket, size=min(n_per_bin, len(bucket)), replace=False)
        selected_indices.extend(pick.tolist())

    # Top up to n_samples if bins didn't fill evenly
    remaining = list(set(range(len(ds))) - set(selected_indices))
    if len(selected_indices) < n_samples and remaining:
        extra = rng.choice(remaining,
                           size=min(n_samples - len(selected_indices), len(remaining)),
                           replace=False)
        selected_indices.extend(extra.tolist())

    selected_indices = selected_indices[:n_samples]
    rng.shuffle(selected_indices)

    print(f"Selected {len(selected_indices)} calibration samples "
          f"(stratified across {N_BINS} wear bins)")

    # Write one .npy per sample: shape (1, 5, 64, 64) float32
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clear any stale files
    for f in OUT_DIR.glob("*.npy"):
        f.unlink()

    wear_vals = []
    for out_idx, ds_idx in enumerate(selected_indices):
        scalogram, wear = ds[ds_idx]
        arr = scalogram.numpy()[np.newaxis]   # (1, 5, 64, 64)
        np.save(OUT_DIR / f"{out_idx:04d}.npy", arr)
        wear_vals.append(wear.item())

    wear_vals = np.array(wear_vals)
    print(f"\nCalibration set wear distribution:")
    print(f"  min={wear_vals.min():.1f}  median={np.median(wear_vals):.1f}  "
          f"max={wear_vals.max():.1f}  std={wear_vals.std():.1f} µm")

    npy_files = sorted(OUT_DIR.glob("*.npy"))
    sample = np.load(npy_files[0])
    print(f"\nSample shape: {sample.shape}  dtype: {sample.dtype}")
    print(f"Saved {len(npy_files)} files → {OUT_DIR}")

    print(f"\nRun quantization:")
    onnx_path = ROOT / "checkpoints" / "onnx" / "phase4_multiscale_sgdm_best_25.onnx"
    int8_path  = ROOT / "checkpoints" / "onnx" / "phase4_multiscale_sgdm_best_25_int8.onnx"
    print(f"  python -m onnx2quant {onnx_path} \\")
    print(f"      -c scalogram;{OUT_DIR}/ \\")
    print(f"      --per-channel \\")
    print(f"      -o {int8_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build INT8 calibration dataset for NXP ONNX2Quant"
    )
    parser.add_argument("--n",    type=int, default=N_DEFAULT,
                        help=f"Number of calibration samples (default: {N_DEFAULT})")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(n_samples=args.n, seed=args.seed)
