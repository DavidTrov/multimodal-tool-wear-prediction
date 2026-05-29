"""
make_npz_dataset.py — Convert calibration .npy files to a single .npz for X-CUBE-AI.

X-CUBE-AI Developer Cloud expects a .npz with:
  'x' : float32 array (N, 5, 64, 64)  — model inputs
  'y' : float32 array (N, 1)           — ground truth wear values (µm)

Usage:
    python experiments/cwt_c/make_npz_dataset.py
Output:
    experiments/cwt_c/calibration_dataset.npz
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT     = Path(__file__).resolve().parents[2]
CALIB    = Path(__file__).parent / "calibration_data"
LABELS   = ROOT / "data" / "raw" / "labels.csv"
SCAL_DIR = ROOT / "data" / "processed" / "scalograms"
OUT      = Path(__file__).parent / "calibration_dataset.npz"

# ── Load inputs ───────────────────────────────────────────────────────────────
files = sorted(CALIB.glob("*.npy"))
if not files:
    sys.exit(f"No .npy files found in {CALIB}\n  Run: python make_calibration_dataset.py")

X = np.concatenate([np.load(f) for f in files], axis=0).astype(np.float32)
# X shape: (N, 5, 64, 64)

# ── Recover matching wear labels ──────────────────────────────────────────────
# make_calibration_dataset.py used rng(42) to pick indices from labels.csv
labels  = pd.read_csv(LABELS).dropna(subset=["wear"]).reset_index(drop=True)
rng     = np.random.default_rng(42)
idxs    = rng.choice(len(labels), size=min(100, len(labels)), replace=False)
valid   = [i for i in idxs if (SCAL_DIR / f"{i}.pt").exists()]
valid   = valid[:len(files)]   # match however many .npy files were saved

Y = np.array([labels.iloc[i]["wear"] for i in valid], dtype=np.float32).reshape(-1, 1)
# Y shape: (N, 1)

assert len(X) == len(Y), f"Shape mismatch: X={len(X)}, Y={len(Y)}"

# ── Save ──────────────────────────────────────────────────────────────────────
np.savez(OUT, x=X, y=Y)

print(f"Saved {OUT.name}")
print(f"  x : {X.shape}  dtype={X.dtype}  range=[{X.min():.3f}, {X.max():.3f}]")
print(f"  y : {Y.shape}  dtype={Y.dtype}  range=[{Y.min():.1f}, {Y.max():.1f}] µm")
print(f"  File size: {OUT.stat().st_size // 1024} KB")
