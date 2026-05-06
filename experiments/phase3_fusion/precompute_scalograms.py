"""
Pre-compute CWT scalograms from raw sensor CSVs.

For each sensor recording the five signal channels (acc, acoustic, fx, fy, fz)
are transformed into a (5, 64, 64) CWT power scalogram and saved to disk as a
PyTorch tensor.  The file name matches the `labels_idx` column in the physics
parquet so the two datasets can be joined by index.

Aircut gating (AC-energy method, adaptive per file)
----------------------------------------------------
Before computing the CWT, the cutting segment is isolated using the
AC-RMS of the force resultant Fr = sqrt(fx²+fy²+fz²).  This removes the large
per-set DC force offset (calibration drift between sessions) and detects the
tooth-engagement fluctuation energy.  Files with a flat AC-RMS profile (no
detectable aircut structure) are kept in full.  See src/data/aircut_mask.py.

Wavelet  : Complex Morlet  (cmor1.5-1.0)
Scales   : 64 log-spaced values → 64 frequency bins
Time     : cutting-segment signal is sub-sampled to 64 points after CWT
Magnitude: |CWT|²  (power), then per-channel min-max normalised to [0, 1]

Output
------
data/processed/scalograms/<labels_idx>.pt   — (5, 64, 64) float32 tensors

Usage
-----
    python experiments/phase3_fusion/precompute_scalograms.py
    python experiments/phase3_fusion/precompute_scalograms.py --force   # overwrite existing

Run from the thesis root.  Requires PyWavelets (pip install PyWavelets).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT      = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_ROOT = ROOT / "data" / "raw"
OUT_DIR   = ROOT / "data" / "processed" / "scalograms"

from src.data.aircut_mask import cutting_mask, extract_cutting_signal

SENSOR_COLS    = ["acc", "acoustic", "fx", "fy", "fz"]
ALL_COLS       = ["acc", "acoustic", "fx", "fy", "fz", "time"]
N_SCALES       = 64
N_TIME         = 64
WAVELET        = "cmor1.5-1.0"   # complex Morlet, B=1.5, C=1.0
MIN_SIGNAL_LEN = 128             # shorter recordings are skipped


def _make_scales(n: int) -> np.ndarray:
    return np.geomspace(1.0, float(n), num=n)


SCALES = _make_scales(N_SCALES)


def compute_scalogram(signal: np.ndarray) -> np.ndarray:
    """
    CWT power scalogram for a 1-D signal.

    Returns ndarray of shape (N_SCALES, N_TIME), dtype float32, values in [0, 1].
    """
    import pywt

    coeffs, _ = pywt.cwt(signal, SCALES, WAVELET)
    power = np.abs(coeffs) ** 2          # (N_SCALES, len_signal)

    idx   = np.linspace(0, power.shape[1] - 1, N_TIME, dtype=int)
    power = power[:, idx]                # (N_SCALES, N_TIME)

    lo, hi = power.min(), power.max()
    if hi > lo:
        power = (power - lo) / (hi - lo)

    return power.astype(np.float32)


def run(force: bool = False):
    try:
        import pywt  # noqa: F401
    except ImportError:
        sys.exit("PyWavelets not found — install with:  pip install PyWavelets")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(DATA_ROOT / "labels.csv")
    df = labels.dropna(subset=["SensorFile", "wear"]).copy()
    df = df[df["SensorFile"].astype(str).str.len() > 0].reset_index(drop=True)
    print(f"Sensor rows with wear label : {len(df)}")
    print(f"Output directory            : {OUT_DIR}")
    print(f"Aircut gating               : enabled (AC-RMS adaptive threshold)")
    print(f"Overwrite existing          : {force}\n")

    done = skipped = already = gated = kept_full = 0

    for labels_idx, row in df.iterrows():
        out_path = OUT_DIR / f"{labels_idx}.pt"

        if out_path.exists() and not force:
            already += 1
            continue

        rel_path    = str(row["SensorFile"]).replace("MATWI/", "", 1)
        sensor_path = DATA_ROOT / rel_path

        if not sensor_path.exists():
            skipped += 1
            continue

        try:
            raw = pd.read_csv(sensor_path, header=None)
            if raw.shape[1] != 6:
                raise ValueError(f"Expected 6 columns, got {raw.shape[1]}")
            raw.columns = ALL_COLS

            if len(raw) < MIN_SIGNAL_LEN:
                raise ValueError(f"Signal too short ({len(raw)} samples)")

            fx = np.array(raw["fx"], dtype=np.float64)
            fy = np.array(raw["fy"], dtype=np.float64)
            fz = np.array(raw["fz"], dtype=np.float64)

            mask       = cutting_mask(fx, fy, fz)
            ratio_kept = mask.sum() / len(mask)

            if ratio_kept < 0.99:
                gated += 1
            else:
                kept_full += 1

            channels = []
            for ch in SENSOR_COLS:
                x = np.array(raw[ch], dtype=np.float64)
                x_cut = extract_cutting_signal(x, mask)
                channels.append(compute_scalogram(x_cut))

            tensor = torch.from_numpy(np.stack(channels, axis=0))  # (5, 64, 64)
            torch.save(tensor, out_path)
            done += 1

        except Exception as exc:
            print(f"  [{labels_idx}] Skipped {sensor_path.name}: {exc}")
            skipped += 1

        total = done + skipped + already
        if total % 100 == 0 and total > 0:
            print(
                f"  {total}/{len(df)}  "
                f"(done={done}, already={already}, skipped={skipped}, "
                f"gated={gated}, full={kept_full})"
            )

    print(
        f"\nFinished — saved: {done}  |  already existed: {already}  |  skipped: {skipped}"
    )
    print(f"Aircut-gated files : {gated}  |  full-signal files : {kept_full}")
    total_files = len(list(OUT_DIR.glob("*.pt")))
    print(f"Total .pt files in {OUT_DIR}: {total_files}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing .pt files (required to re-run after aircut gating)",
    )
    args = parser.parse_args()
    run(force=args.force)
