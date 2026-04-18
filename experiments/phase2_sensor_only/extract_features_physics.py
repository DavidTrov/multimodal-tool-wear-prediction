"""
Physics-based sensor feature extraction — FFT + Wavelet.

Replaces the tsfresh MinimalFCParameters approach with features that encode
the physical signatures of tool wear in sensor signals:

  Time-domain  (5 per channel): RMS, kurtosis, crest factor, skewness,
                                 shape factor
  FFT-domain   (10 per channel): energy in 8 frequency bands, spectral
                                  centroid, high-frequency energy ratio
  Wavelet-domain (5 per channel): energy at 4 detail levels + approximation
                                   (db4 wavelet, level 4)

  Total: 20 features × 5 channels = 100 features

Why these matter for tool wear
------------------------------
- Kurtosis + crest factor: sharp, impulsive vibration events caused by
  chipping or localized wear increase both metrics.
- FFT band energies: a worn tool generates more energy in higher harmonics
  of the tooth-passing frequency and in broadband noise.
- Spectral centroid: shifts upward as wear redistributes energy toward
  higher frequencies.
- HF energy ratio: increases monotonically with wear in vibration and
  acoustic channels.
- Wavelet energies: capture time-localised bursts at different scales —
  useful because wear events are often transient.

Output: data/processed/sensor_features_physics.parquet
        (same metadata columns as sensor_features.parquet — drop-in replacement)

Usage:
    python experiments/phase2_sensor_only/extract_features_physics.py

Run from the thesis root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import rfft, rfftfreq

ROOT      = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "raw"
OUT_PATH  = ROOT / "data" / "processed" / "sensor_features_physics.parquet"

SENSOR_COLS  = ["acc", "acoustic", "fx", "fy", "fz", "time"]
SIGNAL_COLS  = ["acc", "acoustic", "fx", "fy", "fz"]   # time is not a signal
N_FFT_BANDS  = 8
WAVELET_NAME = "db4"
WAVELET_LEVEL = 4

# Try to import pywt; fall back to FFT-only if not installed
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("WARNING: PyWavelets (pywt) not installed — wavelet features skipped.")
    print("         Install with: pip install PyWavelets")


# ── Feature extraction ────────────────────────────────────────────────────────

def time_domain_features(x: np.ndarray, ch: str) -> dict:
    """5 time-domain features that are physically sensitive to wear."""
    rms      = np.sqrt(np.mean(x ** 2))
    mean_abs = np.mean(np.abs(x)) + 1e-10
    peak     = np.max(np.abs(x))

    # Near-constant signals (e.g. a silent channel) have near-zero variance,
    # causing catastrophic cancellation in kurtosis/skewness.  Return neutral
    # values (0) instead of unreliable floats.
    if np.std(x) < 1e-10:
        kurt, skew = 0.0, 0.0
    else:
        kurt = float(stats.kurtosis(x))
        skew = float(stats.skew(x))

    return {
        f"{ch}__rms":           rms,
        f"{ch}__kurtosis":      kurt,                 # impulsiveness
        f"{ch}__crest_factor":  peak / (rms + 1e-10), # peak-to-average
        f"{ch}__skewness":      skew,                 # signal asymmetry
        f"{ch}__shape_factor":  rms / mean_abs,       # waveform regularity
    }


def fft_features(x: np.ndarray, ch: str) -> dict:
    """10 FFT features: 8 band energies + spectral centroid + HF energy ratio."""
    # One-sided FFT magnitude spectrum
    spectrum  = np.abs(rfft(x))
    freqs     = rfftfreq(len(x))           # normalised [0, 0.5]
    power     = spectrum ** 2
    total_pwr = power.sum() + 1e-10

    n_bins    = len(spectrum)
    feats     = {}

    # 8 equally-spaced frequency bands (0–0.5 normalised, each band 0.0625 wide)
    edges = np.linspace(0, n_bins, N_FFT_BANDS + 1, dtype=int)
    for i in range(N_FFT_BANDS):
        band_pwr = power[edges[i]:edges[i + 1]].sum()
        feats[f"{ch}__fft_band_{i}"] = band_pwr / total_pwr   # relative energy

    # Spectral centroid — normalised to [0, 1]
    feats[f"{ch}__spectral_centroid"] = float(
        (freqs * power).sum() / (power.sum() + 1e-10) * 2.0   # *2 → range 0–1
    )

    # High-frequency energy ratio (top 50 % of spectrum)
    hf_start = n_bins // 2
    feats[f"{ch}__hf_energy_ratio"] = power[hf_start:].sum() / total_pwr

    return feats


def wavelet_features(x: np.ndarray, ch: str) -> dict:
    """5 wavelet energy ratios across 4 detail levels + approximation (db4)."""
    if not HAS_PYWT:
        return {}

    coeffs     = pywt.wavedec(x, WAVELET_NAME, level=WAVELET_LEVEL)
    # coeffs = [cA4, cD4, cD3, cD2, cD1]
    energies   = [np.sum(c ** 2) for c in coeffs]
    total_e    = sum(energies) + 1e-10

    labels = ["a4", "d4", "d3", "d2", "d1"]
    return {
        f"{ch}__wavelet_{lbl}": e / total_e
        for lbl, e in zip(labels, energies)
    }


def extract_one(sensor_df: pd.DataFrame) -> dict:
    """Extract all physics features from one sensor recording."""
    feats = {}
    for ch in SIGNAL_COLS:
        # np.array() always returns a writable copy — needed because pandas
        # can return read-only views that scipy/pywt refuse to process
        x = np.array(sensor_df[ch], dtype=np.float64)
        feats.update(time_domain_features(x, ch))
        feats.update(fft_features(x, ch))
        feats.update(wavelet_features(x, ch))
    return feats


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_sensor_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] != 6:
        raise ValueError(f"{path} has {df.shape[1]} cols, expected 6")
    df.columns = SENSOR_COLS
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    labels = pd.read_csv(DATA_ROOT / "labels.csv")
    df     = labels.dropna(subset=["SensorFile", "wear"]).copy()
    df     = df[df["SensorFile"].astype(str).str.len() > 0].reset_index(drop=True)
    print(f"Rows with sensor + wear label: {len(df)}")
    print(f"PyWavelets available: {HAS_PYWT}")
    n_feats = 5 + 10 + (5 if HAS_PYWT else 0)
    print(f"Features per channel: {n_feats}  |  Total: {n_feats * len(SIGNAL_COLS)}\n")

    all_features = []
    meta_rows    = []
    skipped      = 0

    for i, row in df.iterrows():
        rel_path    = str(row["SensorFile"]).replace("MATWI/", "", 1)
        sensor_path = DATA_ROOT / rel_path

        if not sensor_path.exists():
            skipped += 1
            continue

        try:
            sensor_df = load_sensor_csv(sensor_path)
            feats     = extract_one(sensor_df)
            all_features.append(feats)
            meta_rows.append({
                "labels_idx": i,
                "Set":        int(row["Set"]),
                "wear":       float(row["wear"]),
                "ImageFile":  str(row["ImageFile"]) if pd.notna(row["ImageFile"]) else "",
            })
        except Exception as e:
            print(f"  Skipping {sensor_path.name}: {e}")
            skipped += 1

        done = len(all_features)
        if done % 100 == 0 and done > 0:
            print(f"  Extracted {done}/{len(df)} ...")

    print(f"\nExtracted: {len(all_features)}  |  Skipped: {skipped}")

    X    = pd.DataFrame(all_features)
    meta = pd.DataFrame(meta_rows).reset_index(drop=True)

    X["labels_idx"] = meta["labels_idx"].values
    X["Set"]        = meta["Set"].values
    X["wear"]       = meta["wear"].values
    X["ImageFile"]  = meta["ImageFile"].values

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(OUT_PATH)
    feature_cols = [c for c in X.columns if c not in ("labels_idx", "Set", "wear", "ImageFile")]
    print(f"Saved to {OUT_PATH}")
    print(f"Samples: {len(X)}  |  Feature columns: {len(feature_cols)}")
    print(f"\nFeature columns:\n" + "\n".join(f"  {c}" for c in feature_cols))


if __name__ == "__main__":
    run()
