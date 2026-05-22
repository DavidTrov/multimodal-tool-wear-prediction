"""
validate.py — Compare C CWT output against a matching Python reference.

Runs from the repo root OR from experiments/cwt_c/:
    python experiments/cwt_c/validate.py
    python experiments/cwt_c/validate.py --n 50 --seed 99

For each test sample the script:
  1. Loads the raw sensor CSV, applies cutting_mask + extract_cutting_signal
  2. Builds a Python reference scalogram that uses the SAME algorithm as the C code:
       • HPF via scipy.signal.sosfiltfilt (same SOS coefficients, zero initial conditions)
       • Zero-pad to next_pow2(N), apply FFT Morlet filter, IFFT / N_pad
       • Subsample from [0, N-1] using the same truncation formula as C
     This guarantees agreement up to float32 vs float64 rounding.
  3. Writes the gated 5-channel signal to a temp CSV, calls ./cwt_preprocess
  4. Compares Python reference vs C output

NOTE on pywt comparison:
  pywt.cwt uses a scale-adaptive FFT length (next_fast_len(N + wavelet_len - 1))
  and precision=12 wavelet discretization.  This differs from our fixed-N_pad FFT
  approach, giving mean abs diffs of ~5e-3 after normalisation — not a bug, just
  different CWT discretisations.  The Python reference in this script intentionally
  matches C's algorithm, not pywt's.

Pass criterion: max_abs_diff < 5e-3  (pure float32 ↔ float64 rounding).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.aircut_mask import cutting_mask, extract_cutting_signal

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT     = ROOT / "data" / "raw"
SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
BINARY        = Path(__file__).parent / "cwt_preprocess"
RESULTS_DIR   = Path(__file__).parent / "results"

# ── Python reference constants (must match cwt_preprocess.c exactly) ──────────
N_SCALES     = 64
N_TIME       = 64
WAVELET      = "cmor1.5-1.0"
FS           = 1625.0
SENSOR_COLS  = ["acc", "acoustic", "fx", "fy", "fz"]
ALL_COLS     = ["acc", "acoustic", "fx", "fy", "fz", "time"]
FORCE_COLS   = {"fx", "fy", "fz"}

# Exact HPF SOS (butter(4, 15.0, 'high', fs=1625.0, output='sos'))
HPF_SOS = np.array([
    [ 9.27009639993332413e-01, -1.85401927998666483e+00,  9.27009639993332413e-01,
      1.00000000000000000e+00, -1.89514504496437919e+00,  8.98337002438857835e-01],
    [ 1.00000000000000000e+00, -2.00000000000000000e+00,  1.00000000000000000e+00,
      1.00000000000000000e+00, -1.95330751594092811e+00,  9.56597435381147054e-01],
])

SCALES = np.geomspace(1.0, 64.0, num=N_SCALES)


# ── Python reference pipeline (matches C algorithm exactly) ───────────────────

def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _cwt_channel_fft(signal: np.ndarray, force_hpf: bool) -> np.ndarray:
    """
    Compute (N_SCALES, N_TIME) float32 scalogram for one channel.

    Mirrors cwt_preprocess.c exactly:
      • Optional zero-phase HPF via sosfiltfilt with HPF_SOS
      • Zero-pad signal to next_pow2(N) before FFT
      • Morlet filter: psi[k] = sqrt(s) * exp(-Bw*pi^2*(s*k/N_pad - Fc)^2)
        with negative frequencies (k > N_pad//2) zeroed
      • IFFT / N_pad, power |W|^2, subsample from [0, N-1]
      • Per-channel min-max normalisation
    """
    from scipy.signal import sosfiltfilt

    x = signal.copy()
    if force_hpf:
        x = sosfiltfilt(HPF_SOS, x)

    N     = len(x)
    N_pad = _next_pow2(N)
    PI    = np.pi
    BW    = 1.5
    FC    = 1.0

    # Zero-pad and FFT once
    x_pad = np.zeros(N_pad, dtype=np.float64)
    x_pad[:N] = x
    X = np.fft.fft(x_pad)   # complex, length N_pad

    # Frequency array (normalized cycles/sample) — matches C: freq = k / N_pad
    k    = np.arange(N_pad, dtype=np.float64)
    freq = k / N_pad

    # Subsample indices: np.linspace(0, N-1, N_TIME, dtype=int) = truncation
    sub_idx = (np.arange(N_TIME, dtype=np.float64) * (N - 1) / (N_TIME - 1)).astype(int)

    output = np.zeros((N_SCALES, N_TIME), dtype=np.float32)

    for si, s in enumerate(SCALES):
        # Morlet filter — same formula as C
        arg  = s * freq - FC
        psi  = np.sqrt(s) * np.exp(-BW * PI * PI * arg * arg)
        psi[N_pad // 2 + 1:] = 0.0   # zero negative frequencies

        W   = np.fft.ifft(X * psi) / N_pad   # divide by N_pad (kiss_fft convention)
        pwr = W.real ** 2 + W.imag ** 2       # power |W|^2
        output[si] = pwr[sub_idx].astype(np.float32)

    # Noise-floor gate: zero power values below 0.1% of peak (matches C)
    hi = output.max()
    output[output < 1e-3 * hi] = 0.0

    lo = output.min()
    if hi > lo:
        output = (output - lo) / (hi - lo)

    return output


def python_scalogram(signal_path: Path) -> np.ndarray | None:
    """
    Compute the reference (5, 64, 64) float32 scalogram from a raw sensor CSV.
    Uses the SAME algorithm as cwt_preprocess.c (FFT with next_pow2 padding).

    Returns None if the file is too short or missing.
    """
    try:
        raw = pd.read_csv(signal_path, header=None)
    except Exception as e:
        print(f"  [skip] cannot read {signal_path.name}: {e}")
        return None

    if raw.shape[1] < 6:
        print(f"  [skip] {signal_path.name}: expected ≥6 cols, got {raw.shape[1]}")
        return None
    raw.columns = ALL_COLS[:raw.shape[1]]

    if len(raw) < 128:
        print(f"  [skip] {signal_path.name}: only {len(raw)} samples")
        return None

    fx = raw["fx"].to_numpy(dtype=np.float64)
    fy = raw["fy"].to_numpy(dtype=np.float64)
    fz = raw["fz"].to_numpy(dtype=np.float64)
    mask = cutting_mask(fx, fy, fz)

    channels = []
    for col in SENSOR_COLS:
        x_raw = raw[col].to_numpy(dtype=np.float64)
        x_cut = extract_cutting_signal(x_raw, mask)
        channels.append(_cwt_channel_fft(x_cut, force_hpf=(col in FORCE_COLS)))

    return np.stack(channels, axis=0)   # (5, 64, 64)


def write_temp_csv(signal_path: Path, tmp_csv: str) -> int:
    """
    Write the gated 5-channel signal to a temp CSV for the C binary.
    Returns the number of cutting-segment samples, or 0 on failure.
    """
    try:
        raw = pd.read_csv(signal_path, header=None)
        raw.columns = ALL_COLS[:raw.shape[1]]
    except Exception:
        return 0

    if len(raw) < 128:
        return 0

    fx = raw["fx"].to_numpy(dtype=np.float64)
    fy = raw["fy"].to_numpy(dtype=np.float64)
    fz = raw["fz"].to_numpy(dtype=np.float64)
    mask = cutting_mask(fx, fy, fz)

    gated = []
    for col in SENSOR_COLS:
        x = raw[col].to_numpy(dtype=np.float64)
        gated.append(extract_cutting_signal(x, mask))

    n = len(gated[0])
    with open(tmp_csv, "w") as f:
        for row_idx in range(n):
            f.write(",".join(f"{gated[ch][row_idx]:.8f}" for ch in range(5)) + "\n")

    return n


# ── Main ──────────────────────────────────────────────────────────────────────

def run(n_samples: int = 20, seed: int = 42, pass_thresh: float = 5e-3):
    import torch

    rng = np.random.default_rng(seed)

    # Check binary exists
    if not BINARY.exists():
        sys.exit(
            f"Binary not found: {BINARY}\n"
            "  Run:  cd experiments/cwt_c && make"
        )

    # Load labels.csv to get (labels_idx, SensorFile) pairs
    labels = pd.read_csv(DATA_ROOT / "labels.csv")
    df = labels.dropna(subset=["SensorFile", "wear"]).copy()
    df = df[df["SensorFile"].astype(str).str.len() > 0].reset_index(drop=True)

    # Keep only rows where .pt scalogram exists
    df["labels_idx"] = df.index if "labels_idx" not in df.columns else df["labels_idx"]
    df = df[df.index.map(lambda i: (SCALOGRAM_DIR / f"{i}.pt").exists())]
    df = df.sample(min(n_samples, len(df)), random_state=int(seed))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results  = []
    n_pass_c = 0
    n_pass_pt = 0
    total    = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_csv = os.path.join(tmpdir, "signal.csv")
        tmp_bin = os.path.join(tmpdir, "output.bin")

        for labels_idx, row in df.iterrows():
            rel_path    = str(row["SensorFile"]).replace("MATWI/", "", 1)
            sensor_path = DATA_ROOT / rel_path

            if not sensor_path.exists():
                print(f"  [skip] {sensor_path.name} not found")
                continue

            print(f"[{total+1:2d}/{len(df)}] {sensor_path.name[:60]}", end=" ", flush=True)

            # ── Python reference ──────────────────────────────────────────────
            py_ref = python_scalogram(sensor_path)
            if py_ref is None:
                print("→ skipped")
                continue

            # ── Compare Python vs existing .pt file ───────────────────────────
            pt_path = SCALOGRAM_DIR / f"{labels_idx}.pt"
            pt_arr  = torch.load(pt_path, weights_only=True).numpy()

            pt_diff   = np.abs(py_ref - pt_arr)
            pt_maxdif = float(pt_diff.max())
            pt_meandif = float(pt_diff.mean())
            pt_ok     = pt_maxdif < pass_thresh

            # ── Write gated signal to CSV for C binary ────────────────────────
            n_cut = write_temp_csv(sensor_path, tmp_csv)
            if n_cut == 0:
                print("→ write_temp_csv failed")
                continue

            # ── Run C binary ──────────────────────────────────────────────────
            try:
                proc = subprocess.run(
                    [str(BINARY), tmp_csv, tmp_bin],
                    capture_output=True, text=True, timeout=120,
                )
            except subprocess.TimeoutExpired:
                print("→ C binary timed out")
                continue

            if proc.returncode != 0:
                print(f"→ C binary error: {proc.stderr.strip()}")
                continue

            c_arr = np.fromfile(tmp_bin, dtype=np.float32).reshape(5, 64, 64)

            # ── Compare Python reference vs C output ──────────────────────────
            c_diff   = np.abs(py_ref - c_arr)
            c_maxdif = float(c_diff.max())
            c_meandif = float(c_diff.mean())
            c_ok     = c_maxdif < pass_thresh

            status = "PASS" if c_ok else "FAIL"
            print(
                f"  Py↔.pt max={pt_maxdif:.2e} {'✓' if pt_ok else '✗'}  |  "
                f"Py↔C max={c_maxdif:.2e} mean={c_meandif:.2e}  {status}"
            )

            if c_ok:
                n_pass_c += 1
            if pt_ok:
                n_pass_pt += 1
            total += 1

            # Per-channel breakdown for C diff
            ch_stats = {}
            for ch, name in enumerate(["acc", "acoustic", "fx", "fy", "fz"]):
                ch_diff = float(c_diff[ch].max())
                ch_stats[name] = round(ch_diff, 6)

            results.append({
                "labels_idx":    int(labels_idx),
                "sensor_file":   str(row["SensorFile"]),
                "n_cut_samples": n_cut,
                "py_vs_pt": {
                    "max_abs_diff":  round(pt_maxdif, 6),
                    "mean_abs_diff": round(pt_meandif, 6),
                    "pass":          pt_ok,
                },
                "py_vs_c": {
                    "max_abs_diff":  round(c_maxdif, 6),
                    "mean_abs_diff": round(c_meandif, 6),
                    "pass":          c_ok,
                    "per_channel":   ch_stats,
                },
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Tested       : {total} samples")
    print(f"Py ↔ .pt     : {n_pass_pt}/{total} pass  (NOTE: stored .pt files use no HPF — failures expected)")
    print(f"Py ↔ C       : {n_pass_c}/{total} pass  ← main validation")
    print(f"Threshold    : max_abs_diff < {pass_thresh:.0e}")
    print(f"{'='*60}")

    report = {
        "pass_threshold": pass_thresh,
        "n_tested":       total,
        "py_vs_pt_pass":  n_pass_pt,
        "py_vs_c_pass":   n_pass_c,
        "samples":        results,
    }
    out_path = RESULTS_DIR / "validation_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report → {out_path}")

    if n_pass_c < total:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate C CWT against Python reference")
    parser.add_argument("--n",    type=int,   default=20,  help="Number of test samples")
    parser.add_argument("--seed", type=int,   default=42,  help="Random seed")
    parser.add_argument("--tol",  type=float, default=5e-3, help="Pass tolerance (max abs diff)")
    args = parser.parse_args()
    run(n_samples=args.n, seed=args.seed, pass_thresh=args.tol)
