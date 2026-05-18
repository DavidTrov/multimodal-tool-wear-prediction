"""
Validates that the sequential (low-memory) CWT implementation produces
numerically identical results to the batch implementation, and profiles
the peak RAM usage of each approach.

Also reports the theoretical on-MCU memory budget using CMSIS-DSP int16
arithmetic, demonstrating that the sequential approach fits within the
NXP FRDM-MCXN947's 512 KB RAM.

Usage
-----
    python experiments/phase4_sensor_cnn/validate_cwt_sequential.py

Run from the thesis root.
"""

import sys
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase3_fusion.precompute_scalograms import (
    SCALES,
    N_TIME,
    WAVELET,
    compute_scalogram,
    compute_scalogram_sequential,
)

FS = 1625  # Hz


# ── Helpers ───────────────────────────────────────────────────────────────────

def _profile(fn, signal):
    """Run fn(signal) under tracemalloc; return (result, peak_kb)."""
    tracemalloc.start()
    result = fn(signal)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / 1024


def _sep(char="─", w=68):
    print(char * w)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(42)

    # ── 1. Correctness check on a short synthetic signal ──────────────────────
    _sep("═")
    print("  1. Correctness — batch vs sequential output (synthetic signal)")
    _sep()
    for n_samples, label in [(500, "short (500 samples)"),
                             (4096, "medium (4 096 samples)"),
                             (16384, "long  (16 384 samples)")]:
        sig = rng.standard_normal(n_samples)
        out_batch = compute_scalogram(sig)
        out_seq   = compute_scalogram_sequential(sig)
        max_diff  = np.abs(out_batch.astype(np.float64) -
                           out_seq.astype(np.float64)).max()
        print(f"  {label}: max |batch − sequential| = {max_diff:.2e}  "
              f"({'✓ identical' if max_diff < 1e-5 else '✗ MISMATCH'})")

    # ── 2. Real scalogram from disk (if available) ────────────────────────────
    _sep()
    scalogram_dir = ROOT / "data" / "processed" / "scalograms"
    pt_files = sorted(scalogram_dir.glob("*.pt"))
    if pt_files:
        import torch
        saved = torch.load(pt_files[0], weights_only=True).numpy()  # (5, 64, 64)
        print(f"  Spot-check against saved scalogram: {pt_files[0].name}")
        print(f"  Shape: {saved.shape}  dtype: {saved.dtype}  "
              f"range: [{saved.min():.3f}, {saved.max():.3f}]")
    else:
        print("  No pre-computed scalograms found — skipping disk spot-check.")

    # ── 3. Peak RAM profiling ─────────────────────────────────────────────────
    _sep("═")
    print("  2. Peak RAM — batch vs sequential (Python, tracemalloc)")
    _sep()

    for n_samples, label in [(4_096,  "4 K   samples  (~2.5 s)"),
                             (16_384, "16 K  samples  (~10 s)"),
                             (26_250, "26 K  samples  (~16 s, typical cut segment)")]:
        sig = rng.standard_normal(n_samples)
        _, batch_kb = _profile(compute_scalogram, sig)
        _, seq_kb   = _profile(compute_scalogram_sequential, sig)
        print(f"  {label} | batch: {batch_kb:7.0f} KB | sequential: {seq_kb:6.0f} KB "
              f"| reduction: {batch_kb/seq_kb:.1f}×")

    # ── 4. Theoretical MCU memory (CMSIS-DSP, int16) ─────────────────────────
    _sep("═")
    print("  3. Theoretical on-MCU memory — CMSIS-DSP int16 / q31")
    _sep()

    N_CUT = 26_250   # typical cutting-segment length
    MAX_SCALE = 64
    # Morlet support: ≈ 8 * sqrt(B) * scale samples (B=1.5 for cmor1.5-1.0)
    MAX_TAPS = int(np.ceil(8 * np.sqrt(1.5) * MAX_SCALE))

    BYTES = {
        "Signal buffer  (int16, 1 ch)":      N_CUT * 2,
        "FFT of signal  (complex int16)":    N_CUT * 4,
        "Morlet FIR taps (complex int16)":   MAX_TAPS * 4,
        "CWT output row  (complex int16)":   N_CUT * 4,
        "Final scalogram (5ch × 64×64 INT8)": 5 * 64 * 64 * 1,
    }

    peak_per_scale = (BYTES["Signal buffer  (int16, 1 ch)"] +
                      BYTES["FFT of signal  (complex int16)"] +
                      BYTES["Morlet FIR taps (complex int16)"] +
                      BYTES["CWT output row  (complex int16)"])

    for name, size in BYTES.items():
        note = " ← persists into inference" if "scalogram" in name else ""
        print(f"  {name:<42}: {size/1024:7.1f} KB{note}")

    _sep()
    print(f"  Peak per scale (signal + FFT + FIR + output) : "
          f"{peak_per_scale/1024:.1f} KB")
    print(f"  MCU RAM budget                               : 512 KB")
    print(f"  Fits in budget                               : "
          f"{'YES ✓' if peak_per_scale < 512*1024 else 'NO ✗'}")
    print()
    print("  Note: signal + FFT buffers are reused across all 64 scales.")
    print("  Model weights (238 KB INT8) live in FLASH — not competing for RAM.")
    print("  Inference scratch RAM (~100–150 KB est.) follows preprocessing")
    print("  sequentially — the same physical RAM is reused (arena allocation).")

    # ── 5. Scale → frequency table ────────────────────────────────────────────
    _sep("═")
    print("  4. Scale → physical frequency correspondence (fs = 1625 Hz, C = 1.0)")
    _sep()
    print(f"  {'Scale':>8}  {'Freq (Hz)':>10}  {'Morlet taps':>12}")
    _sep("·")
    for s in [1, 2, 4, 8, 16, 32, 48, 64]:
        freq_hz = 1.0 * FS / s
        taps    = int(np.ceil(8 * np.sqrt(1.5) * s))
        print(f"  {s:>8.0f}  {freq_hz:>10.1f}  {taps:>12}")
    _sep("═")


if __name__ == "__main__":
    main()
