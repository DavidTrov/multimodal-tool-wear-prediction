"""
Adaptive AC-energy aircut detector for MATWI sensor recordings.

Algorithm (Ferguson et al. conceptual basis, adapted for MATWI low-force regime):
  1. Compute resultant force  Fr = sqrt(fx² + fy² + fz²)
  2. Sliding-window AC-RMS of Fr  (removes DC spindle preload)
  3. Per-file adaptive threshold  = p20 + 0.5*(p80 - p20)
  4. Morphological closing to fill short intra-pass gaps
  5. Fallback: if signal profile is flat (no aircut structure), keep all samples

Rationale for adaptive threshold: MATWI has large per-set DC force offsets
(calibration drift between sessions), so absolute-force gating is impossible.
AC-RMS computes only fluctuation energy from tooth engagement.
"""

from __future__ import annotations

import numpy as np

# ── Tunable parameters ────────────────────────────────────────────────────────
AC_WINDOW           = 512    # samples for sliding AC-RMS  (~0.32 s at 1.6 kHz)
MORPH_CLOSE_SAMPLES = 1600   # fill intra-pass gaps shorter than this (~1 s)
FLAT_RATIO_MIN      = 2.0    # p80/p20 below this → no aircut structure, keep all
# ─────────────────────────────────────────────────────────────────────────────


def _sliding_ac_rms(signal: np.ndarray, window: int) -> np.ndarray:
    """Vectorised sliding-window AC-RMS using cumulative-sum trick."""
    n = len(signal)
    out = np.full(n, np.nan)
    half = window // 2

    # Cumulative sums for O(n) mean and mean-of-squares
    cs1 = np.concatenate(([0.0], np.cumsum(signal)))
    cs2 = np.concatenate(([0.0], np.cumsum(signal ** 2)))

    lo = np.arange(0, n - window + 1)
    hi = lo + window

    win_mean   = (cs1[hi] - cs1[lo]) / window
    win_meansq = (cs2[hi] - cs2[lo]) / window
    rms        = np.sqrt(np.maximum(win_meansq - win_mean ** 2, 0.0))

    # Centre the result
    out[lo + half] = rms
    return out


def _binary_closing(mask: np.ndarray, width: int) -> np.ndarray:
    """Fill False gaps shorter than `width` samples between True regions."""
    if width <= 0:
        return mask.copy()
    # Dilation: mark True if any neighbour within width/2 is True
    import scipy.ndimage as ndi
    return ndi.binary_closing(mask, structure=np.ones(width, dtype=bool))


def cutting_mask(
    fx: np.ndarray,
    fy: np.ndarray,
    fz: np.ndarray,
    *,
    ac_window:           int   = AC_WINDOW,
    morph_close_samples: int   = MORPH_CLOSE_SAMPLES,
    flat_ratio_min:      float = FLAT_RATIO_MIN,
) -> np.ndarray:
    """
    Return a boolean mask with True where the tool is cutting.

    Parameters
    ----------
    fx, fy, fz : 1-D float arrays of the same length
    ac_window  : sliding-window length for AC-RMS
    morph_close_samples : gap-fill width for morphological closing
    flat_ratio_min : p80/p20 threshold below which profile is treated as flat

    Returns
    -------
    mask : bool ndarray, same length as inputs
        True  → cutting engagement
        False → aircut / idle
    """
    fr    = np.sqrt(fx.astype(np.float64) ** 2
                    + fy.astype(np.float64) ** 2
                    + fz.astype(np.float64) ** 2)
    acrms = _sliding_ac_rms(fr, ac_window)

    valid = ~np.isnan(acrms)
    vals  = acrms[valid]

    if len(vals) < 10:
        return np.ones(len(fr), dtype=bool)

    p20 = np.percentile(vals, 20)
    p80 = np.percentile(vals, 80)

    # Flat profile (no meaningful aircut structure) → keep everything
    if p20 < 1e-9 or (p80 / p20) < flat_ratio_min:
        return np.ones(len(fr), dtype=bool)

    threshold = p20 + 0.5 * (p80 - p20)
    mask      = np.where(valid, acrms >= threshold, False)

    mask = _binary_closing(mask, morph_close_samples)

    # Degenerate fallback
    if not mask.any():
        return np.ones(len(fr), dtype=bool)

    return mask


def extract_cutting_signal(signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return only the cutting-region samples of a 1-D signal."""
    out = signal[mask]
    # Guard against degenerate masks leaving too few samples
    return out if len(out) >= 64 else signal
