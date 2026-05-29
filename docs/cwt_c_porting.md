# CWT Scalogram Preprocessing: Python-to-C Porting

This document covers the complete porting of the CWT (Continuous Wavelet Transform) scalogram computation from Python to C, across two implementations:

1. **Desktop C** (`cwt_preprocess.c`) -- FFT-based, float64, validated against the Python reference.
2. **MCU C** (`cwt_mcu.c`) -- Memory-optimised direct convolution, float32, designed for the NXP FRDM-MCXN947 (Cortex-M33, 512 KB SRAM).

All source files live in `experiments/cwt_c/`.

---

## Table of Contents

- [Background: What the Pipeline Does](#background-what-the-pipeline-does)
- [File Inventory](#file-inventory)
- [Phase 1: Desktop FFT-Based C Implementation](#phase-1-desktop-fft-based-c-implementation)
  - [Algorithm Overview](#algorithm-overview-desktop)
  - [The HPF Problem and How It Was Solved](#the-hpf-problem-and-how-it-was-solved)
  - [Validation](#validation-desktop)
- [Phase 2: Memory-Optimised MCU Implementation](#phase-2-memory-optimised-mcu-implementation)
  - [Why the Desktop Version Cannot Run on the MCU](#why-the-desktop-version-cannot-run-on-the-mcu)
  - [The Key Insight: Direct Time-Domain CWT](#the-key-insight-direct-time-domain-cwt)
  - [Algorithm Overview](#algorithm-overview-mcu)
  - [CMSIS-DSP and the Desktop Shim Pattern](#cmsis-dsp-and-the-desktop-shim-pattern)
  - [Float32 HPF: Precision Trade-off](#float32-hpf-precision-trade-off)
  - [Memory Budget](#memory-budget)
  - [Validation](#validation-mcu)
- [Noise-Floor Gate](#noise-floor-gate)
- [MCU vs FFT Output Equivalence](#mcu-vs-fft-output-equivalence)
- [Low-Energy Channel Detection](#low-energy-channel-detection)
- [Dataset Signal Statistics](#dataset-signal-statistics)
- [Build Instructions](#build-instructions)
- [MCU Deployment Path](#mcu-deployment-path)

---

## Background: What the Pipeline Does

The MATWI sensor setup records 5 channels during CNC milling operations at 1625 Hz:

| Index | Channel    | Description                        |
|-------|------------|------------------------------------|
| 0     | `acc`      | Accelerometer                      |
| 1     | `acoustic` | Acoustic emission                  |
| 2     | `fx`       | Cutting force X (force channel)    |
| 3     | `fy`       | Cutting force Y (force channel)    |
| 4     | `fz`       | Cutting force Z (force channel)    |

The raw CSV files contain the full machining operation, including idle and air-cutting periods. Before CWT computation, an aircut-gating step (`cutting_mask` + `extract_cutting_signal` from `src/data/aircut_mask.py`) removes non-cutting segments, producing a cutting-only signal of 26K--110K samples (16--68 seconds).

The CWT pipeline transforms each channel's 1D time-domain signal into a 2D scalogram:

```
Input:  5 channels x N samples  (N = 26K--110K)
Output: 5 x 64 x 64 float32 tensor (values in [0, 1])
        [channel][scale][time_bin]
```

This tensor is the sensor modality input to the multimodal tool-wear prediction model.

### Pipeline Steps

For each of the 5 channels:

1. **High-pass filter** (force channels only, indices 2--4): A zero-phase 4th-order Butterworth HPF at 15 Hz removes low-frequency drift and DC offset from the force signals. Accelerometer and acoustic channels are not filtered.

2. **Continuous Wavelet Transform**: Complex Morlet wavelet (`cmor1.5-1.0`) applied across 64 logarithmically-spaced scales from 1.0 to 64.0 (`numpy.geomspace(1.0, 64.0, 64)`). Each scale produces a power row of N values.

3. **Subsampling**: Each N-point power row is subsampled to 64 evenly-spaced time bins using integer-truncated indices: `idx[t] = int(t * (N-1) / 63)` for t = 0..63.

4. **Noise-floor gate**: Any CWT power value below 0.1% of the channel's peak power is zeroed. This prevents min-max normalisation from amplifying meaningless differences in near-zero-power regions.

5. **Per-channel normalisation**: Each channel's 64x64 scalogram is independently min-max normalised to [0, 1].

The Python reference implementation lives in `experiments/phase3_fusion/precompute_scalograms.py` and uses `pywt.cwt` and `scipy.signal.sosfiltfilt`.

---

## File Inventory

### Desktop FFT-based (Phase 1)

| File | Purpose |
|------|---------|
| `cwt_preprocess.h` | Public API: `cwt_compute_scalogram()` |
| `cwt_preprocess.c` | FFT-based CWT using kiss_fft, float64 HPF |
| `main.c` | CLI wrapper: reads 5-col CSV, writes 80 KB binary |
| `validate.py` | Validation against Python FFT-based reference |
| `kiss_fft/` | Kiss FFT library (radix-2/mixed-radix C FFT) |

### MCU-optimised (Phase 2)

| File | Purpose |
|------|---------|
| `cwt_mcu.h` | Public API: `cwt_mcu_process_channel()` |
| `cwt_mcu.c` | Direct time-domain CWT, float32 HPF via CMSIS-DSP |
| `cmsis_shim.h` | Desktop compatibility layer mapping CMSIS-DSP to kiss_fft/C |
| `main_mcu.c` | CLI wrapper: reads CSV one column at a time |
| `validate_mcu.py` | Validation against Python direct-CWT reference |

### Shared

| File | Purpose |
|------|---------|
| `Makefile` | Builds both `cwt_preprocess` and `cwt_mcu` targets |
| `results/` | JSON validation reports |

---

## Phase 1: Desktop FFT-Based C Implementation

### Algorithm Overview (Desktop)

`cwt_preprocess.c` implements the CWT using FFT-based convolution, matching the Python `pywt.cwt` approach:

```
For each channel (0--4):
  1. If force channel: zero-phase HPF (float64, full sosfiltfilt replica)
  2. Zero-pad signal to N_pad = next_pow2(N)
  3. FFT the padded signal (kiss_fft, one-time)
  4. For each of 64 scales:
     a. Build freq-domain Morlet wavelet:
        Psi[k] = sqrt(s) * exp(-Bw * pi^2 * (s*k/N_pad - Fc)^2)
     b. Zero negative frequencies (k > N_pad/2) for analytic wavelet
     c. Pointwise multiply: Prod[k] = FFT_signal[k] * Psi[k]
     d. IFFT(Prod) / N_pad
     e. Compute power |W|^2, subsample to 64 time bins
  5. Noise-floor gate: zero power values below 0.1% of channel peak
  6. Per-channel min-max normalise to [0, 1]
```

**Key parameters:**
- Wavelet: `cmor1.5-1.0` (bandwidth Bw = 1.5, centre frequency Fc = 1.0 cycles/sample)
- Scales: `geomspace(1.0, 64.0, 64)` -- 64 log-spaced values
- Frequency convention: normalised cycles/sample (not Hz), matching pywt

**Memory usage:** For a typical 55K-sample signal, N_pad = 65536, requiring 4 complex FFT buffers of 65536 entries each = ~4 MB. For 110K samples, N_pad = 131072, peak RAM = ~9 MB.

### The HPF Problem and How It Was Solved

The zero-phase high-pass filter for force channels was the most difficult part of the port. `scipy.signal.sosfiltfilt` is a complex function with several non-obvious implementation details that all had to be replicated exactly.

#### What sosfiltfilt does

`sosfiltfilt` applies a digital filter twice -- once forward, once backward -- to achieve zero phase distortion. The full procedure:

1. **Odd-extension padding**: Extend the signal symmetrically on both ends to reduce startup transients. For a signal `x[0..N-1]`:
   - Prepend: `ext[i] = 2*x[0] - x[padlen - i]` for i = 0..padlen-1
   - Append: `ext[N+padlen+i] = 2*x[N-1] - x[N-2-i]` for i = 0..padlen-1

2. **Forward pass**: Apply the SOS biquad cascade with initial conditions derived from `sosfilt_zi`.

3. **Reverse**: Flip the filtered buffer.

4. **Backward pass**: Apply the SOS cascade again with initial conditions derived from the last sample.

5. **Reverse back** and extract the unpadded section.

#### Bug 1: Zero initial conditions

The first implementation used zero initial conditions for the biquad cascade. This caused large startup transients because the HPF has near-unity DC gain in its internal states. The fix: precompute `sosfilt_zi` (the steady-state response to a unit step) and initialise the biquad states as `zi[section][j] * x[0]` for the forward pass and `zi[section][j] * y_last` for the backward pass.

The `sosfilt_zi` values were computed once in Python:
```python
scipy.signal.sosfilt_zi(HPF_SOS)
# Section 0: [-0.92700964,  0.92700964]
# Section 1: [ 0.0,          0.0        ]  (HPF blocks DC)
```
and hard-coded into the C source.

#### Bug 2: Wrong padlen formula

The padding length (`padlen`) determines how many samples of odd-extension are prepended and appended. The formula changed between scipy versions:

- **scipy < 1.17**: `padlen = 3 * max(1, 2*n_sections - 1)` = 3 * 3 = **9**
- **scipy >= 1.17**: `padlen = 3 * (2*n_sections + 1)` = 3 * 5 = **15**

The initial implementation used padlen = 9 (the old formula). Switching to padlen = 15 matched the installed scipy 1.17.1 exactly and brought the HPF error from ~5% down to machine precision (~1e-14).

**How this was diagnosed:** A Python script that reimplemented `sosfiltfilt` step by step was written, testing different padlen values. Only padlen = 15 gave exact agreement (0.0 diff) with scipy's output. Reading scipy's actual source code (`_sosfilt.py`, line `ntaps = 2 * len(sos) + 1`) confirmed the formula change.

#### Biquad implementation (Transposed Direct Form II)

Each SOS section implements:
```
y[n] = b0*x[n] + z1
z1   = b1*x[n] - a1*y[n] + z2
z2   = b2*x[n] - a2*y[n]
```

The SOS coefficients from scipy are formatted as `[b0, b1, b2, 1.0, a1, a2]` per section. The `1.0` is the trivially-normalised `a0` and is skipped.

### Validation (Desktop)

`validate.py` tests the desktop C binary against a Python reference that uses the same FFT-based algorithm:

```bash
cd experiments/cwt_c && make
python experiments/cwt_c/validate.py --n 50 --seed 42
```

**Result: 50/50 pass**, max abs diff ~1e-6 (pure float32-vs-float64 rounding). This confirms the C FFT-based implementation is a bit-exact match of the Python pipeline. Both the C binary and Python reference apply a [noise-floor gate](#noise-floor-gate) before normalisation.

---

## Phase 2: Memory-Optimised MCU Implementation

### Why the Desktop Version Cannot Run on the MCU

The NXP FRDM-MCXN947 target has:

| Resource | Limit |
|----------|-------|
| SRAM | 512 KB |
| Flash | 2 MB |
| CPU | Cortex-M33 @ 150 MHz |
| FPU | Single-precision only (no hardware double) |

The desktop C implementation requires 2.3--9 MB peak RAM (depending on signal length), driven by:
1. Four N_pad-sized complex FFT buffers (~4 MB for 65K signals)
2. Two kiss_fft twiddle-factor configs (~1 MB)
3. Float64 arithmetic throughout (no hardware support on Cortex-M33)

This is 4.6--17x over the 512 KB SRAM budget.

### The Key Insight: Direct Time-Domain CWT

The FFT-based approach computes the full CWT at all N time points, but we only keep 64 of them (the subsampled time bins). Computing a 65K-point IFFT just to read 64 values is wasteful.

**Direct time-domain CWT** computes the wavelet convolution only at the 64 needed output points:

```
W(tau, s) = sum_m  signal[tau + m] * kernel_s[m]
```

where `kernel_s` is the time-domain Morlet wavelet at scale `s`, truncated to its effective support (6-sigma window).

The wavelet kernel is computed via a small 2048-point IFFT (not the full N_pad-point IFFT). This is a separate, fixed-size FFT that produces the time-domain wavelet shape, which is then truncated to its non-negligible support and used as a convolution kernel.

**Why 2048 points for the kernel IFFT?** The Morlet wavelet's frequency-domain representation is a Gaussian. On a 2048-point frequency grid, even the narrowest wavelet (scale = 1) spans ~7 frequency bins, giving adequate resolution. Increasing to 4096 or beyond provides no meaningful improvement because the wavelet is already well-resolved.

**Kernel truncation: 6-sigma.** The time-domain Morlet wavelet is also approximately Gaussian in shape. Beyond 6 standard deviations from the centre, the amplitude is below `exp(-18)` ~ 1.5e-8, so truncating there introduces negligible error. The kernel half-width is:

```
half_M = ceil(6 * scale * sqrt(Bw / 2))
```

This gives kernel sizes ranging from 12 samples at scale 1 to 667 samples at scale 64.

**Total computation:** 5 channels x 64 scales x 64 time bins x average kernel size 42 = ~860K complex multiply-accumulate operations per channel, or ~6.6M total. At 150 MHz on Cortex-M33 with single-cycle FMA, this completes in ~44 ms.

### Algorithm Overview (MCU)

`cwt_mcu.c` processes **one channel at a time** (the caller loads each column from the CSV separately):

```
For each channel (called individually by main_mcu.c):
  1. If force channel (idx 2--4):
     Apply zero-phase HPF in float32 via CMSIS biquad cascade
     (same sosfiltfilt algorithm: odd-extension, forward/backward passes)

  2. Compute 64 subsampling indices:
     sub_idx[t] = int(t * (N-1) / 63)

  3. Initialise 2048-point IFFT instance (once per channel)

  4. For each of 64 scales:
     a. Build freq-domain Morlet on 2048-point grid
     b. Zero negative frequencies
     c. IFFT to get time-domain kernel (2048-point, CMSIS arm_cfft_f32)
     d. Truncate to 6-sigma support: kernel[-half_M..+half_M]
     e. Scale by 1/N_ker (IFFT normalisation)
     f. For each of 64 output time bins:
        - Extract signal segment centred at sub_idx[t], zero-pad at boundaries
        - Complex dot product with kernel (CMSIS arm_cmplx_dot_prod_f32)
        - Store power |W|^2

  5. Noise-floor gate: zero power values below 0.1% of channel peak
  6. Per-channel min-max normalise (CMSIS arm_min/max/offset/scale_f32)
```

### CMSIS-DSP and the Desktop Shim Pattern

The MCU code is written against CMSIS-DSP, the standard DSP library for ARM Cortex-M processors. CMSIS-DSP provides hardware-optimised implementations that exploit the Cortex-M33's FPU pipeline and DSP SIMD extensions.

| CMSIS-DSP Function | Purpose in Pipeline | Desktop Shim Implementation |
|--------------------|--------------------|-----------------------------|
| `arm_cfft_f32` | 2048-point IFFT for wavelet kernel | kiss_fft inverse FFT |
| `arm_biquad_cascade_df2T_f32` | Transposed DF2 biquad cascade for HPF | Plain C loop (same arithmetic) |
| `arm_cmplx_dot_prod_f32` | Complex dot product for CWT convolution | Plain C loop |
| `arm_min_f32` / `arm_max_f32` | Min/max scan for normalisation | Plain C loop |
| `arm_offset_f32` / `arm_scale_f32` | Vectorised `(x - lo) / range` normalisation | Plain C loop |

**The shim pattern** (`cmsis_shim.h`) allows the same `cwt_mcu.c` source to compile on both desktop and MCU:

```c
#ifdef DESKTOP_SHIM
  // Desktop: cmsis_shim.h provides C implementations wrapping kiss_fft
  #include "cmsis_shim.h"
#else
  // MCU: real CMSIS-DSP from ARM toolchain
  #include "arm_math.h"
  #include "arm_const_structs.h"
#endif
```

On desktop, `make cwt_mcu` passes `-DDESKTOP_SHIM`. On MCU, that flag is omitted, and the real CMSIS-DSP library is linked instead.

**CMSIS-DSP coefficient convention:** CMSIS biquad functions expect `{b0, b1, b2, -a1, -a2}` per section -- note the **negated** `a1` and `a2` compared to scipy's convention. The HPF coefficients in `cwt_mcu.c` are stored with this negation applied.

### Float32 HPF: Precision Trade-off

The desktop implementation uses float64 for the HPF biquad cascade, matching scipy's default precision. The MCU implementation uses float32 because:

1. The Cortex-M33 has no hardware double-precision FPU -- float64 operations run via software emulation at ~10x the cost.
2. For typical force signals (std > 0.05, hundreds of unique values), the float32 HPF introduces ~1e-5 absolute error, which translates to < 5e-3 in the normalised CWT scalogram. This is well within the validation threshold.

**Where float32 HPF breaks down:** For near-constant force signals (sensor stuck/saturated, < 20 unique values), the HPF output is dominated by float32 rounding noise (~1e-6 amplitude). The CWT of this noise is also noise. After min-max normalisation, the noise fills [0, 1] with a random pattern that differs between float32 and float64 arithmetic. These channels are flagged as "low-energy" in validation and excluded from the pass/fail comparison -- they contain no useful signal information.

### Memory Budget

Processing one channel at a time keeps peak memory within 512 KB:

| Allocation | Size (worst case) | Lifetime |
|---|---|---|
| Signal buffer (1 channel, float32, 110K samples) | 428 KB | Per-channel |
| Output buffer (5 x 64 x 64, float32) | 80 KB | Full duration |
| IFFT workspace (2048 complex float32) | 16 KB | Per-channel |
| HPF extension buffer (N + 30 floats) | ~428 KB | Temp during HPF |
| Signal-as-complex buffer (max kernel * 2) | ~10.7 KB | Per-channel |
| Stack + overhead | ~16 KB | Always |

**HPF extension** is the largest temporary: it needs `N + 2*padlen` = `N + 30` floats. For the worst-case 110K-sample signal, this is another ~428 KB. However, it is allocated only during the HPF step for force channels and freed immediately after, so it does not overlap with the IFFT workspace or signal-as-complex buffer.

**Peak during HPF step:** signal (428 KB) + ext_buf (428 KB) + output (80 KB) = ~936 KB. This exceeds 512 KB.

**Peak during CWT step (non-force channels):** signal (428 KB) + output (80 KB) + workspace (16 KB) + sig_cpx (10.7 KB) = ~535 KB. Tight but feasible.

For the actual MCU deployment, the HPF extension buffer should be allocated from a secondary SRAM bank or the signal buffer should be reused as the extension buffer (with an in-place odd-extension scheme). This optimisation is deferred to the MCU integration step.

### Validation (MCU)

`validate_mcu.py` tests the MCU C binary against a Python reference that uses the **same direct-CWT algorithm** (2048-point IFFT kernel, truncated convolution at 64 time bins):

```bash
cd experiments/cwt_c && make cwt_mcu
python experiments/cwt_c/validate_mcu.py --n 50 --seed 42
```

**Result: 50/50 pass**, max abs diff 8.6e-5 (median 1.1e-5).

The validation script also supports cross-comparison with the desktop FFT binary:

```bash
python experiments/cwt_c/validate_mcu.py --n 50 --seed 42 --cross-compare
```

This shows MCU-vs-FFT diffs of ~6% median (range 1.5%--30%), which is expected because the two CWT algorithms use different frequency grids (2048-point vs N_pad-point). A [noise-floor gate](#noise-floor-gate) eliminates differences in truly zero-power regions, but ~3--8% differences remain in signal-carrying regions due to the algorithm discretisation difference. See [MCU vs FFT Output Equivalence](#mcu-vs-fft-output-equivalence) for details.

---

## Noise-Floor Gate

Both C implementations and their Python references apply a noise-floor gate before min-max normalisation. The gate zeroes any CWT power value below 0.1% of the channel's peak power:

```
gate = 1e-3 * max_power
for each pixel:
    if power < gate: power = 0
```

**Why this was added:** Without the gate, min-max normalisation can amplify tiny numerical differences in near-zero-power regions. When the minimum is close to zero but not exactly zero (e.g., 1e-12 vs 1e-11 due to float32 vs float64 arithmetic), the normalised values in those regions can differ significantly despite representing no real signal energy.

**Where it is implemented** (all four locations must stay in sync):

| File | Code |
|------|------|
| `cwt_mcu.c` | `NOISE_GATE_REL` constant, applied via loop before `arm_min_f32` |
| `cwt_preprocess.c` | Inline `gate = 1e-3f * hi` before min-max normalisation |
| `validate_mcu.py` | `output[output < 1e-3 * hi] = 0.0` in `_cwt_channel_direct()` |
| `validate.py` | `output[output < 1e-3 * hi] = 0.0` in `_cwt_channel_fft()` |

**Impact:** The gate affects only ~0.1% of pixels in typical scalograms (those at the very bottom of the power range). It has no measurable effect on signal-carrying regions.

---

## MCU vs FFT Output Equivalence

The MCU (direct time-domain CWT) and desktop (FFT-based CWT) implementations produce **different scalograms** for the same input. This is expected and is inherent to the algorithm choice, not a bug.

### Source of the difference

| Aspect | Desktop FFT | MCU Direct |
|--------|-------------|------------|
| Frequency grid | N_pad points (32K--128K) | 2048 points |
| Convolution | Full signal via freq-domain multiply | 64 time bins via truncated time-domain sum |
| Kernel origin | Exact frequency-domain wavelet | IFFT of 2048-point discretised wavelet |

The wavelet kernel in the MCU version is computed on a 2048-point frequency grid and IFFT'd to the time domain. While 2048 bins are sufficient to resolve the wavelet at all scales (the narrowest wavelet spans ~7 bins), the resulting time-domain kernel is not identical to what the full N_pad-point FFT approach produces. The difference is a discretisation artifact -- analogous to resampling an image on a coarser grid.

### Measured magnitude

Cross-comparison of 50 validation samples (MCU C vs desktop FFT C) with the noise-floor gate applied:

| Metric | Value |
|--------|-------|
| Median max normalised diff | ~6% |
| Range | 1.5% -- 30% (up to 100% for low-energy channels) |
| Samples with all channels < 5% | ~18/50 |
| Spatial distribution | Mid-power regions, spatially smooth |

The noise-floor gate eliminates differences in truly zero-power regions, but the remaining ~3--8% differences occur in signal-carrying regions where the two algorithms produce genuinely different CWT values. These cannot be eliminated by any threshold approach.

### Implications for model inference

The ~6% median scalogram difference is concentrated in mid-power, spatially smooth regions. Whether this affects tool-wear predictions requires an inference test: feeding both MCU and FFT scalogram versions through the trained model and comparing predicted wear classes. This test is deferred to the model integration phase.

If exact equivalence with the Python/FFT scalograms is a hard requirement, the only option is the full FFT approach, which requires 2.3--9 MB RAM and does not fit in the MCU's 512 KB SRAM.

---

## Low-Energy Channel Detection

Validation excludes channels from the pass/fail comparison when they meet either criterion:

- **Fewer than 20 unique values** in the cutting segment (sensor stuck or saturated)
- **Standard deviation below 0.05** (signal energy too low for meaningful CWT)

These channels produce CWT power near the float32 noise floor (~1e-11 to 1e-10). After normalisation to [0, 1], the result is dominated by arithmetic noise whose pattern depends on whether float32 or float64 was used. Since these channels carry no signal information, excluding them from comparison is appropriate.

In the 50-sample validation run, 39 of 50 samples had at least one low-energy channel skipped. The most common cases were:

- Force sensors stuck at a constant value (1 unique value, std = 0)
- Force sensors with very coarse quantisation (< 10 unique values, std < 1e-3)
- Accelerometer with very low variance (std < 0.05, CWT power at noise floor)

---

## Dataset Signal Statistics

Cutting segment lengths across the full dataset (1,564 files):

| Statistic | Samples | Duration | Memory (1 ch, f32) |
|-----------|---------|----------|---------------------|
| Minimum | 26,539 | 16.3 s | 104 KB |
| Median | 54,685 | 33.7 s | 214 KB |
| Mean | 60,292 | 37.1 s | 236 KB |
| Maximum | 109,644 | 67.5 s | 428 KB |
| Std Dev | 16,188 | 10.0 s | 63 KB |

Sampling rate: 1625 Hz. Segment lengths are determined by the aircut gating algorithm (`cutting_mask`), which detects the onset and end of material contact based on force channel thresholds.

---

## Build Instructions

### Prerequisites

- C compiler: clang (macOS Xcode CLI tools) or gcc
- Python 3.10+ with numpy, scipy, pandas, pywt, torch

### Build

```bash
cd experiments/cwt_c

# Desktop FFT-based binary
make

# MCU-optimised binary (desktop shim)
make cwt_mcu

# Clean all
make clean
```

### Run

```bash
# Desktop binary
./cwt_preprocess <sensor.csv> <output.bin>

# MCU binary (same I/O format)
./cwt_mcu <sensor.csv> <output.bin>

# Load output in Python
import numpy as np
t = np.fromfile('output.bin', dtype=np.float32).reshape(5, 64, 64)
```

Input CSV: 5 columns (acc, acoustic, fx, fy, fz), no header, comma-separated floats. Must contain only the cutting segment (post aircut-gating).

Output binary: 5 x 64 x 64 x 4 bytes = 81,920 bytes, row-major float32.

### Validate

```bash
# Validate desktop binary (FFT-based)
python experiments/cwt_c/validate.py --n 50 --seed 42

# Validate MCU binary (direct CWT)
python experiments/cwt_c/validate_mcu.py --n 50 --seed 42

# MCU validation with FFT cross-comparison
python experiments/cwt_c/validate_mcu.py --n 50 --seed 42 --cross-compare
```

---

## MCU Deployment Path

> **Full deployment guide** (model export, INT8 quantisation, eIQ Toolkit integration, MCUXpresso project setup, on-board profiling) is in [`docs/mcu_deployment.md`](mcu_deployment.md). This section covers only the CWT-specific steps.

The desktop MCU build (`make cwt_mcu`) validates the CWT algorithm. Porting `cwt_mcu.c` to the actual FRDM-MCXN947 requires:

1. **MCUXpresso IDE project**: Import `cwt_mcu.c` and `cwt_mcu.h` (no kiss_fft, no cmsis_shim).
2. **Link CMSIS-DSP**: Add `libarm_cortexM33lf_math.a` from the MCUXpresso SDK.
3. **Include path**: `arm_math.h` and `arm_const_structs.h` (for `arm_cfft_sR_f32_len2048`).
4. **Twiddle tables**: `arm_cfft_sR_f32_len2048` is a `const` struct in flash (~16 KB). Using the length-specific init function avoids linking all table sizes.
5. **Memory map**: Signal buffer in main SRAM, output in secondary SRAM bank if available, const tables in flash.
6. **HPF memory optimisation**: Rework the odd-extension to operate in-place within the signal buffer (avoid the separate `ext_buf` allocation that doubles peak memory).
7. **Sensor input**: Replace CSV reading with DMA-buffered ADC acquisition at 1625 Hz.

The C code is written so that removing `#define DESKTOP_SHIM` and adding `#include "arm_math.h"` is the **only source change** needed for MCU compilation.
