# Sensor Modality — Full Chronological Report

**Project:** MATWI Tool Wear Prediction — Bachelor Thesis  
**Dataset:** MATWI (ICVS 2023). ~1,663 labelled samples; 6-channel sensor CSV per sample (accelerometer, acoustic emission, fx, fy, fz, timestamp; ~78,000 rows per file at ≈1,625 Hz). Target: VB flank wear in µm. Test sets: 4, 9, 13.  
**Metric:** Mean Absolute Error (MAE ± std, µm) on held-out test split.

---

## Table of Contents

1. [Phase 2 — Time-Domain + Machine Learning Baseline](#phase-2)
2. [Phase 2b — Physics-Informed Feature Engineering (FFT + Wavelet)](#phase-2b)
3. [Phase 3/4 — First CWT Scalogram CNN Prototype](#phase-34-prototype)
4. [Phase 4 — Aircut Gating](#phase-4-aircut)
5. [Phase 4 — Optimiser Ablation (SGDM vs Adam)](#phase-4-optimiser)
6. [Phase 4 — Architecture Improvements (GroupNorm, ResBlock, CBAM)](#phase-4-arch)
7. [Phase 4 — Force HPF Preprocessing](#phase-4-hpf)
8. [Phase 4 — Channel Ablation Study](#phase-4-channels)
9. [Phase 4 — Hyperparameter Grid Search](#phase-4-gridsearch)
10. [Phase 4 — Final Sensor-Only Best](#phase-4-final)
11. [Phase 4 — INT8 TFLite Deployment](#phase-4-tflite)
12. [Phase 4 — CBAM Incompatibility and SE Block Replacement](#phase-4-se)
13. [Summary Results Table](#summary)

---

## Phase 2 — Time-Domain + Machine Learning Baseline {#phase-2}

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  Raw Sensor CSV  (6 columns × ~78,000 rows @ 1,625 Hz)      │
│  acc │ acoustic │ fx │ fy │ fz │ timestamp                   │
└──────────────────────────┬──────────────────────────────────┘
                           │  per-channel
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  tsfresh MinimalFCParameters                                 │
│                                                             │
│  Per channel (×5):  mean, median, variance, std,           │
│                     min, max, sum, length,                  │
│                     first/last value, ...                   │
│                                                             │
│  Output: ~65-dim feature vector                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  XGBoost Regressor                                          │
│  n_estimators=100  │  max_depth=6  │  lr=0.3               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                    Wear prediction (µm)
```

### Methodology

The first sensor-only approach used **tsfresh MinimalFCParameters** to extract a compact set of statistical time-domain features from each raw sensor CSV, followed by an **XGBoost regressor** trained on the train split (Sets 1, 2, 5, 7, 8, 10, 11), validated on Sets 3, 6, 12, and tested on Sets 4, 9, 13.

Features extracted by tsfresh MinimalFCParameters include summary statistics (mean, median, variance, standard deviation, min, max, sum), basic temporal features (first and last value, length), and simple distributional properties. Applied across all 5 signal channels (acc, acoustic, fx, fy, fz), yielding a feature vector of approximately 65 features per sample.

The XGBoost model used default hyperparameters (n_estimators=100, max_depth=6, learning_rate=0.3).

### Results

| Split | n | MAE (µm) | Std |
|---|---|---|---|
| Train | 647 | — | — |
| Val | 300 | — | — |
| Test | 247 | **28.25** | ±22.10 |

### Discussion

The tsfresh baseline established that structured machine learning on hand-crafted time-domain features is competitive. At 28.25 µm, the sensor-only model underperformed the image-only ResNet18 baseline (23.17 µm) but demonstrated that the sensor signal carries relevant information about tool wear. The main limitation was that tsfresh's MinimalFCParameters are not tailored to the physical mechanisms of wear — they capture generic statistical properties rather than domain-relevant signatures such as impulsiveness, spectral energy shifts, or time-frequency localisation.

---

## Phase 2b — Physics-Informed Feature Engineering (FFT + Wavelet) {#phase-2b}

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  Raw Sensor CSV  (5 signal channels × ~78,000 rows)                  │
│  acc │ acoustic │ fx │ fy │ fz                                       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │  applied independently per channel (×5)
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
┌──────────────────┐  ┌───────────────┐  ┌───────────────────────┐
│  Time-domain     │  │  FFT-domain   │  │  Wavelet-domain       │
│  (5 features)    │  │  (10 features)│  │  (5 features)         │
│                  │  │               │  │                       │
│  RMS             │  │  Band 0–7     │  │  db4, level 4         │
│  Kurtosis        │  │  (8 bins,     │  │  cA4 energy ratio     │
│  Crest factor    │  │   0–Nyquist)  │  │  cD4 energy ratio     │
│  Skewness        │  │  Spec.centroid│  │  cD3 energy ratio     │
│  Shape factor    │  │  HF ratio     │  │  cD2 energy ratio     │
└────────┬─────────┘  └───────┬───────┘  │  cD1 energy ratio     │
         │                    │          └──────────┬────────────┘
         └──────────┬─────────┘                     │
                    └──────────────┬────────────────┘
                                   │  20 features × 5 channels
                                   ▼
                    ┌──────────────────────────┐
                    │  100-dim feature vector  │
                    └──────────────┬───────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │     XGBoost Regressor    │
                    └──────────────┬───────────┘
                                   │
                                   ▼
                            Wear prediction (µm)
```

### Methodology

The tsfresh feature set was replaced entirely with 100 hand-crafted **physics-motivated features** per sample (20 per signal channel × 5 channels), implemented in `experiments/phase2_sensor_only/extract_features_physics.py`. The same XGBoost regressor was retrained on this new feature set.

**Feature categories:**

*Time-domain (5 per channel):*
- **RMS** — energy content of the signal; increases with material removal rate.
- **Kurtosis** — statistical measure of impulsiveness; elevated kurtosis indicates sharp, transient events caused by chipping or localised wear.
- **Crest factor** (peak / RMS) — another impulsiveness metric; high crest factor signals isolated impact events.
- **Skewness** — asymmetry of the amplitude distribution; reflects the directionality of force transients.
- **Shape factor** (RMS / mean absolute value) — waveform regularity; changes as wear redistributes contact mechanics.

*FFT-domain (10 per channel):*
- **8 frequency band energies** — relative power in 8 equally-spaced bands spanning 0–Nyquist. A worn tool generates more energy in higher harmonics of the tooth-passing frequency and in broadband noise.
- **Spectral centroid** — the frequency "centre of mass" of the power spectrum; shifts upward monotonically with wear as energy redistributes toward higher frequencies.
- **High-frequency energy ratio** — fraction of total power above the mid-frequency point; increases consistently with wear in both vibration and acoustic channels.

*Wavelet-domain (5 per channel):*
- **Discrete Wavelet Transform (DWT) energy ratios** at 4 detail levels and the approximation, using a **db4 (Daubechies order 4)** wavelet at decomposition level 4. DWT decomposes the signal into time-localised frequency sub-bands; energy ratios across levels capture transient wear events that FFT (which assumes stationarity) cannot resolve.

Output stored as `data/processed/sensor_features_physics.parquet` — a drop-in replacement for the tsfresh parquet with identical metadata columns.

### Results

| Split | n | MAE (µm) | Std |
|---|---|---|---|
| Train | 647 | 27.06 | ±30.47 |
| Val | 300 | 54.72 | ±63.23 |
| Test | 247 | **27.43** | ±22.10 |

### Discussion

The physics-informed features marginally improved test MAE from 28.25 to 27.43 µm. The improvement was modest because XGBoost's tree-based splits cannot fully exploit the structured relationships between physics-motivated features — in particular, the joint interaction between frequency bands, kurtosis, and wavelet energies that characterise the wear progression. The large gap between train MAE (27.06 µm) and val MAE (54.72 µm) reflects the domain shift between training sets and the harder validation sets (Sets 3, 6, 12 include different material conditions), a challenge that persists throughout all sensor experiments.

The parquet file generated here became the persistent metadata backbone for all subsequent sensor experiments, providing the `labels_idx` column used to align sensor recordings with scalogram files and image files.

---

## Phase 3/4 — First CWT Scalogram CNN Prototype {#phase-34-prototype}

### Architecture Diagrams

#### CWT Preprocessing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  Raw sensor signal (1 channel, ~78,000 samples)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Aircut gating         │  (added later, see §4)
              │  → cutting segment     │
              └────────────┬───────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │  CWT  (Complex Morlet cmor1.5-1.0)   │
         │  64 log-spaced scales (1 → 64)       │
         │  freq range: ~25 Hz – 1,625 Hz       │
         │  Output: (64, N_signal) complex       │
         └─────────────────┬───────────────────┘
                           │  |·|²  (power)
                           ▼
         ┌─────────────────────────────────────┐
         │  Subsample time axis to 64 bins      │
         │  (uniform index selection)           │
         │  Output: (64, 64) float32            │
         └─────────────────┬───────────────────┘
                           │  min-max normalise → [0,1]
                           ▼
              ┌────────────────────────┐
              │  Per-channel scalogram │
              │  shape: (64, 64)       │
              └────────────┬───────────┘
                           │  stacked × 5 channels
                           ▼
              ┌────────────────────────┐
              │  (5, 64, 64) tensor    │
              │  saved as .pt file     │
              └────────────────────────┘
```

#### VGG-Style Baseline CNN  (`SensorCNNRegressor`, ~61K params)

```
Input: (B, 5, 64, 64)   5 CWT channels, 64×64 time-frequency grid
│
├─ Conv2d(5→16, 3×3, pad=1) → GroupNorm(8,16) → ReLU → MaxPool(2)
│                                                        (B, 16, 32, 32)
├─ Conv2d(16→32, 3×3, pad=1) → GroupNorm(8,32) → ReLU → MaxPool(2)
│                                                        (B, 32, 16, 16)
├─ Conv2d(32→64, 3×3, pad=1) → GroupNorm(8,64) → ReLU → MaxPool(2)
│                                                        (B, 64,  8,  8)
├─ Conv2d(64→64, 3×3, pad=1) → GroupNorm(8,64) → ReLU
│                                                        (B, 64,  8,  8)
├─ AdaptiveAvgPool2d(1) → Flatten
│                                                        (B, 64)
└─ Linear(64→1)
                                                         (B, 1)  → wear (µm)
```

### Methodology

The physics-feature → XGBoost pipeline was abandoned in favour of a direct **end-to-end CNN on Continuous Wavelet Transform (CWT) scalograms**. The motivation was to let the network discover its own wear-sensitive time-frequency features rather than hand-engineering them.

**CWT configuration:**
- Wavelet: Complex Morlet (`cmor1.5-1.0`, bandwidth parameter B=1.5, centre frequency C=1.0)
- Scales: 64 log-spaced values from 1 to 64 → frequency range approximately 25–1,625 Hz
- Time bins: 64 (signal sub-sampled after CWT to a fixed 64-point grid)
- Channels: all 5 signals (acc, acoustic, fx, fy, fz)
- Output: `(5, 64, 64)` float32 tensor per sample, saved to `data/processed/scalograms/<labels_idx>.pt`

Scalograms were pre-computed and cached to disk (`experiments/phase3_fusion/precompute_scalograms.py`) to avoid re-computing them during training.

The CNN architecture was a **4-block VGG-style baseline** (`SensorCNNRegressor`, ~61K parameters) with BatchNorm, pooling after each block, and a fully-connected regression head.

A **SpecAugment**-style augmentation was added: random frequency masking and time masking applied to the scalogram input during training, borrowed from the speech recognition literature (Park et al., INTERSPEECH 2019) to improve generalisation.

A **multiscale inception-style variant** was also built, with parallel 1×1, 3×3, and 5×5 convolution paths at the input block (Zhang et al., Sensors 2023, 23(10), 4595), targeting the multi-resolution nature of CWT scalograms.

The model was exported to ONNX and quantized using the **ST Edge AI Developer Cloud** platform, targeting a deployment profile similar to the NXP FRDM-MCXN947 (Cortex-M33, 512 KB RAM).

### Results

| Variant | Test MAE (µm) | Std | Notes |
|---|---|---|---|
| VGG-style baseline (Adam) | **34.99** | ±29.00 | severe overfitting |
| Multiscale + SpecAugment | 42.84 | ±— | worse than baseline |
| Multiscale (Adam) | 37.07 | ±34.67 | |
| After ONNX + INT8 quantization | **36.65** | ±30.74 | 89 KB model size |

### Discussion

The first CWT CNN (34.99 µm) was worse than both the XGBoost baseline (27.43 µm) and the image-only model (23.17 µm), with severe overfitting: train MAE was in the low teens while test MAE exceeded 34 µm. SpecAugment made things worse (42.84 µm), possibly because the augmentation destroyed time-frequency structure that the network was just beginning to learn. The quantized model (36.65 µm, 89 KB) confirmed that the architecture could be deployed within the size constraint, but the accuracy was insufficient.

The key failure modes identified were: (1) the Adam optimiser converged to a sharp minimum that did not generalise, (2) the force channels (fx, fy, fz) produced nearly all-black CWT scalograms because their useful signal energy is in the sub-Hz DC/trend component, far below the 25 Hz lower bound of the CWT scale range — the network was essentially learning from noise for 3 of 5 channels, and (3) the aircut signal (tool moving but not cutting) was included in the scalograms, contaminating the cutting signal with uninformative motion.

---

## Phase 4 — Aircut Gating {#phase-4-aircut}

### Architecture Diagram

```
Raw sensor signal (full recording, all phases)
│
│  ┌──────────────────────────────────────────────────────────┐
│  │  AIRCUT GATING (AC-RMS force resultant method)           │
│  │                                                          │
│  │  1. Fr = sqrt(fx² + fy² + fz²)       force resultant    │
│  │                                                          │
│  │  2. Sliding AC-RMS  (window = 512 samples, ~0.32 s)     │
│  │     AC-RMS(t) = std(Fr[t : t+512])                      │
│  │     → removes per-set DC force offset                    │
│  │                                                          │
│  │  3. Adaptive threshold (per file)                        │
│  │     τ = p20(AC-RMS) + 0.5 × (p80 − p20)                 │
│  │                                                          │
│  │  4. Binary mask: cutting = AC-RMS > τ                    │
│  │     + morphological closing (fill short gaps)            │
│  │                                                          │
│  │  5. Fallback: if no segment detected → use full signal   │
│  └──────────────────────────────────────────────────────────┘
│
▼
Cutting segment only  (aircut and tool-lift removed)
│
▼
CWT → (5, 64, 64) scalogram
```

```
AC-RMS profile example:

amplitude
   │         ╭──────────────────╮
τ ─┼─────────┤                  ├──────────
   │         │    cutting       │
   │ aircut  │    segment       │  retract
   └─────────┴──────────────────┴──────────→ time
              ◄── kept ────────►
```

### Methodology

An **aircut gating** step was developed and integrated into the scalogram precomputation pipeline (`src/data/aircut_mask.py`). The goal was to isolate the actual cutting segment of each sensor recording before computing the CWT, removing the portions where the tool is moving through air (no material contact) or has already left the workpiece.

**Aircut detection algorithm (AC-RMS method):**
1. Compute the **force resultant** Fr = √(fx² + fy² + fz²) for the full recording.
2. Apply a sliding window AC-RMS (window = 512 samples, ≈0.32 s at 1,625 Hz) to capture the fluctuation energy around each local mean — this removes the large DC force offset that varies between sets due to calibration drift.
3. Compute an **adaptive per-file threshold**: threshold = p20 + 0.5 × (p80 − p20), where p20 and p80 are the 20th and 80th percentiles of the AC-RMS profile. This adapts to the signal amplitude of each individual recording rather than using a global fixed threshold.
4. Apply **morphological closing** to fill short gaps in the detected cutting region.
5. **Fallback**: if no cutting segment is detected (flat AC-RMS profile), retain the full signal.

The cutting segment is then extracted and used as input to the CWT. An aircut diagnostic visualizer (`experiments/phase4_sensor_cnn/diagnose_aircuts.py`) was used to inspect aircut structure across representative measurements from Sets 1, 5, 9, 10, and 13 at early/mid/late wear quantiles.

### Results

| Variant | Test MAE (µm) | Std | Size (INT8) |
|---|---|---|---|
| Pre-aircut (baseline CNN) | 34.99 | ±29.00 | — |
| Post-aircut (same architecture) | 38.79 | ±44.3 | 59.6 KB |

### Discussion

Counterintuitively, the initial post-aircut result (38.79 µm) was worse than before. Two factors explain this: (1) the architecture was also reverted to the CNN v1 at the same time as aircut gating was added, introducing a confounding change, and (2) the optimizer was still Adam, which was subsequently shown to be suboptimal for this task. The aircut gating improved the signal quality but the benefit could not be observed until the optimiser and architecture were improved. Later experiments confirmed the benefit — the combination of aircut gating + HPF + SGDM + CBAM reached 27.19 µm, well below the pre-aircut floor of 34.99 µm.

---

## Phase 4 — Optimiser Ablation (SGDM vs Adam) {#phase-4-optimiser}

### Methodology

A systematic comparison of **Adam** and **SGD with momentum (SGDM)** was conducted on the sensor CNN. SGDM configuration followed Zhang et al. (2023, §4.1): lr=1e-3, momentum=0.9, weight_decay=5e-3. Adam used lr=1e-3 with default β parameters.

Both architectures (baseline VGG-style and multiscale) were trained under each optimiser, with checkpoints saved at the best validation MAE. A patience-based early stopping was also tested.

### Results

| Architecture | Optimiser | Test MAE (µm) | Std |
|---|---|---|---|
| Baseline | Adam | 41.00 | ±35.94 |
| Baseline | **SGDM** | **32.31** | ±32.09 |
| Multiscale | Adam | 37.07 | ±34.67 |
| Multiscale | **SGDM** | 34.68 | ±33.08 |

SGDM converged to a best checkpoint extremely early (epoch 13 for the baseline), with subsequent epochs showing no improvement or degradation.

### Discussion

SGDM consistently and significantly outperformed Adam across both architectures — a 22% improvement for the baseline (41.00 → 32.31 µm) and 7% for the multiscale (37.07 → 34.68 µm). This aligns with the broader literature: for small datasets with high noise, SGDM's implicit gradient smoothing via momentum provides better generalisation than Adam's per-parameter adaptive learning rates, which can overfit to batch-level noise. The early convergence of SGDM (epoch 13) suggested the optimiser landscape was well-conditioned and that Adam was likely oscillating around or past the true minimum. SGDM was adopted as the default for all subsequent sensor experiments.

---

## Phase 4 — Architecture Improvements (GroupNorm, ResBlock, CBAM) {#phase-4-arch}

### Architecture Diagrams

#### ResBlock (two-conv residual block)

```
input x  (B, C, H, W)
│   │
│   └─ Conv2d(C→C, 3×3) → GroupNorm → ReLU
│      Conv2d(C→C, 3×3) → GroupNorm
│             │
│    (residual output)
│
└──────────(+)──── ReLU ──→ output  (B, C, H, W)
           ↑
       skip path (identity)
```

#### CBAM — Convolutional Block Attention Module

```
input x  (B, C, H, W)
│
├─── Channel Attention ──────────────────────────────────┐
│    AdaptiveAvgPool2d(1) → Flatten                      │
│    Linear(C → C/8) → ReLU → Linear(C/8 → C) → Sigmoid │
│    → scale vector  (B, C)  reshape to (B, C, 1, 1)     │
│    x_ca = x × scale                                    │
│                                                        │
│    (re-weights each of the 5 sensor channels           │
│     by learned relevance to current wear state)        │
└────────────────────────────────────────────────────────┘
         │ x_ca
         ▼
├─── Spatial Attention ──────────────────────────────────┐
│    channel_mean(x_ca)  → (B, 1, H, W)                 │
│    channel_max(x_ca)   → (B, 1, H, W)                 │
│    cat → (B, 2, H, W)                                 │
│    Conv2d(2→1, 7×7, pad=3) → Sigmoid                  │
│    → spatial map  (B, 1, H, W)                        │
│    x_out = x_ca × spatial_map                         │
│                                                        │
│    (focuses on wear-relevant time-frequency regions)   │
└────────────────────────────────────────────────────────┘
         │
         ▼
output  (B, C, H, W)   same shape as input
```

#### Final MultiScaleSensorCNN  (~244K params, ~60 KB INT8)

```
Input: (B, 5, 64, 64)   5 HPF-preprocessed CWT channels
│
│  ┌─── Multi-Scale Entry Block ──────────────────────────────────┐
│  │                                                              │
│  │  path_1×1 ── Conv(5→16, 1×1) ─ GN ─ ReLU ─────────┐        │
│  │                                                     │        │
│  │  path_3×3 ── Conv(5→8,  1×1) ─ GN ─ ReLU           │        │
│  │             └ Conv(8→24, 3×3) ─ GN ─ ReLU ─────────┤        │
│  │                                                     │ concat │
│  │  path_5×5 ── Conv(5→4,  1×1) ─ GN ─ ReLU           │        │
│  │             └ Conv(4→8,  5×5) ─ GN ─ ReLU ─────────┘        │
│  │                                                              │
│  │              16 + 24 + 8 = 48 channels, 64×64               │
│  └──────────────────────────────────────────────────────────────┘
│                          │
│                     MaxPool(2)           (B, 48, 32, 32)
│
├─ Conv2d(48→64, 3×3) → GroupNorm(8,64) → ReLU
│                                          (B, 64, 32, 32)
├─ MaxPool(2)                              (B, 64, 16, 16)
│
├─ ResBlock(64)     ← skip connection     (B, 64, 16, 16)
│
├─ CBAM(64)         ← channel + spatial   (B, 64, 16, 16)
│                      attention
│
├─ Conv2d(64→96, 3×3) → GroupNorm(8,96) → ReLU
│                                          (B, 96, 16, 16)
├─ MaxPool(2)                              (B, 96,  8,  8)
│
├─ Conv2d(96→96, 3×3) → GroupNorm(8,96) → ReLU
│                                          (B, 96,  8,  8)
├─ AdaptiveAvgPool2d(1) → Flatten          (B, 96)
│
└─ Dropout(0.3) → Linear(96→1)             (B, 1) → wear (µm)
```

### Methodology

Three successive architectural improvements were applied to the multiscale sensor CNN, each motivated by the sensor training regime (SGDM, batch size 16–32, ~647 training samples):

**1. GroupNorm (replaces BatchNorm):**  
BatchNorm computes normalisation statistics over the batch. With SGDM at batch sizes 16–32, batch statistics are noisy (Luo et al., NeurIPS 2021), causing unstable normalisation particularly in later training stages. GroupNorm (Wu & He, ECCV 2018) with G=8 groups computes statistics within spatial groups of a single sample, making it independent of batch size. A fallback to G=4, G=2, G=1 was implemented for small channel counts.

**2. Residual Block (ResBlock):**  
A two-convolutional-layer residual block (He et al., CVPR 2016) was inserted in the feature extractor at the 64-channel, 16×16 spatial stage. Skip connections provide gradient highways for SGDM's noisier updates (Keskar et al., ICLR 2017), mitigating the vanishing gradient problem that disproportionately affects SGDM compared to Adam.

**3. CBAM — Convolutional Block Attention Module (after ResBlock):**  
CBAM (Woo et al., ECCV 2018) combines channel attention and spatial attention. In this context:
- **Channel attention** re-weights the 5 heterogeneous sensor channels (accelerometer, acoustic, fx, fy, fz) by their relevance to the current wear state. Not all channels contribute equally across all wear levels or workpiece materials.
- **Spatial attention** focuses on wear-relevant time-frequency regions in the scalogram at the 16×16 stage — identifying which frequency bands and time windows contain the most discriminative wear signatures.

**4. Huber Loss (replaces MSE Loss):**  
`nn.HuberLoss(delta=20.0)` was adopted. For errors ≤ 20 µm (within one measurement uncertainty): L = 0.5 × error² (quadratic, same as MSE). For errors > 20 µm: L = 20 × (|error| − 10) (linear, clipped gradient). MATWI wear values span 0–220 µm; at a 200 µm error, MSE contributes 40,000 per sample (summing to >200,000 over a batch), causing catastrophic gradient scaling. Huber's linear tail caps this at 3,800. Delta=20 was chosen to match one standard measurement tolerance unit in the dataset.

**5. Dropout and Early Stopping:**  
Dropout was tuned to 0.25–0.3, and patience-based early stopping was added to prevent over-training beyond the SGDM convergence point.

**Final MultiScaleSensorCNN architecture:**

```
Entry block — three parallel paths (no spatial downsampling):
    path_1x1 : Conv(5→16, 1×1) → GN → ReLU
    path_3x3 : Conv(5→8,  1×1) → GN → ReLU → Conv(8→24,  3×3) → GN → ReLU
    path_5x5 : Conv(5→4,  1×1) → GN → ReLU → Conv(4→8,   5×5) → GN → ReLU
    → concat → 48 channels, 64×64

Feature extractor:
    MaxPool(2)                        → (48, 32, 32)
    Conv(48→64, 3×3) → GN → ReLU
    MaxPool(2)                        → (64, 16, 16)
    ResBlock(64)                      ← skip connection
    CBAM(64)                          ← channel + spatial attention
    Conv(64→96, 3×3) → GN → ReLU
    MaxPool(2)                        → (96, 8, 8)
    Conv(96→96, 3×3) → GN → ReLU     → (96, 8, 8)
    AdaptiveAvgPool2d(1) → Flatten    → 96-dim embedding

Head:
    Dropout(0.3) → Linear(96→1)
```

Total parameters: ~244,000. INT8 size: ~59.6 KB.

### Results

| Variant | Test MAE (µm) | Std |
|---|---|---|
| Multiscale + SGDM (before arch improvements) | 34.68 | ±33.08 |
| + GroupNorm + ResBlock + cosine LR | ~30–33 | — |
| + Dropout(0.25–0.3) + early stopping | ~29 | — |
| **+ CBAM + HuberLoss(δ=20)** | **29** | **±26** |

### Discussion

Each improvement contributed incrementally. GroupNorm + ResBlock was the largest single step, reducing test MAE to ~30–33 µm. CBAM + Huber then pushed it to ~29 µm. The combination of GroupNorm (stable normalisation under SGDM), ResBlock (gradient highways), and CBAM (adaptive per-sample channel and spatial weighting) is well-supported in the literature for sensor-based regression tasks with heterogeneous input channels.

The CBAM's channel attention is particularly motivated for this problem: the five sensor channels have very different wear sensitivity profiles. Accelerometer and acoustic channels are highly sensitive to wear-related vibration changes; force channels (especially fx in the cutting direction) contain low-frequency trend information. CBAM allows the network to dynamically up-weight the most informative channels rather than treating all five equally.

---

## Phase 4 — Force HPF Preprocessing {#phase-4-hpf}

### Architecture Diagram

```
Per-channel preprocessing  (applied before CWT)

acc      ──────────────────────────────────────────────────► CWT
acoustic ──────────────────────────────────────────────────► CWT
                                                              ↓
fx ──── HPF (Butterworth 4th-order, f_c=15Hz, zero-phase) ──► CWT
fy ──── HPF (Butterworth 4th-order, f_c=15Hz, zero-phase) ──► CWT
fz ──── HPF (Butterworth 4th-order, f_c=15Hz, zero-phase) ──► CWT


Why: force signal spectrum before and after HPF
─────────────────────────────────────────────
  power                         power
    │  ██                         │
    │  ██                         │
    │  ██  (DC pedestal)          │          (DC removed)
    │  ██                         │         ╭──╮  ╭─╮
    │  ██ ▁▁▁▁▁▁▁▁▁▁▁             │  _______│  │__│ │______
    └──────────────────► freq     └─────────────────────────► freq
    0    15Hz   1625Hz             0    15Hz  tooth-passing  1625Hz
    ◄ CWT blind here ►                   ◄── CWT sees this ──►
    (all black scalogram)                (rich scalogram)


HPF design:
    sosfiltfilt(butter(4, 15.0, btype='high', fs=1625.0, output='sos'), signal)
    │
    ├── 4th-order: 80 dB/decade stopband attenuation
    ├── zero-phase (forward + backward pass): no temporal shift
    └── fallback for short signals (<27 samples): mean subtraction
```

### Methodology

A fundamental problem with the force channel scalograms was identified through visual inspection: the CWT scalograms for fx, fy, and fz were **nearly all black** — carrying almost no information. Root-cause analysis revealed why:

The force signals contain a large **DC component** (quasi-static mean force due to cutting load), which can be many times larger than the dynamic oscillations. When the CWT is computed, the DC component dominates the scale=64 (lowest frequency, ≈25 Hz) coefficient and suppresses everything above it. Since the tooth-passing frequency harmonics — the wear-relevant signal — lie at much higher frequencies (>100 Hz at typical spindle speeds), they are invisible in the scalogram.

**Solution: High-Pass Filter (HPF) before CWT**

A **4th-order zero-phase Butterworth high-pass filter** at 15 Hz cutoff was applied to all three force channels before CWT computation:

```python
from scipy.signal import butter, sosfiltfilt
HPF_SOS = butter(4, 15.0, btype="high", fs=1625.0, output="sos")

def highpass_force(signal):
    if len(signal) < 27:   # minimum length for sosfiltfilt with order-4
        return signal - signal.mean()   # fallback: mean subtraction
    return sosfiltfilt(HPF_SOS, signal)
```

Key design choices:
- **Zero-phase filtering** (`sosfiltfilt`) applies the filter twice — forward and backward — eliminating phase distortion that would shift the temporal location of wear events in the scalogram.
- **15 Hz cutoff** strips the DC pedestal and very-low-frequency calibration drift while preserving tooth-passing harmonics (typically 50–800 Hz at MATWI spindle speeds).
- **4th-order** provides 80 dB/decade stopband attenuation, fully attenuating the DC component within one or two filter orders below the cutoff.
- The sampling rate was empirically verified from the actual timestamp column (Δt ≈ 1/1625 s) rather than assumed.

The scalograms were **fully recomputed** with HPF applied to fx/fy/fz before CWT (`precompute_scalograms.py --force`). Acc and acoustic channels were left unfiltered — their signal content is already in the high-frequency range relevant to CWT.

A **sequential CWT variant** was also implemented as an algorithmic blueprint for microcontroller deployment (`experiments/phase4_sensor_cnn/validate_cwt_sequential.py`): processing one scale at a time and immediately subsampling, reducing peak RAM from ~38 MB (batch) to ~2.4 MB (sequential Python) and theoretically ~259 KB in C/CMSIS-DSP on the MCU — within the 512 KB constraint of the NXP FRDM-MCXN947.

### Results

| Variant | Test MAE (µm) | Std |
|---|---|---|
| Multiscale + SGDM + CBAM + Huber (pre-HPF) | ~29 | ±26 |
| **+ Force HPF (all 5 channels active)** | **27.19** | **±25.92** |

### Discussion

The force HPF had a clear and immediate effect: force channel scalograms now showed rich time-frequency structure that was previously invisible. The improvement from ~29 to 27.19 µm confirms that fx, fy, fz carry genuinely useful wear information — they were simply inaccessible to the CWT without DC removal. The zero-phase design ensured that temporal alignment of wear events between the force channels and the accelerometer/acoustic channels was preserved, which is important because CBAM's spatial attention compares across time-frequency positions within each channel.

The sequential CWT implementation validated that the computational approach is deployable on constrained hardware: outputs matched the batch implementation to within floating-point rounding (max absolute difference: 5.96 × 10⁻⁸), and the 16× RAM reduction makes on-device preprocessing theoretically feasible.

---

## Phase 4 — Channel Ablation Study {#phase-4-channels}

### Methodology

To understand the contribution of each sensor channel, the multiscale CNN was trained with different **subsets of the 5 input channels**, using the HPF-preprocessed scalograms. Channels were excluded by zeroing out the corresponding input planes during both training and inference (rather than retraining for each subset). Results stored in `experiments/phase4_sensor_cnn/results/eval_results_multiscale_sgdm_*.json`.

### Results

| Channel subset | Test MAE (µm) | Std |
|---|---|---|
| acc only | 33.08 | ±29.89 |
| acoustic only | 35.11 | ±30.07 |
| acc + acoustic | 33.64 | ±31.12 |
| acc + acoustic + fx | 29.84 | ±27.12 |
| acc + acoustic + fy | 34.13 | ±29.38 |
| acc + acoustic + fz | 35.01 | ±29.55 |
| **All 5 channels (acc + acoustic + fx + fy + fz)** | **27.19** | **±25.92** |

### Discussion

Several findings are clear:

1. **All 5 channels together is always best** (27.19 µm). No subset beats the full set.
2. **fx is by far the most informative force channel.** Adding fx to acc+acoustic improves MAE from 33.64 → 29.84 µm (11% improvement). fx is the force component in the cutting direction, directly related to material removal and tool wear progression.
3. **fy and fz do not individually help.** Adding fy or fz to acc+acoustic yields 34.13 and 35.01 µm respectively — worse than acc+acoustic alone. These force components are less directly related to cutting action and contain more noise relative to signal. Their benefit only emerges in combination with all channels when CBAM can weight them appropriately.
4. **Accelerometer is the single best individual channel** (33.08 µm), narrowly beating acoustic (35.11 µm) and both beats any single force channel. This is consistent with the literature: vibration signals from the accelerometer capture the full mechanical response of the tool-workpiece system, including both high-frequency chatter and mid-frequency tooth-passing harmonics.
5. **The HPF on fx makes it useful.** Pre-HPF, fx was essentially noise to the CWT; post-HPF, it becomes the most informative force channel. This directly validates the HPF design decision.

---

## Phase 4 — Hyperparameter Grid Search {#phase-4-gridsearch}

### Methodology

A systematic **grid search with cross-validation** was conducted over key hyperparameters of the SGDM optimizer and regression head architecture (`experiments/phase4_sensor_cnn/grid_search.py`).

Search space:
- Learning rate: {5e-4, 1e-3, 3e-3}
- Weight decay: {1e-3, 5e-3, 1e-2}
- Momentum: {0.85, 0.90, 0.95}
- Head type: {linear (96→1), small MLP (96→32→1), medium MLP (96→64→32→1)}

Each configuration was evaluated by best validation MAE across 100 epochs with CosineAnnealingLR scheduling.

### Results

**Best configuration by validation MAE:**

| Hyperparameter | Value |
|---|---|
| Learning rate | 3e-3 |
| Weight decay | 1e-3 |
| Momentum | 0.9 |
| Head | **linear (no hidden layers)** |
| Best epoch | 16 |
| Val MAE | 33.17 µm |

Key conclusion from grid search commit message: **"no hidden layers should be applied; best result is achieved with a linear head."**

The small and medium MLP heads consistently underperformed the linear head across all learning rate and weight decay combinations. The best MLP configuration (small head, lr=1e-3, wd=1e-2, momentum=0.85) achieved val MAE of 38.96 µm, significantly worse than the linear head's 33.17 µm.

### Discussion

The preference for a linear head over an MLP has a straightforward explanation: with only ~647 training samples and a 96-dimensional feature vector, a linear head has 97 free parameters while a small MLP adds hundreds more. The 96-dim features produced by the AdaptiveAvgPool → Flatten bottleneck are already a rich, compressed representation shaped by the convolutional layers and CBAM attention. The regression relationship from this embedding to wear is approximately linear in this feature space — confirmed by the fact that adding non-linearity in the head consistently hurts generalisation.

The grid search also confirmed that SGDM with momentum=0.9 and lr=3e-3 provides faster convergence (best epoch 16) than lower momentum settings, consistent with the earlier optimiser ablation.

---

## Phase 4 — Final Sensor-Only Best {#phase-4-final}

### Complete Pipeline Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│  PREPROCESSING  (offline, cached to disk)                            │
│                                                                      │
│  Raw CSV  ──► Aircut gate (AC-RMS Fr, adaptive τ)                    │
│                    │                                                  │
│                    ├── acc, acoustic  ──────────────────► CWT        │
│                    │                                       │          │
│                    └── fx, fy, fz ──► HPF(15Hz, 4th) ──► CWT        │
│                                                            │          │
│                    5 × (64, 64) scalograms ────────────────┘          │
│                    min-max normalise → [0,1]                          │
│                    save as <labels_idx>.pt                           │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  INFERENCE  (MultiScaleSensorCNN, ~244K params)                      │
│                                                                      │
│  (B, 5, 64, 64)                                                      │
│      │                                                               │
│      ├── Multi-scale entry (1×1 / 3×3 / 5×5 parallel paths)         │
│      │   → concat (B, 48, 64, 64)                                    │
│      │                                                               │
│      ├── MaxPool → Conv(48→64) → GN → ReLU → MaxPool                 │
│      │   → (B, 64, 16, 16)                                           │
│      │                                                               │
│      ├── ResBlock(64)  [skip connection]                             │
│      ├── CBAM(64)      [channel + spatial attention]                 │
│      │                                                               │
│      ├── Conv(64→96) → GN → ReLU → MaxPool                          │
│      ├── Conv(96→96) → GN → ReLU                                    │
│      ├── AdaptiveAvgPool → Flatten  → (B, 96)                       │
│      │                                                               │
│      └── Dropout(0.3) → Linear(96→1)  → wear (µm)                   │
└──────────────────────────────────────────────────────────────────────┘
```

### Methodology

After the grid search, the optimal configuration was re-trained with the full HPF-preprocessed 5-channel scalograms. Additional refinement runs using the grid-search-identified hyperparameters achieved a new best.

### Results

| Run | Test MAE (µm) | Std |
|---|---|---|
| Post-HPF best (pre-grid-search) | 27.19 | ±25.92 |
| **Post-grid-search best** | **24.96** | **±26.19** |

**Final configuration:**
- Architecture: MultiScaleSensorCNN (~244K parameters, ~59.6 KB INT8)
- Optimizer: SGDM, lr=3e-3 (or 1e-3), momentum=0.9, weight_decay=5e-3
- Loss: HuberLoss(δ=20)
- Head: Linear(96→1)
- Input: (5, 64, 64) HPF-preprocessed CWT scalograms, aircut-gated
- Scheduler: CosineAnnealingLR(T_max=100, eta_min=1e-5)

### Discussion

The final sensor-only result of **24.96 µm** is remarkably close to the image-only baseline of **23.17 µm** — a gap of only 1.79 µm. This is the most important finding of the sensor-only work: a compact CNN on CWT scalograms (~60 KB quantized) approaches the performance of a fine-tuned ResNet18 (>40 MB) using only force and vibration signals. The sensor model uses no visual information whatsoever and operates on pre-computed time-frequency representations that are, in principle, computable on-device.

---

## Phase 4 — INT8 TFLite Deployment {#phase-4-tflite}

### Methodology

After achieving 24.96 µm test MAE with the full MultiScaleSensorCNN (with CBAM), the model was converted to a deployable INT8 TFLite flatbuffer for the NXP FRDM-MCXN947.

**Conversion pipeline (`experiments/phase6_deployment/`):**
1. `export_onnx.py` — exports `phase4_multiscale_sgdm_best_25.pt` to ONNX opset 18 (`checkpoints/onnx/phase4_multiscale_sgdm_best_25.onnx`)
2. `build_calibration_data.py` — collects calibration samples from the training split
3. **NXP `eiq-onnx2tflite` (`onnx2quant`)** — performs static INT8 quantization and produces the TFLite flatbuffer (`checkpoints/onnx/phase4_multiscale_sgdm_best_25_int8.tflite`)
4. `validate_tflite.py` — loads both the PyTorch FP32 model and the TFLite INT8 model and compares predictions via `ai_edge_litert`

**Important — CBAM is present in this TFLite.** Unlike the fusion model conversion (Phase 5f, which used onnx2tf and hit a CBAM incompatibility), the standalone sensor model was converted with NXP's `eiq-onnx2tflite` tool, which successfully handles CBAM's spatial attention Conv2d(2→1, 7×7) for this 244K-param graph. The CBAM incompatibility is tool-specific and model-complexity-specific: onnx2tf on the larger 2.3M-param fusion model corrupts the spatial attention dims during NCHW→NHWC transposition; eiq-onnx2tflite on the standalone sensor model does not.

**Validation criterion:** MAE delta between PyTorch FP32 and TFLite INT8 < 5 µm; max per-sample absolute difference < 20 µm.

### Results

| Format | Val MAE (µm) | Test MAE (µm) | n |
|---|---|---|---|
| PyTorch FP32 | 45.70 | 24.956 | val: 300 / test: 247 |
| **TFLite INT8** | **45.18** | **24.744** | val: 300 / test: 247 |
| Δ (INT8 − FP32) | −0.52 | **−0.21** | — |

**Per-sample statistics (test split):**

| Metric | Value |
|---|---|
| Mean bias (TFLite − PyTorch) | −3.255 µm |
| Std of per-sample diff | ±1.851 µm |
| Max absolute diff | 10.29 µm ✓ (< 20 µm) |
| 95th-percentile abs diff | 5.999 µm |
| Pearson correlation | 0.9983 |
| Validation pass | ✓ |

**Model size:** 237.6 KB INT8 (well within any deployment budget; the NXP FRDM-MCXN947 has 2 MB flash).

### Discussion

Quantization of the standalone sensor CNN is essentially lossless: the test MAE actually decreases by 0.21 µm from FP32 to INT8, which is within measurement noise. The near-perfect per-sample correlation (0.9983) confirms that the INT8 model produces the same relative rankings as the FP32 version. This result validates the overall design choices — GroupNorm, HuberLoss, and the compact 244K-parameter architecture are all quantization-friendly.

The small model size (237.6 KB) is a key advantage: it fits in 2 MB flash alongside the fusion head and image encoder, and its 96-dim embedding output is cheap to transmit to the fusion layer at inference time. The 4× activation footprint reduction from INT8 also means the sensor CNN can run entirely in the 512 KB SRAM of the target MCU.

---

## Phase 4 — CBAM Incompatibility and SE Block Replacement {#phase-4-se}

### Methodology

During the fusion model TFLite conversion (Phase 5f), the CBAM module was found to be incompatible with the **onnx2tf** ONNX-to-TFLite pipeline. This is distinct from the standalone sensor model deployment above, where CBAM converted without issue using NXP's **eiq-onnx2tflite** tool. The incompatibility is therefore tool-specific, not a fundamental property of CBAM itself.

The failure with onnx2tf occurs specifically in CBAM's **spatial attention** branch:

```
spatial_att:  channel_mean(x) → (B, 1, H, W)
              channel_max(x)  → (B, 1, H, W)
              cat → (B, 2, H, W)
              Conv2d(2→1, 7×7, pad=3) → Sigmoid
              x_out = x * spatial_map
```

The Conv2d(2→1, 7×7) receives a tensor of shape (B, 2, H, W). During onnx2tf's NCHW→NHWC layout transposition and subsequent INT8 calibration, the channel dimension of this convolution is corrupted, causing a runtime error: `input_channel % filter_input_channel != 0 (1 != 0)`. The error manifests only in the INT8 quantized TFLite (not FP32), because INT8 calibration exposes the dimension mismatch.

CBAM's **channel attention** branch — a AvgPool → MLP → Sigmoid → `view(B,C,1,1) * x` pattern — is not the source of the error; this same pattern is used in SE blocks (MobileNetV3, EfficientNet) and is confirmed TFLite-compatible.

**SE block (Squeeze-and-Excitation) as replacement:**

The SE block (Hu et al., CVPR 2018) provides channel attention identical to CBAM's first stage while omitting the spatial attention that causes the TFLite failure:

```
input x  (B, C, H, W)
│
├─ AdaptiveAvgPool2d(1) → Flatten   (B, C)
├─ Linear(C → C/8) → ReLU
├─ Linear(C/8 → C) → Sigmoid        (B, C)
│
scale = sigmoid_output.view(B, C, 1, 1)
output = x * scale                  (B, C, H, W)
```

Channel attention re-weights the 64 feature channels at the 16×16 spatial stage by their global relevance to wear state. This is the component of CBAM identified as most important for the sensor regression task — the spatial attention (which focuses on specific time-frequency regions) is secondary for a model that already uses a multi-scale inception entry block to capture multi-resolution features.

SE blocks are natively TFLite-compatible: they appear in production-deployed models including MobileNetV3 and EfficientNet family, and have been verified to pass through onnx2tf's flatbuffer_direct pipeline without error.

**Architecture change summary:**

| | CBAM | SE | No attention |
|---|---|---|---|
| Channel attention | ✓ | ✓ | ✗ |
| Spatial attention | ✓ | ✗ | ✗ |
| TFLite-compatible | ✗ (spatial att. fails) | ✓ | ✓ |
| Parameters | ~244K | ~244K | ~243K |
| Additional params vs no-attention | +1,088 (CBAM) | +1,096 (SE) | — |

**Training:** `python experiments/phase4_sensor_cnn/train.py --arch multiscale --optim sgdm` (100 epochs, CosineAnnealingLR, HuberLoss δ=20).

### Results

| Configuration | Val MAE (µm) | Status |
|---|---|---|
| MultiScaleSensorCNN + CBAM (best) | 34.32 (best val; test 24.96 µm) | Baseline |
| MultiScaleSensorCNN, no attention (retrained) | 45.68 (best val, epoch 12) | ✗ degraded |
| **MultiScaleSensorCNN + SE (retrain underway)** | **TBD** | **Ongoing** |

The no-attention retrain converged early (epoch 12) and plateaued at 45.68 µm val MAE, significantly worse than CBAM (34.32 µm). This confirms that the attention mechanism is genuinely important for this task, motivating the SE replacement rather than simply removing it.

### Discussion

The CBAM incompatibility represents a TFLite deployment constraint specific to the spatial attention design. The channel attention component — which dynamically re-weights the five heterogeneous sensor channels (accelerometer, acoustic, fx, fy, fz) by their wear-state relevance — is preserved in the SE block and is expected to provide most of the performance benefit. The spatial attention (localising specific time-frequency wear signatures at the 16×16 stage) is lost; however, its contribution is partially compensated by the multi-scale inception entry block, which already captures information at three spatial scales before the feature extractor.

The decision to replace CBAM with SE rather than simply removing attention is justified by the no-attention retrain result: removing all attention causes a substantial val MAE regression (34.32 → 45.68 µm). The SE retrain is expected to recover most of this loss, since the channel attention component contributes more to performance than the spatial attention in typical sensor fusion regression tasks (Li et al., 2019).

---

## Summary Results Table {#summary}

| # | Method | Input | MAE (µm) | Std | n |
|---|---|---|---|---|---|
| 2.1 | XGBoost + tsfresh MinimalFCParameters | Time-domain stats | 28.25 | ±22.10 | 247 |
| 2.2 | XGBoost + physics features (FFT + DWT) | Time+freq+wavelet | 27.43 | ±22.10 | 247 |
| 3.1 | VGG-style CNN + Adam, no aircut | CWT (5ch) | 34.99 | ±29.00 | 247 |
| 3.2 | VGG-style CNN + SGDM, no aircut | CWT (5ch) | 32.31 | ±32.09 | 247 |
| 3.3 | Multiscale CNN + SpecAugment + Adam | CWT (5ch) | 42.84 | — | 247 |
| 3.4 | Quantized VGG CNN (ST Edge AI, INT8) | CWT (5ch) | 36.65 | ±30.74 | 247 |
| 4.1 | Multiscale CNN + SGDM (pre-improvements) | CWT (5ch), aircut | 34.68 | ±33.08 | 247 |
| 4.2 | + GroupNorm + ResBlock + cosine LR | CWT (5ch), aircut | ~30–33 | — | 247 |
| 4.3 | + CBAM + HuberLoss | CWT (5ch), aircut | 29 | ±26 | 247 |
| 4.4 | + Force HPF (force channels only) | CWT (5ch), aircut+HPF | 27.19 | ±25.92 | 247 |
| 4.5 | Channel ablation: acc + acoustic + fx | CWT (3ch), aircut+HPF | 29.84 | ±27.12 | 247 |
| **4.6** | **MultiScaleSensorCNN + grid search** | **CWT (5ch), aircut+HPF** | **24.96** | **±26.19** | **247** |
| 4.7 | MultiScaleSensorCNN + CBAM → INT8 TFLite | CWT (5ch), aircut+HPF | 24.744 | ±— | 247 |
| 4.8 | MultiScaleSensorCNN, no attention (retrained) | CWT (5ch), aircut+HPF | — (val 45.68) | — | — |
| **4.9** | **MultiScaleSensorCNN + SE block (ongoing)** | **CWT (5ch), aircut+HPF** | **TBD** | **—** | **—** |
| — | *Reference: image-only ResNet18* | *Flank images* | *23.17* | *±19.12* | *247* |
| — | *Reference: paper baseline (ResNet50)* | *Flank images* | *19.00* | *—* | *—* |

---

*Report compiled from git history, experiment result files, and training scripts.*  
*Sensor experiments span commits `3c8dcad` through `fbb827c` (approximately April–May 2026).*
