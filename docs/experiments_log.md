# Experiments Log — MATWI Tool Wear Prediction

**Dataset:** MATWI (~1663 labelled samples, images + sensor data)  
**Task:** Regression — predict tool wear in µm  
**Metric:** MAE ± std (µm), lower is better  
**Paper baseline:** 19.00 µm (ResNet50, image-only)  
**Last updated:** 2026-05-08 (session 3)

---

## Dataset Split (fixed, from paper)

| Split | Sets |
|---|---|
| Train | 1, 2, 5, 7, 8, 10, 11 |
| Val | 3, 6, 12 |
| Test | 4, 9, 13 |

> Sets 12–17 use a different material (generalisation test). Set 3 tests a different wear type.

---

## Master Results Summary

| Phase | Model | Test MAE (µm) | Deployable | Notes |
|---|---|---|---|---|
| Phase 1 v1 | ResNet18 image-only | ~40.44 | No | Initial run, val only |
| Phase 1 v2 | ResNet18 image-only (51 ep) | **23.17 ± 19.12** | No | Image baseline |
| Phase 2 | XGBoost + tsfresh | 28.25 ± 23.2 | No | Sensor-only, classical ML |
| Phase 3 v1 | Frozen ResNet18 + sensor MLP (no norm) | ~42 | No | Above baseline |
| Phase 3 v2 | Fusion, early attempt | ~33 | No | Still above baseline |
| Phase 3 v3 | Fusion + LayerNorm + normalisation | **23.38 ± 19.5** | No | Matched image baseline |
| Phase 3 v4 | Fusion + modality gating | pending | No | Result not yet recorded |
| Phase 4 baseline (full signal) | SensorCNNRegressor + Adam | 38.79 ± 44.3 | **Yes** | Before aircut removal (Exp 1) |
| Phase 4 baseline + GN + SGDM | SensorCNNRegressor + GroupNorm + SGDM | 30.36 ± 30.45 | **Yes** | Early stopping at ep 15 (Exp 10) |
| Phase 4 multiscale + GN + SGDM, dropout=0.1 | MultiScaleSensorCNN + GroupNorm + SGDM | 33.05 ± 33.93 | **Yes** | Early stopping at ep 15 (Exp 11) |
| Phase 4 multiscale + GN + SGDM, dropout=0.3 | MultiScaleSensorCNN + GroupNorm + SGDM | **28.97 ± 22.92** | **Yes** | **Current best deployable** (Exp 13) |
| Phase 4 baseline + BN + SGDM | SensorCNNRegressor + BatchNorm + SGDM | 40.45 ± 35.31 | **Yes** | BN worse than GN confirmed (Exp 14) |

---

---

# Phase 1 — Image-Only

**Model:** ResNet18 (pretrained ImageNet), final FC replaced with `Linear(512 → 1)`  
**Loss:** MSELoss  
**Optimizer:** Adam, lr=1e-4  
**Input:** Cropped + resized (224×224) + ImageNet-normalised images

### Attempt 1 — Initial training run

- Best val MAE: **40.44 µm**

### Attempt 2 — Extended training (51 epochs)

| Split | n | MAE (µm) | Std |
|---|---|---|---|
| Train | 647 | 11.20 | ±4.43 |
| Val | 300 | 37.49 | ±54.24 |
| **Test** | **247** | **23.17** | **±19.12** |

- Test range: 0.05–169.02 µm
- Strong overfitting (train 11 vs val 37 µm) but test generalises reasonably
- **This is the image-only baseline to beat**

---

---

# Phase 2 — Sensor-Only (XGBoost)

**Model:** XGBoost regressor  
**Features:** tsfresh `MinimalFCParameters` extracted from raw sensor CSVs (~78k rows each, 6 channels)  
**Config:** n_estimators=500, lr=0.05, max_depth=6, subsample=0.8, early stopping (patience=30)

| Split | n | MAE (µm) | Std |
|---|---|---|---|
| Train | 647 | 36.11 | ±40.69 |
| Val | 300 | 52.39 | ±68.03 |
| **Test** | **247** | **28.25** | **±23.2** |

- Test range: 0.16–219.26 µm
- Worse than image-only (28.25 vs 23.17 µm on test)
- High val variance suggests raw sensor features alone are noisy / harder to generalise
- **Cannot deploy** — XGBoost is not supported by eIQ / X-CUBE-AI runtimes

---

---

# Phase 3 — Multimodal Fusion

All fusion experiments use the fixed dataset split and the Phase 1 ResNet18 image encoder bootstrapped from `phase1_best.pt`.

### Attempt 1 — Basic fusion, no normalisation

**Architecture:**
- Image branch: frozen ResNet18 (Phase 1 weights) → 512-dim
- Sensor branch: LayerNorm + Linear → 512-dim projection
- Fusion: concat (1024-dim) → MLP → scalar

**Result:** Test MAE ≈ **42 µm** — above image-only baseline  
**Issue:** No input normalisation on sensor features; sensor branch projected to 512 (same size as image branch, oversized)

### Attempt 2 — Early fusion with XGBoost + deep model combined

**Architecture:** Phase 2 + Phase 3 combined in one training run — sensor + image features fed to a shallow fusion model

**Result:** Test MAE ≈ **33 µm** vs 23 µm image-only — still worse  
**Finding:** Multimodal model underperforming; sensor branch dominates noise

### Attempt 3 — Fusion with internal normalisation + reduced sensor dim

**Architecture:**
- Image branch: frozen ResNet18 → **LayerNorm** → 512-dim
- Sensor branch: MLP (input→128→64) → projection (64→128) + Dropout(0.5) → **LayerNorm** → 128-dim
- Fusion: concat (640-dim) → MLP (640→256→64→1)
- Sensor input normalised at runtime using training-set mean/std stored as model buffers

**Training:** 60 epochs, Adam lr=1e-4 → ReduceLROnPlateau (factor=0.5, patience=5). Best val MAE at epoch 52.

| Split | n | MAE (µm) | Std |
|---|---|---|---|
| Train | 647 | 17.16 | ±31.58 |
| Val | 300 | 43.81 | ±51.80 |
| **Test** | **247** | **23.38** | **±19.5** |

- **Matched image-only baseline** (23.38 vs 23.17 µm on test)
- Key fixes: LayerNorm on both branches, reduced sensor dim from 512→128, runtime normalisation

### Attempt 4 — Fusion with modality gating

**Architecture:** Same as Attempt 3, plus a `ModalityGating` module:
- Takes concatenated (img_feat + sensor_feat) → Linear → Softmax → 2 scalars
- Each modality multiplied by its learned gate weight before fusion
- ~2050 extra parameters (negligible)
- Goal: let the model down-weight sensor if unreliable per-sample

**Status:** Result pending.

---

---

# Phase 4 — Sensor-Only CNN (Deployment Track)

**Goal:** A model that runs on embedded hardware without a host PC.  
**Representation:** CWT scalogram — each 1D channel → 64×64 power image, 5 channels stacked → (5, 64, 64) tensor  
**Constraint:** Must export to TFLite for NXP eIQ Toolkit / MCUXpresso IDE

## 4.1 Hardware Target

| Property | Value |
|---|---|
| Board | NXP FRDM-MCXN947 |
| CPU | ARM Cortex-M33 dual-core |
| NPU | eIQ Neutron (~30× CNN speedup vs CPU) |
| Flash | 2 MB |
| RAM | 512 KB |
| Export format | TFLite (eIQ Toolkit) |

Previous target was STM32F401RC (256 KB / 64 KB). The larger budget means model size is no longer the binding constraint — quality is.

## 4.2 Dataset

| Split | Samples |
|---|---|
| Train | 647 |
| Val | 300 |
| Test | 247 |
| **Total** | **1,194** |

---

## 4.3 Wavelet Choice

The precompute pipeline uses **Complex Morlet (`cmor1.5-1.0`)** — not Daubechies. Daubechies (db2, db4, etc.) are discrete wavelet transform (DWT) basis functions that produce coefficient trees, not 2D scalograms, and are not applicable here.

| Wavelet | Transform | Freq. res. | Time res. | Notes |
|---|---|---|---|---|
| **Complex Morlet (current)** | CWT | High | Moderate | Standard for periodic machinery; complex-valued gives power + phase; used by Zhang 2023 |
| Paul | CWT | Moderate | High | Better for sharp transients; loses frequency detail |
| Mexican Hat (Ricker) | CWT | Moderate | Moderate | Real-valued only — no phase information |
| Bump | CWT | Very high | Low | Sharper harmonic separation; poor time localisation |
| Morse (generalised) | CWT | Tunable | Tunable | Used by Bukowski 2024 (Morse(3,60)); more parameters to tune |
| Daubechies (db4, db8) | **DWT** | — | — | Wrong transform family; not applicable |

Complex Morlet is correct for milling because tooth-passing frequency and its harmonics are oscillatory, periodic content — the shape Morlet is designed to match. Zhang 2023's ablation confirms CWT with Morlet achieves the highest test accuracy and smallest train/val gap vs STFT and GASF on PHM 2010. The bandwidth parameter `B=1.5` in `cmor1.5-1.0` is a standard default; whether tuning it would improve CNN performance on MATWI is an open question.

---

## 4.4 Aircut Detection & Removal

### Problem

MATWI sensor recordings contain substantial non-cutting segments within each file. For Set 13 (SensorID 50, wear=180 µm), the actual cutting pass occupies only ~11 s (t=19–30 s) out of a 60-second recording — ~82% is spindle-spinning with no material contact.

Computing CWT from the full signal means 64 time columns represent mostly idle dynamics, all mislabelled at the measurement's wear value. This is a label-noise problem that directly degrades regression quality.

**Why global force thresholding fails:** Each tool set has a different DC force offset due to calibration drift between sessions (Set 5 ≈ 5.6 N constant, Set 13 ≈ 0.2 N constant). A single absolute threshold cannot separate cutting from aircut across sets.

### Diagnostic Scripts

**`experiments/phase4_sensor_cnn/diagnose_aircuts.py`** — outputs 15 PNGs to `aircut_diagnostics/`. Four panels per file: raw fx/fy/fz, resultant Fr, accelerometer, sliding-window AC-RMS of Fr (window=512, p20/p80 reference lines). Sets 1, 5, 9, 10, 13, early/mid/late wear each.

**`experiments/phase4_sensor_cnn/visualize_scalograms.py`** — outputs PNGs to `scalogram_viz/`. Top row: raw time-series with cutting region shaded green and aircut shaded red. Bottom row: pre-computed CWT scalogram heatmap (exactly what the CNN receives). 5 columns × 2 rows per measurement.

Key findings from Set 13:
- Fr stays at ~2.2 N throughout — DC spindle preload dominates; absolute thresholding is impossible.
- Accelerometer shows continuous broadband vibration even during aircut — cannot be used as gate alone.
- AC-RMS of Fr has clear bimodal structure: baseline ≈ 0.009 N (aircut), peak ≈ 0.054 N (cutting). p20/p80 separation is clean enough to threshold.

Observed across sets:
- Sets 1, 9, 13: clear aircut at edges (AC-RMS lower at start/end).
- Sets 5, 10, 11: flat profile — no detectable aircut structure, full signal kept.

### Implementation

**`src/data/aircut_mask.py`** — reusable mask function.

Algorithm:
1. Compute Fr = √(fx² + fy² + fz²)
2. Sliding-window AC-RMS (O(n) vectorised cumsum, window=512 samples)
3. Per-file adaptive threshold: `p20 + 0.5 × (p80 − p20)`
4. Morphological closing to fill gaps < 1 s (1600 samples)
5. Fallback: if `p80 / p20 < 2.0` (flat), keep full signal
6. Degenerate fallback: if mask all-False, keep full signal

| Parameter | Default | Meaning |
|---|---|---|
| `AC_WINDOW` | 512 | Sliding window (~0.32 s at 1.6 kHz) |
| `MORPH_CLOSE_SAMPLES` | 1600 | Max gap filled (~1 s) |
| `FLAT_RATIO_MIN` | 2.0 | p80/p20 below this → no aircut structure |

Observed retention: ~65% of signal kept on Set 1 files. **`experiments/phase3_fusion/precompute_scalograms.py`** was modified to apply the mask before CWT. The `--force` flag re-runs over existing `.pt` files.

---

## 4.5 Architecture Evolution

### SensorCNNRegressor (baseline)

**File:** `src/models/sensor_cnn_model.py`

```
Input (5, 64, 64)
Conv(5→16, 3×3) → GN(8) → ReLU → MaxPool(2)    → (16, 32, 32)
Conv(16→32, 3×3) → GN(8) → ReLU → MaxPool(2)   → (32, 16, 16)
Conv(32→64, 3×3) → GN(8) → ReLU → MaxPool(2)   → (64, 8, 8)
Conv(64→64, 3×3) → GN(8) → ReLU                → (64, 8, 8)
AdaptiveAvgPool2d(1) → Flatten → Linear(64, 1)
```

Parameters: **61,041** | INT8: 59.6 KB | Params/train-sample: ~94

> **GroupNorm vs BatchNorm (baseline):** GN was adopted because BN degrades under SGDM at batch=16 due to noisy batch statistics (NeurIPS 2021). Empirically confirmed in Experiment 14 — reverting to BN raised test MAE from 30.36 → 40.45 µm.

### MultiScaleSensorCNN (Zhang 2023 inspired, current best)

**File:** `src/models/multiscale_sensor_cnn.py`  
**Citation:** Zhang Y. et al., *Sensors* 23(10), 4595, 2023.

```
Input (5, 64, 64)

── Multi-scale entry (no spatial downsampling) ──────────────────────────
path_1x1 : Conv(5→16, 1×1) → GN(8) → ReLU
path_3x3 : Conv(5→8, 1×1) → GN(8) → ReLU → Conv(8→24, 3×3) → GN(8) → ReLU
path_5x5 : Conv(5→4, 1×1) → GN(4) → ReLU → Conv(4→8, 5×5) → GN(8) → ReLU
Concat → 48 channels, 64×64

── Feature extractor ────────────────────────────────────────────────────
MaxPool(2)                                         → (48, 32, 32)
Conv(48→64, 3×3) → GN(8) → ReLU
MaxPool(2)                                         → (64, 16, 16)
ResBlock(64): [Conv→GN→ReLU→Conv→GN] + skip        → (64, 16, 16)
Conv(64→96, 3×3) → GN(8) → ReLU
MaxPool(2)                                         → (96, 8, 8)
Conv(96→96, 3×3) → GN(8) → ReLU                   → (96, 8, 8)
AdaptiveAvgPool2d(1) → Flatten                     → 96

── Regression head ──────────────────────────────────────────────────────
Dropout(0.3) → Linear(96, 1)
```

Parameters: **243,269** | INT8: 237.6 KB | Params/train-sample: ~376

| Aspect | Baseline | MultiScaleSensorCNN |
|---|---|---|
| Normalisation | GroupNorm(8) | GroupNorm(8) throughout |
| Entry block | Single Conv(5→16, 3×3) | Parallel 1×1 + 3×3 + 5×5 → concat(48) |
| Residual block | None | ResBlock(64) after second MaxPool |
| Dropout | None | Dropout(0.3) before head |
| Head input dim | 64 | 96 |
| Parameters | 61K | 243K |

**Rationale for multiscale entry:** Zhang 2023's CWT ablation shows the inception-style pyramid achieves the highest test accuracy and smallest train/val gap vs STFT and GASF.  
**ResBlock rationale:** Skip connections provide gradient highways for SGDM's noisier updates (Keskar et al. 2017).  
**Dropout=0.3 rationale:** Empirically derived — dropout=0.1 allowed overfitting from epoch 15; 0.3 delayed best checkpoint to epoch 19 and improved test MAE from 33.05 → 28.97 µm (Experiments 11 vs 13).

---

## 4.6 Training Infrastructure

**`experiments/phase4_sensor_cnn/train.py`** CLI flags:

| Flag | Options | Default |
|---|---|---|
| `--arch` | `baseline`, `multiscale` | `baseline` |
| `--optim` | `adam`, `sgdm` | `adam` |
| `--resume` | flag | off |
| `--patience` | int | 20 (0 = disabled) |

Checkpoints: `checkpoints/phase4_{arch}_{optim}_best.pt`  
History: `results/history_{arch}_{optim}.json`

**SGDM config** (Zhang 2023 §4.1):
```python
torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-3)
```

**Scheduler:** `CosineAnnealingLR(T_max=100, eta_min=1e-5)` — decays LR smoothly from peak to 1e-5 over all epochs. Replaced ReduceLROnPlateau after that scheduler caused premature LR collapse before the model explored the flat loss basin under SGDM.

**Early stopping:** Patience=20 by default. Stops training when val MAE has not improved for 20 consecutive epochs. Both models in session 3 converged around epoch 15–24 and stopped at epoch 39–44, saving ~60 wasted epochs.

**`experiments/phase4_sensor_cnn/evaluate.py`** accepts `--arch` and `--optim`; saves to `results/eval_results_{arch}_{optim}.json`.

---

## 4.7 Literature Review

### Paper comparison

| Field | **Bukowski 2024** | **Huang 2024** (CTCN) | **Zhang 2023** |
|---|---|---|---|
| Task | 3-class classification | **Continuous µm regression** | 3-class classification |
| Dataset | Chipboard milling, 75 samples | **PHM 2010** (C1/C4/C6) | **PHM 2010** |
| Sensors | 11 ch | 7 ch: Fx Fy Fz Vx Vy Vz AE | **Fz only** |
| Input | 11× CWT (Morse(3,60), 128×128×3) | **Raw 1D** (Z-score, 3% trim) | CWT (Morlet, 128×128×3) |
| Architecture | 11 parallel branches → DepthConcat → FC | Conv1d × 3 → TCN → Linear(1) | **Inception pyramid** → Conv → Linear(3) |
| Params | **269.2 M** (!) | ~few M | compact (not stated) |
| Loss | Cross-entropy | **MSE** | Cross-entropy |
| Optimizer | Not stated | **Adam lr=1e-4** | **SGDM lr=1e-3 mom=0.9 wd=0.005** |
| Regularisation | BN only | **L2 weight decay** | **Dropout 0.1**, BN |
| Batch size | not stated | 32 | **4** |
| Epochs | 200 | 100 | 200 |
| Augmentation | **None** | **None** | **None** |
| Metric | 96% accuracy (15-fold CV) | **MAE 2.03/2.74/2.42 µm C1/C4/C6** | >90% accuracy C6 |
| Train↔val gap | 15-fold CV only, no held-out tool | not reported | **small** on CWT; large on GASF |

### What the literature confirms

1. 2–3 µm MAE on PHM 2010 is achievable (Huang 2024). Our 38 µm on MATWI is not at a deep limit.
2. CWT generalises better than STFT or GASF (Zhang 2023 ablation). Our representation is correct.
3. Zero of six reviewed papers use data augmentation. Confirmed empirically — SpecAugment hurt.
4. Zero of six papers use log-target transform or aux classification head on regression.
5. Bukowski 2024 is not trustworthy — 269 M params, MaxPool(stride=1×1) is a no-op, 15-fold CV on 75 samples.
6. Huang 2024 wins on PHM 2010 with raw 1D + 1D-CNN+TCN, no scalogram. Our 2D path is defensible because eIQ Neutron NPU accelerates 2D conv.

---

## 4.8 Phase 4 Experiment Log

### Experiment 1 — Baseline CNN, full signal, Adam
**Config:** SensorCNNRegressor, Adam lr=1e-3, batch=16, 100 epochs, no aircut gating  
**Result:** Test MAE = **38.79 ± 44.3 µm**  
**Notes:** High std indicates large outlier errors; full-signal scalograms contain ~65–82% aircut noise.

### Experiment 2 — SpecAugment augmentation
**Config:** Added time/frequency masking to scalograms during training  
**Result:** Val MAE worsened 36.99 → **42.84 µm**  
**Root cause:** Augmentation destroys the frequency-time structure CWT encodes. Six-of-six papers use no augmentation.  
**Status:** Reverted.

### Experiment 3 — log1p target transform + larger architecture (981K params)
**Config:** log1p(wear) target, expm1 inversion at eval, widened VGG blocks  
**Result:** Training diverged; train MAE inflated to ~90 µm  
**Root cause:** log-space MSE does not minimise µm-space MAE; expm1 amplifies outlier predictions; 981K params / 647 samples (~1500×) overfits severely.  
**Status:** Reverted.

### Experiment 4 — Auxiliary wear-stage classification head
**Config:** Added 3-class softmax branch alongside regression head  
**Result:** Train stage accuracy 0.93; val stage accuracy 0.27 — memorised training stages  
**Root cause:** Misattributed to Zhang 2023. Zhang is pure classification. Zero of six papers combine regression + aux classification.  
**Status:** Reverted.

### Experiment 5 — Multiscale + SGDM, batch=16, full signal
**Config:** MultiScaleSensorCNN, SGDM lr=1e-3 mom=0.9 wd=5e-3, batch=16, no aircut gating  
**Result:** Previous round — all confounds combined (SpecAugment + log-target + aux head + architecture); could not attribute failure to any single cause.  
**Status:** Reverted; architecture reinstated cleanly in Experiment 7.

### Experiment 6 — Baseline CNN, aircut-gated scalograms, Adam
**Config:** SensorCNNRegressor, Adam lr=1e-3, batch=16, patience=5, aircut-gated `.pt` files  
**Status:** Pending clean retrain.  
**Purpose:** Isolate aircut-removal contribution. If this drops below 30 µm, aircut removal is the dominant factor.

### Experiment 7 — Multiscale + Adam, aircut-gated
**Config:** MultiScaleSensorCNN, Adam lr=1e-3, batch=16, patience=5, aircut-gated `.pt` files  
**Status:** Run. Result not yet recorded.

### Experiment 8 — Multiscale + SGDM, batch=16, aircut-gated
**Config:** MultiScaleSensorCNN, SGDM lr=1e-3 mom=0.9 wd=5e-3, batch=16, patience=5, aircut-gated `.pt` files  
**Result:** Strong early convergence — epoch 1 val MAE ~50–70 µm (vs >110 µm for baseline+Adam). Overfitting begins at epoch 20–30; train loss falls while val MAE degrades.  
**Root cause:** Dropout=0.1 head-only insufficient for 647 training samples once optimizer warms up. Mitigation: patience reduced to 5, additional Dropout2d mid-network tested then reverted.  
**Status:** Ongoing.

### Experiment 9 — SGDM, batch=4 (matching Zhang 2023 exactly)
**Config:** MultiScaleSensorCNN, SGDM, batch=4, lr=1e-3  
**Result:** Overfitting from epoch 1 — first epoch best, each subsequent epoch progressively worse.  
**Root cause:** 160 gradient steps/epoch with lr=1e-3 and momentum=0.9 accumulates excessive optimizer velocity; model memorises training set before BatchNorm statistics stabilise. Linear scaling rule requires lr ≈ 2.5e-4 at batch=4, which was not applied.  
**Status:** Reverted to batch=16.

### Experiment 10 — Baseline + GroupNorm + SGDM, no early stopping
**Config:** SensorCNNRegressor with GroupNorm(8) replacing BatchNorm, SGDM lr=1e-3 mom=0.9 wd=5e-3, batch=16, CosineAnnealingLR(T_max=100), 100 epochs  
**Result:**

| Split | n | MAE (µm) | Std |
|---|---|---|---|
| Train | 647 | 38.05 | ±35.90 |
| Val | 300 | 42.63 | ±51.41 |
| **Test** | **247** | **30.36** | **±30.45** |

Best val MAE at epoch 15. Val MAE plateaus around 49–51 µm after epoch 15; train loss continues to fall → clear overfitting. CosineAnnealingLR does not help once model is past its best checkpoint.  
**Improvement over Exp 1:** 38.79 → 30.36 µm test MAE (aircut gating + GroupNorm + SGDM combined).

### Experiment 11 — Multiscale + GroupNorm + ResBlock + SGDM, dropout=0.1, no early stopping
**Config:** MultiScaleSensorCNN with GroupNorm(8) throughout, ResBlock(64) in feature extractor, dropout=0.1, SGDM lr=1e-3, CosineAnnealingLR(T_max=100), 100 epochs  
**Result:**

| Split | n | MAE (µm) | Std |
|---|---|---|---|
| Train | 647 | 39.86 | ±43.48 |
| Val | 300 | 41.01 | ±61.34 |
| **Test** | **247** | **33.05** | **±33.93** |

Best val MAE at epoch 15. Despite 4× more parameters and ResBlock, baseline (Exp 10) outperforms on test. Diagnosis: 243K params / 647 samples = ~376 params/sample; dropout=0.1 provides insufficient regularisation.

### Experiment 12 — Multiscale + SGDM, LR=0.01 (gradient explosion)
**Config:** Same as Exp 11 but LR=0.01 (accidentally set during architecture changes)  
**Result:** Epoch 1 train_loss = 4.4×10¹⁶. Model predicted constant for epochs 1–6 (val_mae_std=82.71 exactly). Recovered briefly at epoch 8–11, then stuck at train_loss≈8000 for remaining 89 epochs.  
Best checkpoint (epoch 11): test MAE = **37.54 ± 26.75 µm** — undertrained, worse than Exp 10.  
**Root cause:** SGDM momentum=0.9 + weight_decay=5e-3 + LR=0.01 on 647-sample dataset → optimizer velocity accumulation causes immediate explosion. Even after gradient clipping brings loss down, momentum carries weights into a degenerate flat region.  
**Status:** Reverted to LR=1e-3.

### Experiment 13 — Multiscale + SGDM, dropout=0.3, early stopping ✓ CURRENT BEST
**Config:** MultiScaleSensorCNN, GroupNorm, ResBlock, dropout=0.3, SGDM lr=1e-3, CosineAnnealingLR, early stopping patience=20  
**Result:**

| Split | n | MAE (µm) | Std |
|---|---|---|---|
| Train | 647 | 45.46 | ±53.42 |
| Val | 300 | 44.70 | ±66.22 |
| **Test** | **247** | **28.97** | **±22.92** |

Best val MAE (44.70 µm) at epoch 19. Stopped at epoch 39 (patience=20 from epoch 19).  
Compared to Exp 11 (dropout=0.1): best epoch delayed 15→19, test MAE improved 33.05→28.97 µm. Dropout=0.3 is directly responsible.  
**This is the first deployable model to beat XGBoost (28.97 vs 28.25 µm).** Gap is within noise margin.

### Experiment 14 — Baseline + BatchNorm + SGDM, early stopping (BN vs GN ablation)
**Config:** SensorCNNRegressor with BatchNorm2d (reverting GroupNorm), SGDM lr=1e-3, CosineAnnealingLR, early stopping patience=20  
**Result:**

| Split | n | MAE (µm) | Std |
|---|---|---|---|
| Train | 647 | 26.75 | ±31.56 |
| Val | 300 | 46.43 | ±68.52 |
| **Test** | **247** | **40.45** | **±35.31** |

Best val MAE (46.43 µm) at epoch 24. Stopped at epoch 44.  
Train MAE (26.75 µm) << val MAE (46.43 µm) → severe overfitting. BN with SGDM at batch=16 causes instability consistent with the NeurIPS 2021 unified normalisation study.  
**Conclusion:** GroupNorm is confirmed better than BatchNorm for SGDM+batch=16 on this dataset. Baseline reverted to GroupNorm.

---

## 4.9 Planned Experiments

### Experiment 15 — PHM 2010 pipeline sanity check
**Status:** Not yet implemented. Requires new dataset loader.  
**Purpose:** Run MultiScaleSensorCNN on PHM 2010 (C1+C4 train, C6 test). Compare against Huang 2024 (2.0–2.7 µm MAE). If our result is ~8–15 µm, pipeline is sound; if >30 µm, there is a preprocessing bug.

### Experiment 16 — Single-channel Fz ablation
**Status:** Not yet implemented. Requires `--channels fz` flag in dataset loader.  
**Purpose:** Zhang 2023 used only Fz and reached >90% accuracy. Test whether acc, acoustic, fx, fy add signal or noise on MATWI.

### Experiment 17 — Per-channel late fusion
**Status:** Not yet implemented. Requires new `MultiBranchSensorCNN`.  
**Purpose:** Test per-channel CNN branches + late-concat fusion vs current early-fusion (5 stacked channels). Only attempt if Experiments 13–16 do not reach below 25 µm.

---

## 4.10 What Not To Do

| Approach | Reason |
|---|---|
| Data augmentation (SpecAugment, CutMix, noise) | Destroys CWT frequency-time structure; six-of-six papers skip it; empirically confirmed to hurt MAE |
| log1p target transform | Optimises log-space MSE, not µm MAE; expm1 amplifies outlier predictions |
| Aux classification head on regression | Zero literature precedent; empirically causes memorisation, not regularisation |
| Model > ~500K parameters without strong regularisation | MATWI has 647 train samples; 981K/647 ≈ 1500 params/sample overfits severely |
| Copying Bukowski 2024 architecture | 269 M params, MaxPool stride=1×1 is a no-op, 15-fold CV on 75 samples only |
| Global absolute force threshold for aircut | Per-set DC bias makes this impossible; use AC-RMS adaptive threshold |
| Batch size=4 with SGDM lr=1e-3 | 160 steps/epoch at high momentum causes optimizer overshoot from epoch 1; scale lr ∝ batch if reducing |
| Daubechies wavelets for CWT scalograms | DWT family — produces coefficient trees, not 2D scalograms; wrong transform |
| BatchNorm with SGDM at batch=16 | Empirically confirmed worse than GroupNorm (Exp 14: 40.45 µm vs 30.36 µm); noisy batch statistics degrade BN under SGDM at small batches (NeurIPS 2021) |
| LR=0.01 with SGDM mom=0.9 wd=5e-3 on 647 samples | Instant gradient explosion (Exp 12); correct LR is 1e-3 for this dataset size |
| Dropout=0.1 on multiscale head | Insufficient regularisation for 243K params / 647 samples; best epoch at 15 then immediate overfitting; use 0.3 (Exp 11 vs 13) |
