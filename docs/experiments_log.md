# Experiments Log — MATWI Tool Wear Prediction

**Dataset:** MATWI (~1663 labeled samples, images + sensor data)  
**Task:** Regression — predict tool wear in µm  
**Metric:** MAE ± std (µm), lower is better  
**Paper baseline:** 19.00 µm  

## Dataset Split (fixed, from paper)

| Split | Sets |
|-------|------|
| Train | 1, 2, 5, 7, 8, 10, 11 |
| Val | 3, 6, 12 |
| Test | 4, 9, 13 |

> Sets 12–17 use a different material (generalization test). Set 3 tests a different wear type.

---

## Phase 1 — Image-Only Baseline

**Model:** ResNet18 (pretrained ImageNet), final FC replaced with `Linear(512 → 1)`  
**Loss:** MSELoss  
**Optimizer:** Adam, lr=1e-4  
**Input:** Cropped + resized (224×224) + ImageNet-normalized images  

### Attempt 1 — Initial training run
- Best val MAE: **40.44 µm**

### Attempt 2 — Extended training (51 epochs)

| Split | n | MAE (µm) | Std |
|-------|---|----------|-----|
| Train | 647 | 11.20 | ±4.43 |
| Val | 300 | 37.49 | ±54.24 |
| **Test** | **247** | **23.17** | **±19.12** |

- Test range: 0.05–169.02 µm
- Strong overfitting (train 11 vs val 37), but test generalises reasonably
- **This became the baseline to beat**

---

## Phase 2 — Sensor-Only Baseline

**Model:** XGBoost regressor  
**Features:** tsfresh `MinimalFCParameters` extracted from raw sensor CSVs (~78k rows each, 6 channels)  
**Config:** n_estimators=500, lr=0.05, max_depth=6, subsample=0.8, early stopping (patience=30)

| Split | n | MAE (µm) | Std |
|-------|---|----------|-----|
| Train | 647 | 36.11 | ±40.69 |
| Val | 300 | 52.39 | ±68.03 |
| **Test** | **247** | **28.25** | **±23.2** |

- Test range: 0.16–219.26 µm
- **Worse than image-only** (28.25 vs 23.17 µm on test)
- High variance on val suggests sensor data alone is noisy / harder to generalise

---

## Phase 3 — Multimodal Fusion

All fusion experiments use the same dataset split and the Phase 1 ResNet18 image encoder bootstrapped from `phase1_best.pt`.

### Attempt 1 — Basic fusion, no normalization
**Architecture:**
- Image branch: frozen ResNet18 (from Phase 1 weights) → 512-dim
- Sensor branch: LayerNorm + Linear → 512-dim projection
- Fusion: concat (1024-dim) → MLP → scalar

**Result:** Test MAE ≈ **42 µm** — above the image-only baseline  
**Issue identified:** No input normalisation on sensor features; sensor branch projected to 512 (same size as image)

---

### Attempt 2 — Early fusion with early XGBoost + deep model
**Architecture:** (Phase 2+3 combined in one training run)
- Combined sensor + image features fed to a shallow fusion model

**Result:** Test MAE ≈ **33 µm** vs 23 µm image-only — still worse  
**Finding:** Multimodal model was underperforming at this stage

---

### Attempt 3 — Fusion with internal normalisation + reduced sensor dim
**Architecture:**
- Image branch: frozen ResNet18 → **LayerNorm** → 512-dim
- Sensor branch: MLP (input→128→64) → projection (64→128) + Dropout(0.5) → **LayerNorm** → 128-dim
- Fusion: concat (640-dim) → MLP (640→256→64→1)
- Sensor input normalised at runtime using training-set mean/std (stored as model buffers)

**Training:** 60 epochs, Adam lr=1e-4 → ReduceLROnPlateau (factor=0.5, patience=5)  
Best val MAE reached at epoch 52: **43.81 µm**

| Split | n | MAE (µm) | Std |
|-------|---|----------|-----|
| Train | 647 | 17.16 | ±31.58 |
| Val | 300 | 43.81 | ±51.80 |
| **Test** | **247** | **23.38** | **±19.5** |

- **Matched image-only baseline** (23.38 vs 23.17 µm on test)
- Key fixes: LayerNorm on both branches, reduced sensor dim from 512→128, runtime normalisation

---

### Attempt 4 — Fusion with modality gating (current)
**Architecture:** Same as Attempt 3, plus a `ModalityGating` module:
- Takes concatenated (img_feat + sensor_feat) → Linear → Softmax → 2 scalars
- Each modality is multiplied by its learned weight before fusion
- ~2050 extra parameters (negligible)
- Goal: let the model learn to down-weight sensor if unreliable

**Status:** Training in progress / just committed — results pending

---

## Summary Table

| Experiment | Model | Test MAE (µm) | Notes |
|------------|-------|---------------|-------|
| Phase 1 v1 | ResNet18 (image-only) | ~40.44 | Initial run, val only |
| Phase 1 v2 | ResNet18 (image-only, 51 ep) | **23.17 ± 19.12** | Baseline |
| Phase 2 | XGBoost + tsfresh | 28.25 ± 23.2 | Sensor-only, worse |
| Phase 3 v1 | Frozen ResNet18 + sensor MLP (no norm) | ~42 | Above baseline |
| Phase 3 v2 | Fusion (early attempt) | ~33 | Still above baseline |
| Phase 3 v3 | Fusion + LayerNorm + sensor normalisation | **23.38 ± 19.5** | Matched baseline |
| Phase 3 v4 | Fusion + gating layer | TBD | Current |

**Paper baseline:** 19.00 µm (ResNet50 image-only)
