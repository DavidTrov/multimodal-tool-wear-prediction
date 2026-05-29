# Abandoned Approaches

This document records architectures, training strategies, and design choices that were tried during the thesis and either failed or were superseded. It serves as a reference for the thesis discussion section and prevents re-exploration of dead ends.

For full per-experiment details, see `docs/experiments_log.md` (Phase 4, Section 4.8 and 4.10).

> **Pre-restructure tag:** `git show pre-restructure:<path>` recovers any deleted file.

---

## 1. SpecAugment on CWT Scalograms

**Experiment 2 (Phase 4)**

Applied time-masking and frequency-masking augmentation to CWT scalograms during training, following speech-processing literature.

- **Result:** Val MAE worsened from 37 to 43 um.
- **Why it failed:** CWT scalograms encode frequency-time structure that SpecAugment destroys. Zero of six reviewed papers on CWT-based tool wear prediction use data augmentation.
- **Reverted immediately.**

---

## 2. log1p Target Transform

**Experiment 3 (Phase 4)**

Applied `log1p(wear)` during training with `expm1` inversion at evaluation, paired with a wider architecture (981K params).

- **Result:** Training diverged; train MAE inflated to ~90 um.
- **Why it failed:** Log-space MSE does not minimise um-space MAE. `expm1` amplifies outlier predictions. The 981K-param model with only 647 training samples severely overfits.
- **Reverted immediately.**

---

## 3. Auxiliary Wear-Stage Classification Head

**Experiment 4 (Phase 4)**

Added a 3-class softmax branch (low/medium/high wear) alongside the regression head, inspired by a misreading of Zhang 2023 (which is pure classification, not regression + classification).

- **Result:** Training stage accuracy 0.93; validation accuracy 0.27 (memorised).
- **Why it failed:** No precedent in the literature for combining regression with auxiliary classification on this task. The classification head learned to memorise the training set without generalising.
- **Reverted immediately.**

---

## 4. BatchNorm with SGDM at Small Batch Size

**Experiment 14 (Phase 4)**

Tested BatchNorm2d in place of GroupNorm(8) on the SensorCNNRegressor with SGDM, batch=16.

- **Result:** Test MAE = 40.45 um (vs 30.36 um with GroupNorm).
- **Why it failed:** BN statistics are unstable at batch=16 under SGDM momentum, consistent with the NeurIPS 2021 unified normalisation study. GroupNorm is definitively better for this setting.
- **GroupNorm(8) adopted as permanent choice.**

---

## 5. High Learning Rate (LR=0.01)

**Experiment 12 (Phase 4)**

Accidentally set LR=0.01 (10x intended) with SGDM momentum=0.9 and weight_decay=5e-3.

- **Result:** Epoch 1 train loss = 4.4e16. Gradient explosion. Best checkpoint (epoch 11): test MAE = 37.54 um.
- **Why it failed:** Optimizer velocity accumulation under SGDM with momentum at high LR on a 647-sample dataset causes immediate explosion.
- **Reverted to LR=1e-3.**

---

## 6. Stacked ResBlocks

**Experiments 15-16 (Phase 4)**

Tested adding a second ResBlock at the 96-channel (8x8) stage (Exp 15, 409K params) and stacking two ResBlocks at the 64-channel (16x16) stage (Exp 16, 317K params).

- **Result:** Both worse than single ResBlock (Exp 13, 243K params).
- **Why it failed:** Additional depth overfits on 647 samples without adding representational benefit. The 8x8 spatial stage is too small for a residual block to capture useful patterns.
- **Single ResBlock(64) at 16x16 retained as optimal.**

---

## 7. Two-Layer Regression Head

**Experiment 18 (Phase 4)**

Replaced the single-linear head with `Linear(96,32) -> ReLU -> Dropout(0.1) -> Linear(32,1)`.

- **Result:** No improvement over single-linear head.
- **Why it failed:** The non-linear bottleneck adds parameters without benefit at this dataset size. CBAM's contribution could not be isolated in this combined test.
- **Single-linear head retained.**

---

## 8. CBAM Spatial Attention

**Experiments 19-27 (Phase 4); Phase 5 compression**

CBAM (Convolutional Block Attention Module) was initially adopted because it improved sensor-only accuracy (CBAM FP32: 24.96 um vs SE FP32: 29.27 um, a 4.31 um gap).

- **Result:** CBAM is incompatible with TFLite conversion via the `onnx2tf` pipeline. The spatial attention branch (`Conv2d(2->1, 7x7)` applied to concatenated channel-mean and channel-max maps) causes a dimension corruption during NCHW->NHWC transposition, producing a runtime error that does not occur in the FP32 variant.
- **Resolution:** Replaced with SE (Squeeze-and-Excitation) blocks. SE is natively compatible with TFLite-deployed models like MobileNetV3. The 4.31 um accuracy gap represents an architectural compatibility cost.
- **Future direction (thesis Section VI):** A TFLite-compatible CBAM implementation would recover this gap.

---

## 9. Hybrid Sensor CNN

**Codebase only (never integrated into experiments)**

`src/models/hybrid_sensor_cnn.py` — an experimental architecture combining a CWT scalogram branch (accelerometer + acoustic only, 2 channels) with a force scalar feature MLP (69 physics features), concatenated before a regression head.

- **Result:** Never trained or evaluated in a full experiment.
- **Why abandoned:** The CWT-based MultiScaleSensorCNN on all 5 channels outperformed approaches that separated modalities. The 69-dimensional physics feature MLP overfits at 647 samples. Aircut gating was not yet implemented when this model was designed.
- **Superseded by:** `MultiScaleSensorCNN` operating on 5-channel CWT scalograms.

---

## 10. Intermediate Feature Fusion (Naive Concatenation)

**Phase 3, Attempt 1 (thesis Section II.G.1)**

Concatenated 512-dim frozen ResNet18 embedding with 64-dim sensor MLP embedding (from tsfresh features), passed through a three-layer MLP fusion head.

- **Result:** Test MAE = 30.64 +/- 26.92 um (vs image-only 23.17 um).
- **Why it failed:** No input normalisation on sensor features. Sensor branch projected to 512 dimensions (same as image branch, oversized for 50-dim input). Sensor branch gradient magnitudes differed by orders of magnitude from image branch.
- **Superseded by:** Decision-level fusion, then two-tower intermediate fusion.

---

## 11. Decision-Level Blend Fusion

**Phase 3, Attempt 2 (thesis Section II.G.2)**

Each modality independently produces a scalar wear prediction. A learned Linear(2->1) blend layer combines them, initialized with `w_img=1, w_sensor=0`.

- **Result:** Test MAE = 23.70 +/- 19.74 um (tsfresh), 23.50 +/- 20.30 um (physics features).
- **Why it was superseded:** Only marginally better than image-only (23.17 um). The decision-level architecture cannot learn cross-modal feature interactions — it can only re-weight independent predictions. The sensor branch's weight norm (2.39) was much lower than the image branch (5.47), confirming it under-contributed.
- **Superseded by:** Two-tower intermediate fusion (15.55 um).

---

## 12. Standalone ResNet Compression (Without Fusion Retraining)

**Compression experiments (Phase 5)**

Applied structured pruning + distillation to the standalone ResNet18, then evaluated the compressed model without retraining the fusion pipeline around it.

- **Result at 50/85/90/95% sparsity:** Progressive accuracy drop. The compressed ResNet alone was usable (2M QAT: 19.06 um), but plugging a compressed ResNet into a pre-trained fusion model without retraining caused MAE to jump from 15.55 to ~35 um.
- **Resolution:** The fusion model must be retrained end-to-end with the compressed image encoder. This is what the final `train_compressed.py` pipeline does.
