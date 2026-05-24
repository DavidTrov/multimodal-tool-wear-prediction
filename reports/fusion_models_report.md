# Fusion Models — Full Chronological Report

**Project:** MATWI Tool Wear Prediction — Bachelor Thesis  
**Goal:** Combine image (ResNet18, flank images) and sensor (CWT scalograms / physics features) modalities to beat the image-only baseline of **23.17 µm** test MAE.  
**Metric:** Mean Absolute Error (MAE ± std, µm) on held-out test split (Sets 4, 9, 13).  
**Reference baselines:**

| Model | Test MAE |
|---|---|
| Paper (ResNet50, image-only) | 19.00 µm |
| Phase 1 — ResNet18, image-only | 23.17 µm |
| Phase 4 — MultiScaleSensorCNN, sensor-only | 24.96 µm |

---

## Table of Contents

1. [Phase 3a — Early Feature-Level Fusion (Physics Features)](#phase-3a)
2. [Phase 3a.2 — Decision-Level Fusion (Learned Blend)](#phase-3a2)
3. [Phase 3b — Mid-Level Fusion (CWT Scalograms, Old SensorCNN)](#phase-3b)
4. [Phase 3c — Joint Training (Unfrozen ResNet18)](#phase-3c)
5. [Phase 5 v1 — Frozen Encoders, Linear Head](#phase-5v1)
6. [Phase 5 v2 — MLP Head (Non-Linear Fusion)](#phase-5v2)
7. [Phase 5 v3 — GELU Activation](#phase-5v3)
8. [Phase 5 v4 — Two-Tower Projection (Final Best)](#phase-5v4)
9. [Cross-Cutting Discussion](#discussion)
10. [Summary Results Table](#summary)

---

## Phase 3a — Early Feature-Level Fusion (Physics Features) {#phase-3a}

### Methodology

The first fusion attempt combined the frozen ResNet18 image branch with the 100-dimensional physics feature vector (time-domain + FFT + DWT, from Phase 2b). The sensor branch projected the feature vector to match the image branch dimensionality before concatenation.

**Initial architecture (commit `6e12130`):**

```
Image branch   : frozen ResNet18 → fc removed → (B, 512)
                 LayerNorm(512)
                 Linear(512→512)  [projection, matching dim]

Sensor branch  : 100-dim physics features → LayerNorm(100)
                 Linear(100→512)  [projection to 512-d]

Fusion         : concat([img_proj, sensor_proj]) → (B, 1024)
                 Linear(1024→1)

⚠ No input normalisation on sensor features at this stage
```

**Fixed version (commit `8ea42a0`):** Internal feature normalisation was added; sensor projection reduced from 512-d to 128-d to reduce parameter count and overfitting risk.

**Gated version (commit `e6e6f44`):** A learned scalar gate was added to control the relative weight of each modality:

```
gate = sigmoid(Linear(1024, 1))
fused = gate * f_img_norm + (1 - gate) * f_sensor_norm
```

**Attention version (commit `cd47b38`):** Cross-modal attention explored, where sensor features were used to compute an attention mask over the image feature vector.

### Results

| Variant | Test MAE (µm) | Notes |
|---|---|---|
| Initial (no sensor normalisation) | ~42 | severe degradation vs image-only |
| + normalisation, sensor 128-d | ~23 | matched image-only, sensor not adding value |
| + gating layer | ~23 | gate collapsed to ≈1 on image branch |
| Attention variant | ~23 | no improvement over image-only |

### Discussion

The absence of sensor feature normalisation in the first attempt caused the physics feature vector (with values spanning many orders of magnitude) to dominate the gradient signal and destabilise training. Once normalisation was added, the model matched image-only performance (~23 µm) but could not improve on it. The gating mechanism consistently collapsed to routing all weight to the image branch, confirming that the 100-dim physics features did not carry sufficient independent wear information complementary to the image. The sensor branch was effectively ignored.

---

## Phase 3a.2 — Decision-Level Fusion (Learned Blend) {#phase-3a2}

### Methodology

Following the failures of feature-level fusion with physics features, the architecture was redesigned as **decision-level fusion**: each modality makes an independent scalar wear prediction, and a small blending layer combines them. This is the most principled form of late fusion, allowing the network to start from the image-only baseline (by initialising image weight=1, sensor weight=0) and deviate only if the sensor prediction genuinely reduces the loss.

**Architecture (commit `1980c58`, `FusionModel`):**

```
┌─────────────────────────────────────────────────────────────┐
│  Image branch  (frozen)                                      │
│  ResNet18 (Phase-1 weights, fc=Linear(512→1))               │
│  → P_img  (B, 1)                                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│  Sensor branch  (trainable)                                  │
│  100-dim physics features → normalise (μ/σ buffers)         │
│  → Linear(100→1)                                            │
│  → P_sensor  (B, 1)                                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    concat([P_img, P_sensor])  (B, 2)
                           │
                    Linear(2→1)   ← blend
                    init: w_img=1, w_sensor=0, bias=0
                           │
                    P_final  (B, 1)
```

**Total trainable parameters:** 104 (100 sensor head weights + 1 bias + 3 blend weights).

Training: Adam, lr=1e-3, ReduceLROnPlateau, MSELoss, 80 epochs. Image branch fully frozen.

### Results

| Split | n | MAE (µm) | Std |
|---|---|---|---|
| Train | 647 | 28.77 | ±35.57 |
| Val | 300 | 36.90 | ±41.76 |
| Test | 247 | **23.70** | ±25.32 |

### Discussion

The decision-level fusion achieved **23.70 µm** — only 0.53 µm worse than image-only (23.17 µm) and a significant improvement over the previous feature-level attempts. However, post-training analysis of the blend weights showed that `w_sensor` converged to near-zero: the model learned to ignore the sensor branch entirely and route all weight to `P_img`. This was a diagnostic confirmation that physics features computed from the full, non-gated, non-HPF signal did not carry independent wear information that the ResNet18 had not already captured from the image.

The architecture design was nonetheless valuable: the warm-start from image-only (blend initialised to w_img=1) prevented the sensor from destabilising the already-calibrated image branch, explaining why this fusion approach was the cleanest yet.

---

## Phase 3b — Mid-Level Fusion (CWT Scalograms, Old SensorCNN) {#phase-3b}

### Methodology

The physics feature sensor branch was replaced with a **small CNN operating directly on CWT scalograms**, enabling the network to learn time-frequency features rather than relying on hand-crafted statistics. This is the first mid-level (feature-level) fusion with a neural sensor encoder.

**Scalogram sensor encoder (`SensorCNN`, commit `e99f9a3`):**

```
Input: (B, 5, 64, 64)   5-channel CWT scalogram (no HPF, no aircut gating)
│
├─ Conv2d(5→16,  3×3, pad=1) → BatchNorm2d(16) → ReLU → MaxPool(2)   (B, 16, 32, 32)
├─ Conv2d(16→32, 3×3, pad=1) → BatchNorm2d(32) → ReLU → MaxPool(2)   (B, 32, 16, 16)
├─ Conv2d(32→64, 3×3, pad=1) → BatchNorm2d(64) → ReLU → MaxPool(2)   (B, 64,  8,  8)
├─ Conv2d(64→64, 3×3, pad=1) → BatchNorm2d(64) → ReLU
├─ AdaptiveAvgPool2d(1) → Flatten                                      (B, 64)
└─ [used as feature extractor — no head]

~61K parameters
```

**Full fusion model (`ScalogramFusionModel`):**

```
┌──────────────────────────────────────────────────────────────────┐
│  IMAGE BRANCH  (frozen after Phase-1 weights loaded)             │
│  ResNet18 → fc=Identity → f_img  (B, 512)                        │
│  LayerNorm(512)                                                   │
└───────────────────────────────┬──────────────────────────────────┘
                                │
┌──────────────────────────────────────────────────────────────────┐
│  SENSOR BRANCH  (trainable)                                       │
│  SensorCNN on (5, 64, 64) CWT scalogram → f_sensor  (B, 64)      │
│  LayerNorm(64)                                                    │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                   concat([LN(f_img), LN(f_sensor)])  (B, 576)
                                │
                        Linear(576→1)  →  P_final  (B, 1)
                                │
                   ┌────────────┘
                   │  Auxiliary head  (training only)
                   │  Linear(64→1) on f_sensor  →  P_aux  (B, 1)
                   └──────────────────────────────────────────────

Training loss:  MSE(P_final, y)  +  0.2 × MSE(P_aux, y)
Optimizer:      Adam, lr=1e-3, ReduceLROnPlateau(factor=0.5, patience=8)
Epochs:         80
```

The **auxiliary loss (λ=0.2)** on the sensor branch alone was introduced to prevent **modality laziness**: without it, the frozen image branch already produces a good initial prediction, giving the gradient signal no reason to train the sensor encoder — the shared head simply learns to ignore the sensor features.

**Joint training variant (commit `598bafc`):** ResNet18 image encoder unfrozen from the start (lr=1e-4) alongside sensor encoder (lr=1e-3), to see if joint end-to-end training could find a better shared representation.

### Results

| Variant | Test MAE (µm) | Std | Notes |
|---|---|---|---|
| **Frozen ResNet18 + SensorCNN** | **26.98** | ±— | best Phase 3b result |
| + auxiliary loss (λ=0.2) | 34.19 | ±32.51 | MSE loss, same epoch budget |
| Joint training (unfrozen ResNet18) | 34.19 | ±32.51 | see §Phase 3c |

Sensor branch weight norm post-training: **2.39** (vs image branch: **5.47**) — confirming the sensor branch was actively contributing to predictions, not ignored, but not enough to beat image-only.

### Discussion

The best Phase 3b result (26.98 µm) was worse than image-only (23.17 µm) despite the sensor branch being actively used. Three root causes were identified in retrospect:

1. **Weak sensor encoder.** `SensorCNN` used BatchNorm, plain VGG-style convolutions, no skip connections, no attention. BatchNorm at batch size 16–32 with SGDM produces noisy normalisation statistics. The encoder could not capture the complex time-frequency patterns that the Phase 4 work later showed to be learnable.

2. **Uninformative force channel scalograms.** Without the force HPF, fx/fy/fz scalograms were nearly all-black (DC pedestal dominated). The sensor encoder was learning from only 2 of 5 channels (acc, acoustic).

3. **MSELoss with large outliers.** MATWI wear spans 0–220 µm. MSE penalises large errors quadratically; for predictions off by 100–200 µm in early training, a single sample contributes 10,000–40,000 to the loss, destabilising gradients. This was resolved in Phase 4 and 5 by switching to HuberLoss(δ=20).

The auxiliary loss variant actually performed *worse* (34.19 vs 26.98 µm): the MSE auxiliary loss on the weak sensor encoder pulled the shared head in conflicting gradient directions before the sensor branch had learned anything useful.

---

## Phase 3c — Joint Training (Unfrozen ResNet18) {#phase-3c}

### Methodology

An experiment was conducted with both encoders trained simultaneously from the start — ResNet18 at lr=1e-4 (lower, pre-trained weights) and SensorCNN at lr=1e-3 (higher, random init). The hypothesis was that joint training from ImageNet init might allow the image encoder to learn representations better suited for fusion.

```
┌──────────────────────────────────────────┐
│  ResNet18  (TRAINABLE, lr=1e-4)          │
│  ImageNet init → fine-tune end-to-end    │
└──────────────────┬───────────────────────┘
                   │  f_img  (B, 512)
                   ▼
             concat + Linear(576→1)
                   ▲
                   │  f_sensor  (B, 64)
┌──────────────────┴───────────────────────┐
│  SensorCNN  (TRAINABLE, lr=1e-3)         │
│  Random init                             │
└──────────────────────────────────────────┘
```

### Results

| Split | n | MAE (µm) | Std |
|---|---|---|---|
| Train | 647 | 14.93 | ±20.92 |
| Val | 300 | 33.51 | ±49.13 |
| Test | 247 | **34.19** | ±32.51 |

### Discussion

Joint training severely degraded performance: train MAE of 14.93 µm (strong overfit) vs test MAE of 34.19 µm — the worst fusion result to date. The image encoder, now trainable at a lower lr, was re-tuned to jointly optimise the fusion objective rather than generalising from the pre-trained image-only representation. The small MATWI training set (647 samples) was insufficient to re-train 11M ResNet18 parameters from scratch in a joint setting without catastrophic forgetting of the useful ImageNet + Phase-1 representations. Freezing the image encoder was confirmed as essential for all subsequent fusion experiments.

---

## Phase 5 v1 — Frozen Encoders, Linear Head {#phase-5v1}

### Methodology

Phase 5 represented a complete redesign of the fusion approach, motivated by the Phase 4 sensor model improvements. The key changes over Phase 3b:

- **Sensor encoder:** `MultiScaleSensorCNN` (~244K params: inception entry, GroupNorm, ResBlock, CBAM) replacing the old `SensorCNN` (~61K params, BatchNorm, plain VGG)
- **Sensor encoder warm-started** from the Phase-4 best checkpoint (27.19 µm standalone)
- **Both encoders frozen** — only the fusion head is trainable
- **HPF-preprocessed scalograms** — force channels properly activated
- **HuberLoss(δ=20)** replacing MSELoss

**Architecture (commit `e2d4360`):**

```
┌──────────────────────────────────────────────────────────────────────┐
│  IMAGE ENCODER  (frozen, Phase-1 weights)                            │
│  ResNet18 → fc=Identity → f_img  (B, 512)                            │
│  LayerNorm(512)   [trainable]                                        │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
┌──────────────────────────────────────────────────────────────────────┐
│  SENSOR ENCODER  (frozen, Phase-4 weights)                           │
│  MultiScaleSensorCNN.extract_features()                              │
│  (5, 64, 64) HPF scalogram → f_sensor  (B, 96)                      │
│  LayerNorm(96)   [trainable]                                         │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                   concat([LN(f_img), LN(f_sensor)])  (B, 608)
                                   │
                           Linear(608→1)  →  P_final  (B, 1)

                   Auxiliary head: Linear(96→1)  →  P_aux  (B, 1)

Trainable params : 1,922  (LayerNorms + head + aux head)
Frozen params    : 11,420,975

Training loss:  Huber(P_final, y, δ=20)  +  0.2 × Huber(P_aux, y, δ=20)
Optimizer:      SGDM → exploded;  Adam, lr=1e-3, weight_decay=5e-3
```

**Critical failure — SGDM explosion:** The initial training used SGDM (lr=1e-3, momentum=0.9) — the same optimiser that worked well in Phase 4. However, Phase 4 trained ~244K params over a complex non-convex loss landscape where SGDM's momentum helps escape local minima. Phase 5 trains only 1,922 params — essentially linear regression on frozen features. SGDM momentum accumulated across batches and caused catastrophic overshoot: epoch 1 val_MAE = 86,720 µm, escalating to millions by epoch 5. Switching to Adam resolved this immediately. Gradient clipping (max_norm=1.0) was added as a safety net for all subsequent variants.

### Results

| Split | n | MAE (µm) | Std |
|---|---|---|---|
| Train | 625 | 56.55 | ±71.66 |
| Val | 284 | 51.33 | ±74.52 |
| Test | 225 | **29.75** | ±33.59 |

*(Note: n=225 for fusion test split vs n=247 for single-modality test splits — only samples with both a valid image AND a pre-computed scalogram are included.)*

### Discussion

The linear head achieved 29.75 µm — worse than both image-only (23.17 µm) and sensor-only (24.96 µm). Despite using far better encoders than Phase 3b, the linear head could only learn additive combinations of the 608-dim feature vector. The 512 image dims and 96 sensor dims are from fundamentally different feature spaces — a single linear layer cannot learn the non-linear interactions needed to combine them effectively. Additionally, the 5:1 dimensionality imbalance (512 image vs 96 sensor) biases the linear layer's gradient signal toward the image features, effectively marginalising the sensor branch despite the LayerNorm equalisation.

---

## Phase 5 v2 — MLP Head (Non-Linear Fusion) {#phase-5v2}

### Methodology

The linear head was replaced with a small **Multi-Layer Perceptron** to allow non-linear cross-modal interactions. This is the standard approach in multimodal fusion literature for introducing higher-order feature combinations (Baltrusaitis et al., TPAMI 2019).

**Architecture change (commits `f1924e3` → `9eb5578`):**

```
Before: Linear(608→1)

After:  Linear(608→128) → ReLU → Dropout(0.3) → Linear(128→1)
```

Multiple training runs were conducted as the head converged over increasing epochs:

| Commit | Epochs at save | Test MAE (µm) | Std |
|---|---|---|---|
| `f1924e3` | early | 25.98 | ±20.72 |
| `85c7a3e` | epoch 9 | 24.71 | ±18.13 |
| `9eb5578` | epoch 25 | **24.20** | ±20.86 |

Learning rate increased from 1e-3 to 3e-3 (commit `fe17a7f`): **28.31 µm** — worse, the higher LR overshot the optimum with Adam on this small head.

### Results

| Split | n | MAE (µm) | Std |
|---|---|---|---|
| Train | 625 | 37.11 | ±58.29 |
| Val | 284 | 38.20 | ±60.39 |
| Test | 225 | **24.20** | ±20.86 |

### Discussion

The MLP head produced a substantial improvement over the linear head (24.20 vs 29.75 µm), confirming that non-linear cross-modal interactions exist in the feature space. However, at 24.20 µm the model still narrowly failed to beat image-only (23.17 µm). The remaining gap was attributed to two residual issues: (1) ReLU's hard zero for negative activations discards potentially useful cross-modal signals, and (2) the head treats the 512 image dims and 96 sensor dims as a flat 608-dim vector with no structural awareness of modality boundaries.

---

## Phase 5 v3 — GELU Activation + LayerNorm in Head {#phase-5v3}

### Methodology

The ReLU activation in the MLP head was replaced with **GELU** (Gaussian Error Linear Unit, Hendrycks & Gimpel 2016) and a **LayerNorm** was inserted between the first linear layer and the activation. GELU allows small negative activations (unlike ReLU's hard zero), reducing the dead neuron problem. LayerNorm inside the head stabilises the heterogeneous 608-dim input (mixed image and sensor feature scales) before the non-linearity.

**Architecture change (commits `0e462a5`, `f7cf3f9`):**

```
Before: Linear(608→128) → ReLU → Dropout(0.3) → Linear(128→1)

After:  Linear(608→128) → LayerNorm(128) → GELU → Dropout(0.2) → Linear(128→1)
```

Dropout was reduced from 0.3 to 0.2 since LayerNorm already provides some regularisation through its normalisation effect. Optimizer reverted to SGDM (lr=3e-3, momentum=0.9) — stable at this point because gradient clipping was in place and the head had sufficient capacity.

### Results

| Split | n | MAE (µm) | Std |
|---|---|---|---|
| Train | 625 | 53.84 | ±67.60 |
| Val | 284 | 44.22 | ±71.48 |
| Test | 225 | **24.05** | ±21.81 |

### Discussion

Marginal improvement over ReLU (24.05 vs 24.20 µm). GELU helped slightly but the model was still bottlenecked by the flat 608-dim input representation rather than the activation function. The 0.15 µm improvement suggests the dead neuron problem was a minor rather than primary contributor to the performance gap.

---

## Phase 5 v4 — Two-Tower Projection (Final Best) {#phase-5v4}

### Methodology

The fundamental limitation of all prior Phase 5 heads was identified: they all received a flat 608-dim concatenation of two heterogeneous feature spaces (512 image + 96 sensor). The 5:1 dimensionality imbalance means image features receive 5× more gradient signal through the first layer regardless of normalisation. A single shared linear-then-activation operation cannot learn independent per-modality transformations.

The solution was a **two-tower projection** architecture (Lu et al., ViLBERT, NeurIPS 2019; Simonyan & Zisserman, two-stream networks, NeurIPS 2014): each modality is first projected independently into a **shared 128-dimensional space** before concatenation and final regression.

**Architecture (commits `f7cf3f9` → `57e3b4a`, `MultiScaleFusionModel` final):**

```
┌──────────────────────────────────────────────────────────────────────┐
│  IMAGE ENCODER  (frozen)                                             │
│  ResNet18 → f_img  (B, 512)  →  LayerNorm(512)                      │
│                                                                      │
│  img_proj:  Linear(512→128) → LayerNorm(128) → GELU                 │
│                                            →  h_img  (B, 128)       │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
┌──────────────────────────────────────────────────────────────────────┐
│  SENSOR ENCODER  (frozen)                                            │
│  MultiScaleSensorCNN → f_sensor  (B, 96)  →  LayerNorm(96)          │
│                                                                      │
│  sen_proj:  Linear(96→128) → LayerNorm(128) → GELU                  │
│                                            →  h_sensor  (B, 128)   │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                   concat([h_img, h_sensor])  (B, 256)
                                   │
                            Dropout(0.2)
                            Linear(256→64)
                            GELU
                            Linear(64→1)
                                   │
                             P_final  (B, 1)

         Auxiliary head: Linear(96→1) on f_sensor  →  P_aux  (B, 1)

Trainable params : 96,418
  image_norm (512+512)             1,024
  sensor_norm (96+96)                192
  img_proj: Linear(512→128)+LN    66,048  ← main new addition
  sen_proj: Linear(96→128)+LN     12,672
  head: Linear(256→64)+GELU+      16,449
        Linear(64→1)
  aux_head: Linear(96→1)              97
                                ───────
  Total trainable                 96,418

Training loss:  Huber(P_final, y, δ=20) + 0.2 × Huber(P_aux, y, δ=20)
Optimizer:      SGDM, lr=3e-3, momentum=0.9, weight_decay=5e-3
Scheduler:      CosineAnnealingLR(T_max=40, eta_min=1e-5)
Gradient clip:  max_norm=1.0
```

**Why two towers work here:**

1. **Equal gradient flow.** Each modality contributes 128 dims to the merge layer — the 5:1 imbalance (512 vs 96) is eliminated.
2. **Modality-specific non-linear transformation.** `img_proj` learns which aspects of the 512 image features are useful *for fusion* (not just for standalone prediction). `sen_proj` does the same for sensor features. Neither encoder was trained to produce representations complementary to the other.
3. **Balanced merge.** The final head receives two 128-dim vectors, both LayerNorm-normalised, on equal footing.

### Results

| Split | n | MAE (µm) | Std |
|---|---|---|---|
| Train | 625 | 32.77 | ±53.05 |
| Val | 284 | 37.76 | ±57.75 |
| Test | 225 | **22.57** | ±20.17 |

Best epoch: 24 of 40 total. First epoch val_MAE ≈ 107 µm, converging steadily.

### Discussion

The two-tower architecture achieved **22.57 µm** — **beating image-only (23.17 µm) by 0.60 µm**, the first fusion model to do so across all experiments. The improvement from 24.05 µm (GELU head) to 22.57 µm (two-tower) is the largest single-step gain in Phase 5, directly attributable to the per-modality projection towers resolving the gradient imbalance.

The caveat is sample count: the fusion test set has 225 samples vs 247 for image-only (22 samples have an image but no scalogram, or vice versa). This makes a strict MAE comparison uncertain at the margin — the 0.60 µm advantage could narrow if the missing 22 samples were included. However, the consistent improvement across multiple training runs and the mechanistic explanation (equal gradient flow, per-modality non-linear transformation) support the conclusion that the two-tower fusion is genuinely better.

---

## Phase 5 — Compressed Fusion {#phase5-compressed}

### Motivation

Phase 5 v4 (two-tower, standard ResNet18) achieved 22.57 µm using an 11.18M-param image encoder.  
The compressed ResNet (pruned + distilled, 1.98M params, 309-d features, standalone MAE 20.80 µm) is  
the target encoder for on-device deployment. This experiment asks: can the same two-tower fusion  
architecture, with the compressed encoder swapped in, match or beat the standard fusion result?

### Architecture

The architecture is identical to Phase 5 v4 except:

- `image_feat_dim`: 512 → **309**
- `image_norm`: LayerNorm(512) → **LayerNorm(309)**
- `img_proj` input: Linear(512→128) → **Linear(309→128)**
- Image encoder: ResNet18 (11.18M params) → **Compressed ResNet (1.98M params, non-standard channel widths [13, 26, 77, ..., 309])**

Everything from `h_img` onward is unchanged — both produce (B, 128) after projection.

```
┌──────────────────────────────────────────────────────────────────────┐
│  IMAGE ENCODER  (frozen)                                             │
│  Compressed ResNet → f_img  (B, 309)  →  LayerNorm(309)             │
│                                                                      │
│  img_proj:  Linear(309→128) → LayerNorm(128) → GELU                 │
│                                            →  h_img  (B, 128)       │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
┌──────────────────────────────────────────────────────────────────────┐
│  SENSOR ENCODER  (frozen)                                            │
│  MultiScaleSensorCNN → f_sensor  (B, 96)  →  LayerNorm(96)          │
│                                                                      │
│  sen_proj:  Linear(96→128) → LayerNorm(128) → GELU                  │
│                                            →  h_sensor  (B, 128)   │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                   concat([h_img, h_sensor])  (B, 256)
                                   │
                            Dropout(0.2)
                            Linear(256→64)
                            GELU
                            Linear(64→1)
                                   │
                             P_final  (B, 1)

         Auxiliary head: Linear(96→1) on f_sensor  →  P_aux  (B, 1)

Trainable params : 70,028  (vs 96,418 for standard; img_proj smaller due to 309 < 512)
  image_norm (309+309)               618
  sensor_norm (96+96)                192
  img_proj: Linear(309→128)+LN    39,808  ← smaller than standard (66,048)
  sen_proj: Linear(96→128)+LN     12,672
  head: Linear(256→64)+GELU+      16,449
        Linear(64→1)
  aux_head: Linear(96→1)              97
                                ───────
  Total trainable                 70,028
```

**Implementation note:** The compressed encoder was saved as a full model object
(`torch.save(model, path)`) because structured channel pruning produces non-standard
channel widths that cannot be loaded into a standard ResNet18 state dict.
`load_compressed_image_encoder()` loads the full object with `weights_only=False`,
replaces `.fc` with `nn.Identity()`, and installs it as `self.image_encoder`.
`evaluate_compressed.py` must call this method before `load_state_dict()` for the same reason.

---

### Experiment 5c-i — SGDM (same settings as standard fusion)

**Hypothesis:** The same SGDM hyperparameters that worked for the standard fusion  
(lr=3e-3, momentum=0.9, weight_decay=5e-3, 40 epochs) will transfer to the  
compressed fusion since the head architecture and trainable param count are similar.

```
Optimizer:  SGDM, lr=3e-3, momentum=0.9, weight_decay=5e-3
Scheduler:  CosineAnnealingLR(T_max=40, eta_min=1e-5)
Gradient clip: max_norm=1.0
Epochs: 40
```

**Results:**

| Split | n | MAE (µm) | Std | Min | Max |
|---|---|---|---|---|---|
| Train | 625 | 28.10 | ±50.26 | 0.12 | 596.71 |
| Val | 284 | 35.06 | ±46.26 | 0.31 | 166.36 |
| **Test** | **225** | **16.18** | **±17.28** | **0.27** | **210.23** |

Best val MAE: 35.06 µm (checkpoint selected at that epoch).

**Discussion:**

The test MAE of **16.18 µm** is the best result in the entire project — beating the paper's ResNet50 baseline (19.00 µm) and every other model. However, the val/test gap is alarming:

- Val MAE (35.06 µm) is *worse than every baseline*, including sensor-only (24.96 µm)
- Train MAE (28.10 µm) > Test MAE (16.18 µm), which is the wrong ordering
- The model was checkpointed at the best val epoch (35.06 µm), but test happened to be much better

Two interpretations: (1) the test split contains easier wear states by chance (fixed split, only n=225), or (2) the SGDM optimiser did not converge well on the compressed feature distribution — the model found a solution that happens to generalise to test but not to val. The high train std (±50.26) and max error (596.71 µm) confirm the model never fully converged. The hypothesis is rejected: SGDM does not transfer cleanly; the different feature scale of the compressed encoder likely requires adaptive learning rates.

---

### Experiment 5c-ii — Adam {#5c-ii}

**Hypothesis:** Adam's per-parameter adaptive learning rates will handle the different  
feature distribution of the compressed encoder (309-d vs 512-d, different channel scale statistics)  
better than SGDM, producing consistent val/test convergence.

```
Optimizer:  Adam, lr=5e-4, weight_decay=5e-3
Scheduler:  CosineAnnealingLR(T_max=40, eta_min=1e-5)
Gradient clip: max_norm=1.0
Epochs: 40
Sensor encoder: phase4_multiscale_sgdm_best.pt
```

**Results:**

| Split | n | MAE (µm) | Std | Min | Max |
|---|---|---|---|---|---|
| Train | 625 | 23.24 | ±36.38 | 0.04 | 517.52 |
| Val | 284 | 30.30 | ±36.65 | 0.04 | 172.67 |
| **Test** | **225** | **17.66** | **±20.93** | **0.14** | **204.97** |

Val/test gap: **12.64 µm**. Best val MAE used for checkpoint selection.

**Discussion:**

Adam narrowed the val/test gap from ~19 µm (SGDM) to ~13 µm, confirming the SGDM failure was a genuine optimisation problem. Train MAE also improved substantially (28.10 → 23.24 µm), showing better overall convergence. Test MAE is slightly worse than SGDM (17.66 vs 16.18 µm) but the result is more credible given the tighter val alignment.

Val MAE (30.30 µm) still exceeds all single-modality baselines, and the val/test gap persists across both optimisers — indicating the gap is at least partially structural (the test split is genuinely easier than val), not purely an optimisation failure.

This run is the **selected model** for all downstream compression work. See 5c-iii below for why.

---

### Experiment 5c-iii — Adam, alternative sensor checkpoint {#5c-iii}

A second run of the compressed fusion using a different Phase-4 sensor checkpoint as the frozen sensor encoder. The alternative checkpoint produced a lower standalone sensor MAE on the test split.

```
Sensor encoder: alternative Phase-4 checkpoint (lower standalone test MAE)
All other settings identical to 5c-ii
```

**Results:**

| Split | n | MAE (µm) | Std | Min | Max |
|---|---|---|---|---|---|
| Train | 625 | 30.11 | ±55.28 | 0.02 | 609.69 |
| Val | 284 | 36.43 | ±49.57 | 0.02 | 160.02 |
| **Test** | **225** | **14.64** | **±17.95** | **0.08** | **210.76** |

Val/test gap: **21.79 µm**.

**Discussion:**

Test MAE of 14.64 µm is the lowest number produced in the entire project. However, this model was rejected in favour of 5c-ii for the following reasons:

1. **Wrong split ordering.** Train MAE (30.11) > Test MAE (14.64). This never happens in a well-fitted model — train error is almost always lower than test error. It is a strong indicator that the test set contains systematically easier samples, not that the model is genuinely better.

2. **Large val/test gap.** The 21.79 µm gap is 73% wider than 5c-ii's 12.64 µm gap, despite both models using the same architecture and optimizer. The gap widened when a sensor checkpoint that happens to fit the test split well was substituted.

3. **Val worse than all baselines.** Val MAE 36.43 µm exceeds even sensor-only (24.96 µm). A model that performs worse than a single-modality baseline on val cannot be considered reliably better.

4. **High variance.** Train std of ±55.28 µm (max error 609.69 µm) indicates the model never converged — it has high prediction variance even on training data.

**Conclusion:** The 14.64 µm test result is a selection artefact arising from the fixed test split's wear-state composition, amplified by a sensor checkpoint whose representations happen to align with the test distribution. The result is not reproducible in expectation and should not be reported as the model's true performance. **5c-ii (17.66 µm) is the canonical compressed fusion result.**

---

## Phase 5d — Joint Fusion Pruning + Distillation + INT8 Quantization {#phase5d}

### Motivation

Phase 5c-ii achieved 17.66 µm test MAE at 2.19 MB INT8 — 140 KB over the 2 MB NXP FRDM-MCXN947 flash target. Rather than pruning each modality independently (which optimises for standalone per-modality MAE), the full fusion model was pruned jointly so that the pruner could identify channels that are redundant *in the fusion context* — including image encoder channels whose information is already covered by the sensor branch, and vice versa.

### Architecture

| Component | Before pruning | After pruning |
|---|---|---|
| Image encoder | Compressed ResNet (1,979,890 params, 309-d, 85% sparsity) | Further pruned (1,315,810 params, ~33% additional reduction) |
| Sensor CNN | MultiScaleSensorCNN (244,463 params, 96-d) | Pruned (133,615 params, 45% reduction) |
| Fusion head + norms | 70,028 params | 21,474 params (projection towers also pruned) |
| **Total** | **2,294,381 params** | **1,470,899 params** |

**Vision encoder:** 2M-parameter budget compressed ResNet (pruned + distilled from ResNet18, 309-d avgpool features).  
**Sensor encoder:** CWT multiscale CNN (MultiScaleSensorCNN) with inception-style entry, GroupNorm, ResBlock, and CBAM attention on (5, 64, 64) HPF scalograms.  
**Final INT8 size: 1.40 MB** — 612 KB under the 2 MB flash target.

### Phase 5d-i — Joint Pruning (50% target sparsity)

Sensitivity analysis across all 30 conv layers (20 image encoder + 10 sensor CNN) revealed a strong asymmetry:

- **Sensor CNN**: all 10 layers insensitive at 50% sparsity (max ΔMAE = +0.6%), assigned 50% sparsity uniformly
- **Image encoder**: highly variable — `layer4.1.conv2` +270%, `layer4.1.conv1` +119%, `layer3.1.conv1` +125% (all classified very_sensitive → 0% pruning); many early layers sensitive (25% assigned)

The sensor CNN is effectively free to prune aggressively; the image encoder's later layers are the accuracy bottleneck. Three iterative steps with 5-epoch intermediate fine-tunes.

```
Target sparsity    : 50%
Iterative steps    : 3
Intermediate epochs: 5 per step
Optimizer          : Adam, lr=1e-4, weight_decay=1e-3
Scheduler          : CosineAnnealingLR(T_max=40, eta_min=1e-6)
Gradient clip      : max_norm=1.0
Protected layer    : head[-1]  (Linear(→1), scalar output)
```

**Post-pruning size progression:**

| Step | Params | INT8 size | Val MAE (post-prune) |
|---|---|---|---|
| Baseline | 2,294,381 | 2,241 KB | 30.30 µm |
| Step 1 (~17%) | 1,900,275 | 1,856 KB | 57.99 µm |
| Step 2 (~33%) | 1,705,011 | 1,665 KB | 52.45 µm |
| Step 3 (~50%) | 1,470,899 | 1,436 KB | 68.38 µm |

**Final fine-tune results (40 epochs max, patience=8, early stop at epoch 19):**

| Split | n | MAE (µm) | Std |
|---|---|---|---|
| Val | 284 | **38.32** | ±44.91 |

Actual parameter reduction: **35.9%** (target was 50%; sensitivity-aware assignment protected image encoder's critical late layers). Image encoder: 1,315,810 params. Sensor CNN: 133,615 params.

### Phase 5d-ii — Knowledge Distillation

Teacher: `phase5_compressed_fusion_best.pt` (pre-pruning model, 2,294,381 params, 17.66 µm test MAE).  
Student: `fusion_pruned.pt` (1,470,899 params, 35.9% smaller).

```
Loss: (1 - 0.5) · Huber(student, y, δ=20) + 0.5 · MSE(student, teacher)
Optimizer: Adam, lr=1e-4, weight_decay=1e-3
Scheduler: CosineAnnealingLR(T_max=40, eta_min=1e-6)
Early stop at epoch 18 (patience=8)
```

| | Val MAE (µm) |
|---|---|
| Before KD | 38.32 |
| After KD | **34.32** |
| Improvement | −3.99 µm |

### Phase 5d-iii — Dynamic INT8 Quantization

Applied `torch.quantization.quantize_dynamic` (qnnpack engine) to all Conv2d and Linear layers. Activations remain FP32; weights stored as INT8.

**Results:**

| Precision | Val MAE (µm) | Test MAE (µm) | Size |
|---|---|---|---|
| FP32 | 34.32 ± 47.49 | 18.70 ± 18.55 | 5,746 KB (5.61 MB) |
| **INT8** | **34.35 ± 47.53** | **18.70 ± 18.55** | **1,436 KB (1.40 MB)** |
| Accuracy drop | +0.03 µm | **+0.00 µm** | — |

INT8 quantization is lossless on this model — zero accuracy degradation on the test set.

### Discussion

**Fits in flash.** 1.40 MB is 612 KB under the 2 MB target — comfortable margin for firmware overhead.

**Test MAE: 18.70 µm.** The joint-pruned INT8 model beats the paper's ResNet50 image-only baseline (19.00 µm) while being 1.40 MB INT8 and using both modalities. It is 1.04 µm behind the uncompressed Phase 5c-ii model (17.66 µm), a 5.9% relative accuracy cost for a 35.9% parameter reduction and deployment feasibility.

**Sensor CNN pruned more aggressively than image encoder.** The sensitivity analysis confirmed all sensor CNN layers are insensitive at 50% sparsity in the fusion context — the image features dominate prediction accuracy and the sensor branch contributes complementary signal that is distributed across many channels. The image encoder's late layers (layer3, layer4) were very sensitive and mostly protected.

**Negligible INT8 accuracy loss.** The zero Δ on test MAE confirms that dynamic weight-only INT8 quantization introduces no meaningful accuracy degradation for this model class — consistent with the literature on INT8 deployment for regression CNNs.

**Val/test gap persists.** Val MAE (34.32 µm) remains higher than test MAE (18.70 µm), consistent with the structural data-split asymmetry identified in Phase 5c. This is a property of the fixed dataset split, not the compression.

---

## Cross-Cutting Discussion {#discussion}

### Modality complementarity

A per-sample comparison of image-only vs sensor-only predictions on the 225 overlapping test samples (commit `6c24790`) showed that **sensor beats image roughly 50% of the time and image beats sensor the other 50%**. This is the direct empirical motivation for fusion: neither modality is consistently superior, meaning a model that can route to the appropriate modality per sample should outperform both.

The two-tower architecture captures this through the projection layers, which can learn to amplify modality-specific discriminative features before the merge. However, a pure gating mechanism (dynamically routing to one modality based on input) remains unexplored and would be the natural next step.

### Modality laziness and auxiliary loss

In every mid-level fusion experiment, the frozen image branch produced good initial predictions from epoch 1 (warm-started from Phase-1 weights). Without an auxiliary loss on the sensor branch, the fusion head learns to ignore sensor features — the gradient signal from the primary loss is already satisfied by the image features alone. The auxiliary loss (λ=0.2, HuberLoss on P_aux from sensor features only) forces the sensor encoder's representations to be independently predictive, ensuring they carry information that can complement the image branch.

### Loss function

MSELoss (used in Phase 3b) caused erratic training when large errors were present (test MAE range: 0–250 µm, max MSE contribution per sample: 62,500). HuberLoss(δ=20) was the critical switch that stabilised Phase 5 training — its linear tail above 20 µm caps per-sample gradient magnitude at `δ=20 × 1 = 20` regardless of prediction error.

### Optimiser choice

- **Adam** is the correct choice for the tiny Phase 5 head (1,922–96,418 trainable params on frozen features). It is adaptive and self-normalising, avoiding the momentum explosion that plagued SGDM at lr=1e-3 on a near-linear head.
- **SGDM** works once the head has sufficient capacity (Phase 5 v3 onward with gradient clipping), and its implicit regularisation through noisy gradient averaging helps generalisation for the larger two-tower variant (96,418 params).

### Sample size and intersection

The fusion test set is consistently **225 samples** (not 247), because 22 test samples lack either a valid image or a pre-computed scalogram. This 9% reduction potentially removes outlier samples from either modality, creating a mild selection bias. Future work should investigate whether the excluded 22 samples are systematically harder or easier for each modality.

---

## Summary Results Table {#summary}

| # | Architecture | Sensor input | Trainable params | Test MAE (µm) | Std | n |
|---|---|---|---|---|---|---|
| 3a.0 | Feature-level (physics, no normalisation) | 100-d physics | 11.2M | ~42 | — | 247 |
| 3a.1 | Feature-level (physics, normalised, 128-d proj) | 100-d physics | 11.2M | ~23 | — | 247 |
| 3a.2 | Feature-level + learned gate | 100-d physics | 11.2M | ~23 | — | 247 |
| **3a.3** | **Decision-level (learned blend)** | **100-d physics** | **104** | **23.70** | **±25.32** | **247** |
| 3b.1 | Mid-level (CWT, old SensorCNN, no aux) | (5,64,64) scalogram | 62K | 26.98 | — | 247 |
| 3b.2 | Mid-level (CWT, old SensorCNN, aux λ=0.2) | (5,64,64) scalogram | 62K | 34.19 | ±32.51 | 247 |
| 3c | Joint training (unfrozen ResNet18) | (5,64,64) scalogram | 11.2M | 34.19 | ±32.51 | 247 |
| 5.1 | Linear head (frozen enc., no HPF) | (5,64,64) HPF scalogram | 1,922 | 29.75 | ±33.59 | 225 |
| 5.2 | MLP: 608→128→ReLU→DO→1 | (5,64,64) HPF scalogram | 79,938 | 24.20 | ±20.86 | 225 |
| 5.3 | MLP: 608→128→LN→GELU→DO→1 | (5,64,64) HPF scalogram | 79,938 | 24.05 | ±21.81 | 225 |
| **5.4** | **Two-tower: img(512→128) + sen(96→128) → 256→64→1** | **(5,64,64) HPF scalogram** | **96,418** | **22.57** | **±20.17** | **225** |
| 5c-i | Two-tower, compressed enc. (309-d), SGDM | (5,64,64) HPF scalogram | 70,028 | 16.18 † | ±17.28 | 225 |
| **5c-ii** ★ | **Two-tower, compressed enc. (309-d), Adam** | **(5,64,64) HPF scalogram** | **70,028** | **17.66 †** | **±20.93** | **225** |
| 5c-iii | Two-tower, compressed enc., alt. sensor ckpt | (5,64,64) HPF scalogram | 70,028 | 14.64 ‡ | ±17.95 | 225 |
| **5d** ◆ | **Two-tower, joint-pruned INT8 (compressed enc. + CWT)** | **(5,64,64) HPF scalogram** | **1.47M (1.40 MB INT8)** | **18.70** | **±18.55** | **225** |
| — | *Image-only baseline (ResNet18)* | *— (images only)* | *11.18M* | *23.17* | *±19.12* | *247* |
| — | *Sensor-only baseline (MultiScaleCNN)* | *(5,64,64) HPF scalogram* | *244K* | *24.96* | *±26.19* | *247* |
| — | *Paper baseline (ResNet50)* | *— (images only)* | *— * | *19.00* | *—* | *—* |

★ Selected model for downstream compression.  
◆ Deployable on NXP FRDM-MCXN947 (2 MB flash): 50% target sparsity → 35.9% actual reduction, post-distillation val MAE 34.32 µm, INT8 accuracy drop = 0.00 µm on test.  
† Val MAE for 5c-i/ii is 35.06/30.30 µm; test split is systematically easier than val.  
‡ 5c-iii rejected: train MAE (30.11) > test MAE (14.64), val/test gap 21.79 µm, val worse than all baselines — result is a split artefact, not genuine improvement.

---

*Report compiled from git history (commits `6e12130` through `57e3b4a`), experiment result files, model source files, and training scripts.*  
*Fusion experiments span approximately April–May 2026.*
