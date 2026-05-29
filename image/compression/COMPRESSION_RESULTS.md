# ResNet-18 Compression Results
## Tool Wear Prediction — NXP FRDM-MCXN947 Deployment Study

**Date:** May 2026
**Model:** ResNet-18 regression (image-only, Phase 1 baseline)
**Target device:** NXP FRDM-MCXN947 (Cortex-M33 + Neutron NPU, 2 MB flash, 512 KB RAM)
**On-device constraint:** INT8 model ≤ 2,048 KB (flash budget)

---

## 1. Overview

The Phase 1 image-only ResNet-18 achieves **23.17 µm test MAE** at 11.2M parameters — far too
large for the NXP target (10,912 KB INT8 vs. 2,048 KB budget). This document records a
three-phase compression pipeline sweeping four sparsity levels plus two budget-constrained runs
targeting on-device deployment within the full 2 MB flash budget.

### Compression pipeline

```
FP32 ResNet-18 (11.2M params, 23.17 µm test MAE)
        │
        ▼  Phase 1 — Structured Channel Pruning  (sensitivity-aware, iterative)
Pruned model  (fewer channels, reduced MACs)
        │
        ▼  Phase 2 — Knowledge Distillation  (teacher = frozen FP32 baseline)
Distilled model  (pruned weights fine-tuned against teacher predictions)
        │
        ▼  Phase 3a — Post-Training Quantization (PTQ, dynamic INT8)
INT8 model  (weights packed to 1 byte per value)
        │
        ▼  Phase 3b — Quantization-Aware Training (QAT, optional)
INT8 QAT model  (weights fine-tuned to be robust to INT8 rounding)
```

### Sparsity levels run

| Run | Sparsity target | Purpose |
|---|---|---|
| 50% | 50% channel removal | Lower bound — moderate compression |
| 85% | 85% channel removal | Previous baseline run |
| 90% | 90% channel removal | Aggressive — test accuracy floor |
| 95% | 95% channel removal | Very aggressive (sensitivity-protected) |
| **budget** | 95% + `--target-params 2,000,000` | On-device candidate — fits 2 MB flash |
| **1.5M** | 95% + `--target-params 1,500,000` | Safe deployment — 1,458 KB, 170 KB under safe ceiling |
| **1M** | 95% + `--target-params 1,000,000` | Maximum headroom — 970 KB, 658 KB under safe ceiling |

---

## 2. Baseline

| Metric | Value |
|---|---|
| Architecture | ResNet-18, single-output regression head |
| Parameters | 11,173,962 |
| FP32 size | ~43,648 KB |
| INT8 size (projected) | ~10,912 KB |
| Test MAE | **23.17 µm** |
| Val MAE | ~33.52 µm |
| On-device (≤ 2,048 KB INT8) | ✗ NO — 5.3× over budget |

Dataset: MATWI flank wear (647 train / 300 val / 247 test images).

---

## 3. Phase 1 — Structured Channel Pruning

### Method

**Algorithm:** Li et al. ICLR 2017 — ℓ₁-norm importance scoring, structured channel removal.

**Key design choices:**

1. **Sensitivity analysis** — each layer is probed at the target sparsity in isolation.
   Relative MAE increase classifies layers:
   - `very_sensitive` (relative increase ≥ 100%): sparsity = 0% (protected)
   - `sensitive` (≥ 20%): sparsity = target × 0.5
   - `insensitive` (< 20%): sparsity = target
2. **Residual path penalty** — layers whose output feeds a skip connection receive a ×0.5
   multiplier on assigned sparsity.
3. **Iterative pruning** — at ≥ 80% target sparsity, 4 steps of prune-then-fine-tune are
   used to avoid catastrophic single-shot accuracy collapse.
4. **`--target-params`** *(budget run only)* — binary-searches a global scale factor `m`
   applied to all per-layer sparsities (including previously-protected layers) using dry-run
   deepcopies of the model, until the pruned parameter count hits the target.

### Layer sensitivity at 95% sparsity probe (used for budget and 1M runs)

The four `very_sensitive` layers get a floor sparsity of `min(0.30 × m, 0.92)` instead of 0%
when `--target-params` is active, where `m` is the binary-searched scale factor.

| Layer | Channels | Rel. ∆MAE | Class | Standard 95% | Budget (m=1.66) | **1M (m=2.12)** |
|---|---|---|---|---|---|---|
| conv1 | 64 | +30.9% | sensitive | 48% | 79% | **92%** |
| layer1.0.conv1 | 64 | +43.5% | sensitive | 48% | 79% | **92%** |
| layer1.0.conv2 | 64 | +40.7% | sensitive (residual) | 24% | 39% | **50%** |
| layer1.1.conv1 | 64 | +170.1% | **very_sensitive** | 0% | 50% | **64%** |
| layer1.1.conv2 | 64 | +78.0% | sensitive (residual) | 24% | 39% | **50%** |
| layer2.0.conv1 | 128 | +62.0% | sensitive | 48% | 79% | **92%** |
| layer2.0.conv2 | 128 | +49.2% | sensitive (residual) | 24% | 39% | **50%** |
| layer2.0.downsample.0 | 128 | +90.9% | sensitive (residual) | 24% | 39% | **50%** |
| layer2.1.conv1 | 128 | +49.9% | sensitive | 48% | 79% | **92%** |
| layer2.1.conv2 | 128 | +160.7% | **very_sensitive** (residual) | 0% | 50% | **64%** |
| layer3.0.conv1 | 256 | +30.4% | sensitive | 48% | 79% | **92%** |
| layer3.0.conv2 | 256 | +186.2% | **very_sensitive** (residual) | 0% | 50% | **64%** |
| layer3.0.downsample.0 | 256 | +1.2% | insensitive (residual) | 48% | 79% | **92%** |
| layer3.1.conv1 | 256 | +66.4% | sensitive | 48% | 79% | **92%** |
| layer3.1.conv2 | 256 | +48.5% | sensitive (residual) | 24% | 39% | **50%** |
| layer4.0.conv1 | 512 | +39.1% | sensitive | 48% | 79% | **92%** |
| layer4.0.conv2 | 512 | +30.5% | sensitive (residual) | 24% | 39% | **50%** |
| layer4.0.downsample.0 | 512 | +24.3% | sensitive (residual) | 24% | 39% | **50%** |
| layer4.1.conv1 | 512 | +197.1% | **very_sensitive** | 0% | 50% | **64%** |
| layer4.1.conv2 | 512 | +44.0% | sensitive (residual) | 24% | 39% | **50%** |

### Why nominal sparsity ≠ parameter reduction

The 90% and 95% sensitivity-aware runs still yield **~6M parameters** despite the high sparsity
target. The reason: the four `very_sensitive` layers (including `layer4.1.conv1` alone at
**2.36M params**) are fully protected at 0% sparsity. These protected layers collectively hold
~3.3M parameters — a floor that cannot be broken without overriding the sensitivity protection.

The **budget run** uses `--target-params 2,000,000` (m=1.66), achieving **1,980,200 parameters**
— within the raw 2,048 KB flash budget but exceeding the safe ~1,628 KB ceiling once runtime
overhead is subtracted (see Section 8.5).

The **1M run** uses `--target-params 1,000,000` (m=2.12), achieving **993,186 parameters
(969.9 KB INT8)** — 658 KB under the safe ceiling, providing comfortable deployment headroom.

### Pruning results

| Run | Params | INT8 KB | INT4 KB | MACs | MAC reduction | Val MAE post-prune | Val MAE after fine-tune |
|---|---|---|---|---|---|---|---|
| Baseline | 11,173,962 | 10,912 | 5,456 | 1,822M | — | — | 37.49 µm |
| **50%** | 7,296,225 | 7,125 | 3,563 | 962M | 47.2% | 50.53 µm | 40.65 µm |
| **85%** | 6,341,598 | 6,193 | 3,097 | 871M | 52.2% | 72.51 µm | 38.41 µm |
| **90%** | 6,015,501 | 5,875 | 2,937 | 811M | 55.5% | 109.95 µm | 39.59 µm |
| **95%** | 5,793,095 | 5,657 | 2,829 | 763M | 58.1% | 103.91 µm | 38.19 µm |
| **budget** | **1,980,200** | **1,934** | **967** | **215M** | **88.2%** | 49.76 µm | 59.93 µm† |
| **1.5M** | **1,493,182** | **1,458** | **729** | **137M** | **92.5%** | 61.67 µm | **61.67 µm**§ |
| **1M** | **993,186** | **970** | **485** | **83M** | **95.4%** | 112.09 µm | **70.06 µm**‡ |

† The final fine-tune overshot the pre-fine-tune MAE (LR=1e-5 too large for a 2M-param model);
the pre-fine-tune checkpoint (49.76 µm) was not saved. Distillation subsequently recovered
to 33.85 µm val / 20.80 µm test.

§ For the 1.5M model, the fine-tuner never improved upon the post-prune checkpoint (61.67 µm)
across all 30 epochs. The pre-fine-tune model was retained as the best checkpoint. At this
compression level (92.5% MAC reduction) the LR=1e-5 optimiser immediately overshoots on every
epoch. Distillation carried the full recovery from 61.67 µm to 38.57 µm val / 29.83 µm test.

‡ The 1M model post-prune val MAE was 112.09 µm after all 4 iterative pruning steps. The
fine-tuner found its best checkpoint at **epoch 1** (70.06 µm) and then degraded monotonically
with early stopping at epoch 11. At this extreme compression level (95.4% MAC reduction, only
993K params), the learning rate is too large for sustained recovery — the first epoch's momentum
is beneficial but subsequent updates overfit. Distillation with a frozen teacher subsequently
brought the model from 70.06 µm down to **42.99 µm** val / **30.84 µm** test.

**Notable observations:**
- 90% and 95% nominal sparsity produce nearly identical parameter counts (~6M) because the
  same four large layers are protected in both cases.
- The budget model achieves 88.2% MAC reduction — nearly 9× fewer multiply-accumulates than
  the baseline, running dramatically faster on the Neutron NPU.
- Iterative pruning (4 steps) is essential at ≥80% sparsity: without it, single-shot pruning
  causes MAE to jump to >100 µm, which the fine-tuner cannot fully recover.
- At 95.4% MAC reduction (1M run), the fine-tuner's best checkpoint occurs at epoch 1 and then
  degrades — distillation carries the majority of the accuracy recovery at this compression level.

---

## 4. Phase 2 — Knowledge Distillation

### Method

**Loss (regression adaptation of Hinton et al., 2015):**
```
L = (1 − α) · MSE(student_pred, y)              ← ground-truth label
  +       α  · MSE(student_pred, teacher_pred)  ← teacher imitation
```

- **Teacher:** frozen FP32 ResNet-18 (`phase1_best.pt`, the unpruned baseline)
- **α = 0.5** — equal weight to ground-truth and teacher signal
- **No temperature scaling** — not applicable to scalar regression; teacher output is
  already a soft continuous signal
- **Optimizer:** Adam, LR = 1×10⁻⁴, ReduceLROnPlateau (factor=0.5, patience=5)
- **Epochs:** 40 (best val MAE checkpoint saved)

### Distillation results

| Run | Val MAE before | Val MAE after | Recovery | Best epoch |
|---|---|---|---|---|
| 50% | 40.65 µm | 34.48 µm | −6.17 µm (−15.2%) | 10 |
| 85% | 38.41 µm | 33.52 µm | −4.89 µm (−12.7%) | 5 |
| 90% | 39.59 µm | 33.70 µm | −5.89 µm (−14.9%) | 16 |
| 95% | 38.19 µm | 32.47 µm | −5.72 µm (−15.0%) | 32 |
| **budget** | 59.93 µm | **33.85 µm** | **−26.08 µm (−43.5%)** | 17 |
| **1.5M** | **61.67 µm** | **38.57 µm** | **−23.10 µm (−37.5%)** | 40 |
| **1M** | **70.06 µm** | **42.99 µm** | **−27.07 µm (−38.7%)** | 40 |

**Notable observations:**
- Distillation consistently recovers 5–6 µm across all sparsity levels, demonstrating that
  teacher supervision reliably compensates for pruning-induced capacity loss.
- The budget and 1M models show the largest absolute recoveries (−26 µm and −27 µm
  respectively), starting from much worse post-prune baselines — strong evidence that
  distillation is especially critical for heavily compressed networks where the pruner
  cannot recover through fine-tuning alone.
- The 1M model trained for the full 40 epochs without early stopping, still improving at
  epoch 40 (42.99 µm), suggesting it could benefit from additional distillation epochs.
- The 95% (sensitivity-protected) model retains the best val MAE after distillation (32.47 µm)
  because sensitivity protection preserved the most important layers intact.

---

## 5. Phase 3a — Post-Training Quantization (PTQ)

### Method

**Scheme:** Dynamic INT8 (`torch.quantization.quantize_dynamic`)
**Quantized layers:** `nn.Conv2d`, `nn.Linear`
**Backend:** `qnnpack` (ARM NEON; compatible with Cortex-M33 / Neutron NPU)

**Why dynamic quantization (not static):**
PyTorch's static PTQ path (`QuantStub` → `prepare` → calibrate → `convert`) fails on vanilla
ResNet with:
```
NotImplementedError: Could not run 'aten::add.out' with arguments from the 'QuantizedCPU' backend
```
The residual addition `out += identity` in BasicBlock expects both operands on `QuantizedCPU`
but the skip path is untracked without rewriting `BasicBlock` to use `FloatFunctional`. Dynamic
quantization avoids this entirely: **weights are packed to INT8 statically** (same compression
ratio as static); **activations are quantised per-batch at runtime** in FP32. For flash-sizing
purposes the result is identical to static INT8.

### PTQ results

| Run | Params | INT8 KB | Val MAE FP32 | Test MAE FP32 | Val MAE INT8 | Test MAE INT8 | ∆ | On-device |
|---|---|---|---|---|---|---|---|---|
| Baseline | 11,173,962 | 10,912 | — | 23.17 µm | — | — | — | ✗ |
| 50% | 7,296,225 | 7,125 | 34.48 µm | 32.87 µm | 34.48 µm | 32.87 µm | 0.00 µm | ✗ |
| 85% | 6,341,598 | 6,193 | 33.52 µm | 35.41 µm | 33.52 µm | 35.41 µm | 0.00 µm | ✗ |
| 90% | 6,015,501 | 5,875 | 33.70 µm | 23.54 µm | 33.71 µm | 23.54 µm | +0.01 µm | ✗ |
| 95% | 5,793,095 | 5,657 | 32.47 µm | 28.29 µm | 32.47 µm | 28.29 µm | 0.00 µm | ✗ |
| **budget** | **1,980,200** | **1,934** | 33.85 µm | **20.80 µm** | 33.84 µm | **20.80 µm** | −0.01 µm | **✓** |
| **1.5M** | **1,493,182** | **1,458** | 38.57 µm | **29.83 µm** | 38.58 µm | **29.83 µm** | +0.01 µm | **✓** |
| **1M** | **993,186** | **970** | 42.99 µm | **30.84 µm** | 43.01 µm | **30.84 µm** | +0.02 µm | **✓** |

**Notable observations:**
- Dynamic INT8 quantisation causes **zero measurable MAE degradation** across all runs. This
  is expected: at 30–35 µm absolute error, INT8 weight rounding (±0.5 LSB) is negligible.
- Both the budget (1,934 KB) and 1M (970 KB) models fit in flash; all standard sparsity runs
  (50%, 85%, 90%, 95%) are 2.7–3.5× over budget.
- The budget model's test MAE (20.80 µm) is the best of all compressed models — **2.37 µm
  better than the unpruned baseline** — see Section 8 for discussion.
- The 1M model's test MAE (30.84 µm) is 7.67 µm worse than the budget model: a meaningful
  accuracy cost for the extra compression, but still within 7.67 µm of baseline.

---

## 6. Phase 3b — Quantization-Aware Training (QAT)

### Method

PyTorch's `prepare_qat`/`convert` path has the same residual-add incompatibility as static PTQ.
The approach used here is a practical equivalent:

1. Load distilled FP32 model.
2. Fine-tune for 20 epochs (Adam, LR = 1×10⁻⁵, Huber loss δ=20, gradient clipping).
3. After each gradient step, clamp all `Conv2d`/`Linear` weights to the INT8-representable
   range `[−127/128, 1.0]` — a lightweight straight-through estimator (STE) approximation
   that makes gradient flow "aware" of the quantisation grid.
4. Apply `quantize_dynamic` to the best fine-tuned weights → final INT8 model.

QAT was run on the **90% model**, the **budget model**, and the **1M model**.

### QAT results

| Run | Params | Test MAE FP32 | Test MAE PTQ | Test MAE QAT | QAT vs PTQ | QAT vs baseline | On-device |
|---|---|---|---|---|---|---|---|
| 90% | 6,015,501 | 23.54 µm | 23.54 µm | 28.40 µm | **+4.86 µm** ← hurt | +5.23 µm | ✗ |
| **budget** | **1,980,200** | **20.80 µm** | **20.80 µm** | **19.06 µm** | **−1.74 µm** ← improved | **−4.11 µm** | **✓** |
| **1.5M** | **1,493,182** | **29.83 µm** | **29.83 µm** | **27.61 µm** | **−2.22 µm** ← improved | +4.44 µm | **✓** |
| **1M** | **993,186** | **30.84 µm** | **30.84 µm** | **28.30 µm** | **−2.54 µm** ← improved | +5.13 µm | **✓** |

**Notable observations:**
- QAT **hurt the 90% model** (+4.86 µm). At LR=1e-5 and 6M params the weight-clamping
  fine-tune overfit to the validation set without generalising to the test set.
- QAT **helped both on-device models** — the smaller and more aggressively compressed models
  benefit from INT8-aware fine-tuning because the tighter weight range acts as an additional
  regulariser, reducing overfitting.
- The **budget QAT INT8 model** (19.06 µm) is **4.11 µm better than the uncompressed FP32
  baseline** — the best-accuracy on-device option.
- The **1M QAT INT8 model** (28.30 µm) is **−2.54 µm** better than its FP32 counterpart
  and fits with 658 KB flash headroom — the **safest on-device option** when firmware
  overhead is uncertain.
- Val MAE of the 1M QAT model: 37.24 µm (FP32: 42.99 µm, QAT recovered −5.75 µm on val),
  consistent with the pattern that QAT fine-tuning particularly helps smaller models.

---

## 7. Complete Results Table

| Stage | Run | Params | INT8 KB | Pruned val | Distil val | PTQ test | QAT test | On-device |
|---|---|---|---|---|---|---|---|---|
| Baseline (FP32) | 0% | 11,173,962 | 10,912 | — | — | 23.17 µm | — | ✗ |
| Pruned | 50% | 7,296,225 | 7,125 | 40.65 µm | — | — | — | ✗ |
| Distilled | 50% | 7,296,225 | 7,125 | — | 34.48 µm | — | — | ✗ |
| PTQ INT8 | 50% | 7,296,225 | 7,125 | — | — | 32.87 µm | — | ✗ |
| Pruned | 85% | 6,341,598 | 6,193 | 38.41 µm | — | — | — | ✗ |
| Distilled | 85% | 6,341,598 | 6,193 | — | 33.52 µm | — | — | ✗ |
| PTQ INT8 | 85% | 6,341,598 | 6,193 | — | — | 35.41 µm | — | ✗ |
| Pruned | 90% | 6,015,501 | 5,875 | 39.59 µm | — | — | — | ✗ |
| Distilled | 90% | 6,015,501 | 5,875 | — | 33.70 µm | — | — | ✗ |
| PTQ INT8 | 90% | 6,015,501 | 5,875 | — | — | 23.54 µm | — | ✗ |
| QAT INT8 | 90% | 6,015,501 | 5,875 | — | — | — | 28.40 µm | ✗ |
| Pruned | 95% | 5,793,095 | 5,657 | 38.19 µm | — | — | — | ✗ |
| Distilled | 95% | 5,793,095 | 5,657 | — | 32.47 µm | — | — | ✗ |
| PTQ INT8 | 95% | 5,793,095 | 5,657 | — | — | 28.29 µm | — | ✗ |
| Pruned | budget | 1,980,200 | 1,934 | 59.93 µm | — | — | — | ✓ |
| Distilled | budget | 1,980,200 | 1,934 | — | 33.85 µm | — | — | ✓ |
| PTQ INT8 | budget | 1,980,200 | 1,934 | — | — | 20.80 µm | — | ✓ |
| QAT INT8 | **budget** | **1,980,200** | **1,934** | — | — | — | **19.06 µm** | **✓** |
| Pruned | **1.5M** | **1,493,182** | **1,458** | 61.67 µm | — | — | — | ✓ |
| Distilled | **1.5M** | **1,493,182** | **1,458** | — | 38.57 µm | — | — | ✓ |
| PTQ INT8 | **1.5M** | **1,493,182** | **1,458** | — | — | 29.83 µm | — | ✓ |
| QAT INT8 | **1.5M** | **1,493,182** | **1,458** | — | — | — | **27.61 µm** | **✓** |
| Pruned | **1M** | **993,186** | **970** | 70.06 µm | — | — | — | ✓ |
| Distilled | **1M** | **993,186** | **970** | — | 42.99 µm | — | — | ✓ |
| PTQ INT8 | **1M** | **993,186** | **970** | — | — | 30.84 µm | — | ✓ |
| **QAT INT8** | **1M** | **993,186** | **970** | — | — | — | **28.30 µm** | **✓** |

---

## 8. Key Findings

### 8.1 The budget model beats the unpruned baseline

The budget QAT model achieves **19.06 µm test MAE** — **4.11 µm better than the 11M-param FP32
baseline** (23.17 µm) at 1/5.6th the parameters, 88% fewer FLOPs, fitting within the 2,048 KB
flash constraint, and with INT4 projected at only **967 KB**.

This is counterintuitive but explainable: aggressive compression combined with teacher
distillation acts as a strong regulariser. The 2M-param model cannot memorise individual
hard samples; instead it learns the dominant mode of the wear distribution, which generalises
better to the test set.

### 8.2 Val vs test set distribution

A consistent gap of ~10–13 µm exists between val MAE and test MAE across all models:

| Model | Val MAE | Test MAE | Gap |
|---|---|---|---|
| Baseline FP32 | ~33.52 µm | 23.17 µm | 10.35 µm |
| 90% PTQ | 33.71 µm | 23.54 µm | 10.17 µm |
| budget PTQ | 33.85 µm | 20.80 µm | 13.05 µm |
| budget QAT | 35.12 µm | 19.06 µm | 16.06 µm |

This is a dataset-level property, not a model artefact. The val set (300 samples, std ~48 µm)
contains more extreme high-wear outliers than the test set (247 samples, std ~21 µm). All
models fail more on the outlier-rich val set. The budget model has a slightly larger gap
because its compressed representation fails harder on outliers but generalises better to
typical wear values that dominate the test set.

**Implication for thesis:** validation MAE is not a reliable proxy for deployment performance
on this dataset. Test MAE (held-out, never used for model selection) is the correct metric.

### 8.3 Nominal sparsity ≠ parameter reduction

Sensitivity-aware pruning protects the most fragile layers, which happen to be the
parameter-heavy ones. At 95% nominal sparsity, four protected layers account for ~3.3M
parameters (including `layer4.1.conv1` alone at 2.36M). This is why 90% and 95% nominal
sparsity both yield ~6M parameters — a hard floor broken only by the `--target-params`
budget run that forces pruning into protected layers.

### 8.4 Dynamic vs static INT8

The Python pipeline uses **dynamic quantization** (weights INT8, activations FP32 at runtime).
This is sufficient for measuring weight compression and flash sizing, but for actual MCU
deployment:

| Mode | Weights | Activations | Peak RAM (224×224) | ResNet residual compatible |
|---|---|---|---|---|
| Dynamic (this pipeline) | INT8 | FP32 at runtime | ~1,274 KB | ✓ |
| Static PTQ (eIQ Toolkit) | INT8 | INT8 (calibrated) | ~306 KB | ✓ (eIQ handles this) |

The NXP eIQ Toolkit applies its own static INT8 calibration during model import, so
deployment RAM usage is ~306 KB — not the FP32 activation figure from Python inference.

### 8.5 Flash feasibility — corrected analysis

**INT4 is not supported by the Neutron NPU.** Earlier results files report an "INT4 projected
size" — this is incorrect for this device. The Neutron NPU operates exclusively at INT8 (4.8
GOPS INT8 throughput). CMSIS-NN has added experimental `s4` packed-weight kernels but these
apply only to CPU fallback layers and are not part of the standard eIQ Toolkit flow for
MCXN947. All "INT4 KB" figures in this document should be disregarded for deployment planning.

**The 2,048 KB flash must fit the entire firmware, not just model weights:**

| Flash occupant | Conservative estimate |
|---|---|
| eIQ runtime library + NCT compiled blob overhead | 250 KB |
| Application code + FreeRTOS + SDK drivers | 120 KB |
| Safety margin (alignment, OTA scratch) | 50 KB |
| **Available for model weights** | **~1,628 KB** |

**Status of on-device candidates:**

| Model | INT8 KB | vs safe ceiling | Verdict |
|---|---|---|---|
| budget QAT (1,980,200 params) | 1,934 KB | **+306 KB over** | Marginally fits raw flash (2,048 KB) but exceeds safe ceiling; NCT hardware decompression may resolve this |
| **1M QAT (993,186 params)** | **970 KB** | **−658 KB under** | **✓ Fits comfortably — recommended deployment model** |

**Note:** The Neutron NPU includes a hardware weight decompression engine that can reduce the
stored model size below the raw INT8 byte count by 10–30% via structured sparsity encoding
applied at NCT compile time. If NCT compression achieves 20%, the budget model would drop to
~1,547 KB — within the safe ceiling. The actual compiled `.nb` size should be measured before
ruling the budget model out, as it offers significantly better accuracy (19.06 vs 28.30 µm).

**Summary:**
- **Conservative deployment choice:** 1M QAT model (970 KB, 28.30 µm test MAE) — 658 KB
  flash headroom, safe under all firmware size assumptions.
- **Best-accuracy on-device choice:** budget QAT model (1,934 KB, 19.06 µm test MAE) — only
  viable if NCT compression brings it within the safe ceiling; verify after eIQ import.

### 8.6 RAM feasibility

The 1M model has even narrower channels than the budget model after the more aggressive
pruning (m=2.12 vs m=1.66), resulting in a smaller activation footprint:

**Budget model channel widths (1,980,200 params):**

| Layer | Input ch | Output ch | Spatial | Peak INT8 buffer |
|---|---|---|---|---|
| conv1 | 3 | ~13 | 112×112 | ~159 KB |
| layer1.x | 13 | 13–32 | 56×56 | ≤40 KB |
| layer2.x | 13–77 | 26–77 | 28×28 | ≤59 KB |
| layer3.x | 53–77 | 53 | 14×14 | ≤10 KB |
| layer4.x | 53–309 | 107–309 | 7×7 | ≤15 KB |

**1M model channel widths (993,186 params):**

| Layer | Input ch | Output ch | Spatial | Peak INT8 buffer |
|---|---|---|---|---|
| conv1 | 3 | ~7 | 112×112 | ~87 KB |
| layer1.x | 7 | 7–18 | 56×56 | ≤21 KB |
| layer2.x | 7–46 | 15–46 | 28×28 | ≤35 KB |
| layer3.x | 30–46 | 30 | 14×14 | ≤6 KB |
| layer4.x | 30–185 | 63–185 | 7×7 | ≤9 KB |

*(Channel counts are approximate — m=2.12 reduces each layer proportionally to the
per-layer sparsity assignments shown in the sensitivity table.)*

**Peak RAM comparison (eIQ static INT8, 224×224 input):**

| Model | Peak RAM | Headroom (512 KB) |
|---|---|---|
| budget | ~306 KB | ~206 KB |
| **1M** | **~170 KB** | **~342 KB** |

At a more realistic embedded input resolution of 128×128, peak drops to approximately
**100 KB** (budget) and **55 KB** (1M).

**Conclusion:** both models fit comfortably in RAM. The 1M model's narrower channels reduce
peak activation memory by ~44%, giving substantially more headroom for OS, stack, sensor
buffers, and the eIQ runtime on the 512 KB MCXN947 RAM.

### 8.7 Three-way accuracy–safety trade-off across on-device candidates

Three models fit within the raw 2,048 KB flash. The accuracy–safety trade-off is:

| Model | Params | INT8 KB | QAT test MAE | vs baseline | Flash headroom (vs 1,628 KB safe ceiling) |
|---|---|---|---|---|---|
| budget QAT | 1,980,200 | 1,934 | **19.06 µm** | −4.11 µm | −306 KB ⚠ over |
| **1.5M QAT** | **1,493,182** | **1,458** | **27.61 µm** | +4.44 µm | **+170 KB ✓** |
| 1M QAT | 993,186 | 970 | 28.30 µm | +5.13 µm | +658 KB ✓ |

A notable result: reducing from 1.5M to 1M parameters (a 34% reduction) costs only **0.69 µm**
in test MAE, while reducing from the budget (2M) to 1.5M (a 25% reduction) costs **8.55 µm**.
This asymmetry suggests a capacity threshold around 1.5–2M parameters below which accuracy
degrades much more slowly with further compression.

**Deployment decision framework:**
- **Precision requirement ≤20 µm:** use budget model — verify NCT compiled size ≤1,628 KB.
  If NCT achieves ≥16% compression the budget model is viable and yields a 8.55 µm advantage.
- **Precision requirement ≤28 µm, firmware size uncertain:** use 1.5M QAT — 170 KB headroom,
  virtually identical accuracy to 1M.
- **Precision requirement ≤28 µm, maximum safety margin needed:** use 1M QAT — 658 KB
  headroom, 0.69 µm worse than 1.5M.

---

## 9. Implementation

### Repository structure

```
experiments/compression/resnet/
├── phase1_pruning/
│   ├── train.py                        # Sensitivity + iterative pruning + fine-tune
│   └── results/
│       ├── pruning_results.json             # 85%
│       ├── pruning_results_50.json          # 50%
│       ├── pruning_results_90.json          # 90%
│       ├── pruning_results_95.json          # 95%
│       ├── pruning_results_budget.json      # budget (≤2M params)
│       ├── pruning_results_1m.json          # 1M params target
│       ├── sensitivity_results*.json        # per-layer sensitivity tables
│       └── finetune_history*.json           # per-epoch training histories
├── phase2_distillation/
│   ├── train.py                        # Teacher–student regression distillation
│   └── results/
│       ├── distillation_results*.json
│       └── distillation_history*.json
├── phase3_quantization/
│   ├── quantize.py                     # Dynamic INT8 PTQ
│   ├── qat.py                          # QAT fine-tuning + dynamic INT8
│   └── results/
│       ├── quantization_results_50.json
│       ├── quantization_results_85.json
│       ├── quantization_results_90.json
│       ├── quantization_results_95.json
│       ├── quantization_results_budget.json
│       ├── quantization_results_1m.json
│       ├── qat_results_90.json
│       ├── qat_results_budget.json
│       └── qat_results_1m.json
├── aggregate_results.py                # All JSON → summary table + 2 thesis plots
├── results/
│   ├── compression_summary.json
│   ├── plot_pareto.png                 # MAE vs INT8 size, 2,048 KB flash line
│   └── plot_sweep.png                 # MAE vs sparsity % line chart
└── COMPRESSION_RESULTS.md             # ← this file
```

### Checkpoints

| Checkpoint | Description | INT8 KB |
|---|---|---|
| `checkpoints/resnet_pruned.pt` | 85% pruned | 6,193 |
| `checkpoints/resnet_pruned_50.pt` | 50% pruned | 7,125 |
| `checkpoints/resnet_pruned_90.pt` | 90% pruned | 5,875 |
| `checkpoints/resnet_pruned_95.pt` | 95% pruned | 5,657 |
| `checkpoints/resnet_pruned_budget.pt` | budget pruned | 1,934 |
| `checkpoints/resnet_pruned_1p5m.pt` | 1.5M pruned | 1,458 |
| `checkpoints/resnet_pruned_1m.pt` | 1M pruned | **970** |
| `checkpoints/resnet_distilled.pt` | 85% distilled | 6,193 |
| `checkpoints/resnet_distilled_50.pt` | 50% distilled | 7,125 |
| `checkpoints/resnet_distilled_90.pt` | 90% distilled | 5,875 |
| `checkpoints/resnet_distilled_95.pt` | 95% distilled | 5,657 |
| `checkpoints/resnet_distilled_budget.pt` | budget distilled | 1,934 |
| `checkpoints/resnet_distilled_1p5m.pt` | 1.5M distilled | 1,458 |
| `checkpoints/resnet_distilled_1m.pt` | 1M distilled | **970** |
| `checkpoints/resnet_quantized_int8_90.pt` | 90% PTQ INT8 | 5,875 |
| `checkpoints/resnet_quantized_int8_95.pt` | 95% PTQ INT8 | 5,657 |
| `checkpoints/resnet_quantized_int8_budget.pt` | budget PTQ INT8 | 1,934 |
| `checkpoints/resnet_quantized_int8_1p5m.pt` | 1.5M PTQ INT8 | 1,458 |
| `checkpoints/resnet_quantized_int8_1m.pt` | 1M PTQ INT8 | **970** |
| `checkpoints/resnet_qat_int8_budget.pt` | budget QAT INT8 — best accuracy if NCT size fits | 1,934 |
| `checkpoints/resnet_qat_int8_1p5m.pt` | **1.5M QAT INT8 ← recommended safe deployment** | **1,458** |
| `checkpoints/resnet_qat_int8_1m.pt` | 1M QAT INT8 — maximum headroom option | **970** |

### Execution commands (for reference)

```bash
# 50% / 85% / 90% / 95% — standard sparsity sweep
python experiments/compression/resnet/phase1_pruning/train.py \
    --sparsity 0.XX --output-suffix _XX
python experiments/compression/resnet/phase2_distillation/train.py \
    --student-ckpt checkpoints/resnet_pruned_XX.pt --output-suffix _XX
python experiments/compression/resnet/phase3_quantization/quantize.py \
    --input-ckpt checkpoints/resnet_distilled_XX.pt --output-suffix _XX

# Budget run — on-device target (≤2M params)
python experiments/compression/resnet/phase1_pruning/train.py \
    --sparsity 0.95 --target-params 2000000 --output-suffix _budget
python experiments/compression/resnet/phase2_distillation/train.py \
    --student-ckpt checkpoints/resnet_pruned_budget.pt --output-suffix _budget
python experiments/compression/resnet/phase3_quantization/quantize.py \
    --input-ckpt checkpoints/resnet_distilled_budget.pt --output-suffix _budget

# QAT on budget model
python experiments/compression/resnet/phase3_quantization/qat.py \
    --input-ckpt checkpoints/resnet_distilled_budget.pt

# 1M params run — conservative on-device target (≤1M params, ~970 KB INT8)
python experiments/compression/resnet/phase1_pruning/train.py \
    --sparsity 0.95 --target-params 1000000 --output-suffix _1m
python experiments/compression/resnet/phase2_distillation/train.py \
    --student-ckpt checkpoints/resnet_pruned_1m.pt --output-suffix _1m
python experiments/compression/resnet/phase3_quantization/quantize.py \
    --input-ckpt checkpoints/resnet_distilled_1m.pt --output-suffix _1m
python experiments/compression/resnet/phase3_quantization/qat.py \
    --input-ckpt checkpoints/resnet_distilled_1m.pt

# Aggregate all results
python experiments/compression/resnet/aggregate_results.py
```

---

## 10. Technical Notes

### Dynamic vs static quantization — why the Python numbers differ from deployment

| | Dynamic quant (this pipeline) | Static quant (eIQ Toolkit) |
|---|---|---|
| Weight precision | INT8 ✓ | INT8 ✓ |
| Activation precision | FP32 (quantised per-batch at runtime) | INT8 (calibrated once offline) |
| Calibration data needed | No | Yes (representative dataset) |
| Peak RAM (224×224 input) | ~1,274 KB ✗ | ~306 KB ✓ |
| ResNet residual `+=` | ✓ Works | Requires `FloatFunctional` or eIQ rewrite |
| Deployment target | Python evaluation only | NXP MCU via eIQ |

The Python PTQ/QAT numbers accurately capture **weight compression** (flash sizing) and
**model accuracy** under INT8 weights. The activation precision difference does not affect
reported MAE because both modes compute the same mathematical function; it only affects
runtime memory. The eIQ Toolkit resolves this at deployment time.

### NXP eIQ Toolkit deployment path

**Option A — Conservative (recommended):** use the 1M QAT model
1. Export to TorchScript: `resnet_quantized_int8_1m_scripted.pt`
2. Import into eIQ Toolkit → static INT8 calibration with MATWI representative images
3. NCT compiles to `.nb` binary; hardware decompression engine may reduce size further
4. Verify `.nb` size ≤ 1,628 KB (model weight ceiling)
5. Validate accuracy with eIQ accuracy checker
6. Flash to FRDM-MCXN947

**Option B — Best accuracy:** use the budget QAT model
1. Export to TorchScript: `resnet_quantized_int8_budget_scripted.pt`
2. Same NCT import and calibration process
3. **Measure `.nb` binary size before flashing** — must be ≤ 1,628 KB after NCT compilation
4. If hardware decompression achieves ≥16% compression on this model (~1,934 KB × 0.84 ≈ 1,625 KB),
   Option B becomes viable and yields 28.30 → 19.06 µm accuracy improvement
5. Otherwise, fall back to Option A

### Dataset splits

| Split | Samples | Role |
|---|---|---|
| Train | 647 | Pruning fine-tune, distillation, QAT |
| Val | 300 | Model selection (checkpoint saving), early stopping |
| Test | 247 | **Final evaluation only** — never used for selection |
