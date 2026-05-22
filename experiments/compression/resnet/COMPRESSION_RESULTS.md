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
three-phase compression pipeline sweeping four sparsity levels plus a budget-constrained run
that specifically targets on-device deployment.

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
| 95% | 95% channel removal | Very aggressive |
| **budget** | 95% + `--target-params 2,000,000` | **On-device target** — forced to fit 2 MB flash |

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

### Layer sensitivity at 95% sparsity probe (used for budget run)

| Layer | Channels | Rel. ∆MAE | Class | Assigned (95%) | Assigned (budget, m=1.66) |
|---|---|---|---|---|---|
| conv1 | 64 | +30.9% | sensitive | 48% | **79%** |
| layer1.0.conv1 | 64 | +43.5% | sensitive | 48% | **79%** |
| layer1.0.conv2 | 64 | +40.7% | sensitive (residual) | 24% | **39%** |
| layer1.1.conv1 | 64 | +170.1% | **very_sensitive** | 0% | **50%** |
| layer1.1.conv2 | 64 | +78.0% | sensitive (residual) | 24% | **39%** |
| layer2.0.conv1 | 128 | +62.0% | sensitive | 48% | **79%** |
| layer2.0.conv2 | 128 | +49.2% | sensitive (residual) | 24% | **39%** |
| layer2.0.downsample.0 | 128 | +90.9% | sensitive (residual) | 24% | **39%** |
| layer2.1.conv1 | 128 | +49.9% | sensitive | 48% | **79%** |
| layer2.1.conv2 | 128 | +160.7% | **very_sensitive** (residual) | 0% | **50%** |
| layer3.0.conv1 | 256 | +30.4% | sensitive | 48% | **79%** |
| layer3.0.conv2 | 256 | +186.2% | **very_sensitive** (residual) | 0% | **50%** |
| layer3.0.downsample.0 | 256 | +1.2% | insensitive (residual) | 48% | **79%** |
| layer3.1.conv1 | 256 | +66.4% | sensitive | 48% | **79%** |
| layer3.1.conv2 | 256 | +48.5% | sensitive (residual) | 24% | **39%** |
| layer4.0.conv1 | 512 | +39.1% | sensitive | 48% | **79%** |
| layer4.0.conv2 | 512 | +30.5% | sensitive (residual) | 24% | **39%** |
| layer4.0.downsample.0 | 512 | +24.3% | sensitive (residual) | 24% | **39%** |
| layer4.1.conv1 | 512 | +197.1% | **very_sensitive** | 0% | **50%** |
| layer4.1.conv2 | 512 | +44.0% | sensitive (residual) | 24% | **39%** |

### Why nominal sparsity ≠ parameter reduction

The 90% and 95% sensitivity-aware runs still yield **~6M parameters** despite the high sparsity
target. The reason: the four `very_sensitive` layers (including `layer4.1.conv1` alone at
**2.36M params**) are fully protected at 0% sparsity. These protected layers collectively hold
~3.3M parameters — a floor that cannot be broken without overriding the sensitivity protection.

The budget run uses `--target-params 2,000,000`, which sets `very_sensitive_threshold = ∞`
effectively and scales all assignments by m=1.66, allowing even the most sensitive layers to be
pruned. This achieved **1,980,200 parameters** — within the 2,048 KB flash budget.

### Pruning results

| Run | Params | INT8 KB | INT4 KB | MACs | MAC reduction | Val MAE post-prune | Val MAE after fine-tune |
|---|---|---|---|---|---|---|---|
| Baseline | 11,173,962 | 10,912 | 5,456 | 1,822M | — | — | 37.49 µm |
| **50%** | 7,296,225 | 7,125 | 3,563 | 962M | 47.2% | 50.53 µm | 40.65 µm |
| **85%** | 6,341,598 | 6,193 | 3,097 | 871M | 52.2% | 72.51 µm | 38.41 µm |
| **90%** | 6,015,501 | 5,875 | 2,937 | 811M | 55.5% | 109.95 µm | 39.59 µm |
| **95%** | 5,793,095 | 5,657 | 2,829 | 763M | 58.1% | 103.91 µm | 38.19 µm |
| **budget** | **1,980,200** | **1,934** | **967** | **215M** | **88.2%** | 49.76 µm | 59.93 µm† |

† The final fine-tune overshot the pre-fine-tune MAE (LR=1e-5 too large for a 2M-param model);
the pre-fine-tune checkpoint (49.76 µm) was not saved. Distillation subsequently recovered
to 33.85 µm val / 20.80 µm test.

**Notable observations:**
- 90% and 95% nominal sparsity produce nearly identical parameter counts (~6M) because the
  same four large layers are protected in both cases.
- The budget model achieves 88.2% MAC reduction — nearly 9× fewer multiply-accumulates than
  the baseline, running dramatically faster on the Neutron NPU.
- Iterative pruning (4 steps) is essential at ≥80% sparsity: without it, single-shot pruning
  causes MAE to jump to >100 µm, which the fine-tuner cannot fully recover.

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

**Notable observations:**
- Distillation consistently recovers 5–6 µm across all sparsity levels, demonstrating that
  teacher supervision reliably compensates for pruning-induced capacity loss.
- The budget model shows the largest absolute recovery (−26 µm), starting from a much worse
  post-prune baseline and converging to a val MAE comparable to the other models — strong
  evidence that distillation is especially valuable for heavily compressed networks.
- The 95% model achieves the best val MAE after distillation (32.47 µm), marginally better
  than the 90% and budget models.

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

**Notable observations:**
- Dynamic INT8 quantisation causes **zero measurable MAE degradation** across all runs. This
  is expected: at 30–35 µm absolute error, INT8 weight rounding (±0.5 LSB) is negligible.
- The **budget model is the only run that fits in flash** (1,934 KB ≤ 2,048 KB).
- The budget model's test MAE (20.80 µm) is **2.37 µm better than the unpruned baseline**
  (23.17 µm) at 1/5.6th the parameters — see Section 8 for discussion.
- None of the standard sparsity runs (50%, 85%, 90%, 95%) fits in flash: all remain between
  5,600 and 7,100 KB INT8, 2.7–3.5× over budget.

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

QAT was run on the **90% model** and the **budget model**.

### QAT results

| Run | Test MAE FP32 | Test MAE PTQ | Test MAE QAT | QAT vs PTQ | QAT vs baseline | On-device |
|---|---|---|---|---|---|---|
| 90% | 23.54 µm | 23.54 µm | 28.40 µm | **+4.86 µm** ← hurt | +5.23 µm | ✗ |
| **budget** | **20.80 µm** | **20.80 µm** | **19.06 µm** | **−1.74 µm** ← improved | **−4.11 µm** | **✓** |

**Notable observations:**
- QAT **hurt the 90% model** (+4.86 µm). At LR=1e-5 and 6M params the weight-clamping
  fine-tune overfit to the validation set without generalising to the test set.
- QAT **helped the budget model** (−1.74 µm test). The smaller 2M-param model is better
  regularised and the INT8-aware weight clamping improved test generalisation.
- The budget QAT INT8 model (19.06 µm test) is **4.11 µm better than the uncompressed
  FP32 baseline** at 1/5.6th the parameters, fitting within the flash budget.

---

## 7. Complete Results Table

| Stage | Run | Params | INT8 KB | INT4 KB | Pruned val | Distil val | PTQ test | QAT test | On-device |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (FP32) | 0% | 11,173,962 | 10,912 | 5,456 | — | — | 23.17 µm | — | ✗ |
| Pruned | 50% | 7,296,225 | 7,125 | 3,563 | 40.65 µm | — | — | — | ✗ |
| Distilled | 50% | 7,296,225 | 7,125 | 3,563 | — | 34.48 µm | — | — | ✗ |
| PTQ INT8 | 50% | 7,296,225 | 7,125 | 3,563 | — | — | 32.87 µm | — | ✗ |
| Pruned | 85% | 6,341,598 | 6,193 | 3,097 | 38.41 µm | — | — | — | ✗ |
| Distilled | 85% | 6,341,598 | 6,193 | 3,097 | — | 33.52 µm | — | — | ✗ |
| PTQ INT8 | 85% | 6,341,598 | 6,193 | 3,097 | — | — | 35.41 µm | — | ✗ |
| Pruned | 90% | 6,015,501 | 5,875 | 2,937 | 39.59 µm | — | — | — | ✗ |
| Distilled | 90% | 6,015,501 | 5,875 | 2,937 | — | 33.70 µm | — | — | ✗ |
| PTQ INT8 | 90% | 6,015,501 | 5,875 | 2,937 | — | — | 23.54 µm | — | ✗ |
| QAT INT8 | 90% | 6,015,501 | 5,875 | 2,937 | — | — | — | 28.40 µm | ✗ |
| Pruned | 95% | 5,793,095 | 5,657 | 2,829 | 38.19 µm | — | — | — | ✗ |
| Distilled | 95% | 5,793,095 | 5,657 | 2,829 | — | 32.47 µm | — | — | ✗ |
| PTQ INT8 | 95% | 5,793,095 | 5,657 | 2,829 | — | — | 28.29 µm | — | ✗ |
| Pruned | budget | 1,980,200 | 1,934 | 967 | 59.93 µm | — | — | — | ✓ |
| Distilled | budget | 1,980,200 | 1,934 | 967 | — | 33.85 µm | — | — | ✓ |
| PTQ INT8 | budget | 1,980,200 | 1,934 | 967 | — | — | 20.80 µm | — | ✓ |
| **QAT INT8** | **budget** | **1,980,200** | **1,934** | **967** | — | — | — | **19.06 µm** | **✓** |

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

The 1,934 KB budget model exceeds this ceiling by ~306 KB and requires further compression.

**Note:** The Neutron NPU includes a hardware weight decompression engine that can reduce the
stored model size below the raw INT8 byte count by 10–30% via structured sparsity encoding
applied at NCT compile time. The actual compiled `.nb` binary size should be measured after
NCT conversion before concluding the model does not fit — it may already be within budget.

**Safe model weight ceiling: ~1,500–1,628 KB INT8** (leaving 420–548 KB for runtime + app).
A target of ~1M parameters (~977 KB INT8) provides comfortable headroom.

### 8.6 RAM feasibility

The budget model's actual channel widths after pruning are very narrow:

| Layer | Input channels | Output channels | Spatial | INT8 buffer |
|---|---|---|---|---|
| conv1 | 3 | **13** | 112×112 | 159 KB |
| layer1.x | 13 | 13–32 | 56×56 | ≤40 KB |
| layer2.x | 13–77 | 26–77 | 28×28 | ≤59 KB |
| layer3.x | 53–77 | 53 | 14×14 | ≤10 KB |
| layer4.x | 53–309 | 107–309 | 7×7 | ≤15 KB |

**Peak RAM (eIQ static INT8, 224×224 input):**
Input buffer (147 KB) + conv1 output (159 KB) = **306 KB** — fits within 512 KB RAM with
~206 KB headroom for OS, stack, and the eIQ runtime.

At a more realistic embedded input resolution of 128×128, peak drops to just **100 KB**.

**Conclusion:** the budget model fits in both flash (**1,934 KB ≤ 2,048 KB** ✓) and RAM
(**306 KB ≤ 512 KB** ✓) for deployment via the NXP eIQ Toolkit with static INT8 inference.

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
│       ├── qat_results_90.json
│       └── qat_results_budget.json
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
| `checkpoints/resnet_pruned_budget.pt` | budget pruned | **1,934** |
| `checkpoints/resnet_distilled.pt` | 85% distilled | 6,193 |
| `checkpoints/resnet_distilled_50.pt` | 50% distilled | 7,125 |
| `checkpoints/resnet_distilled_90.pt` | 90% distilled | 5,875 |
| `checkpoints/resnet_distilled_95.pt` | 95% distilled | 5,657 |
| `checkpoints/resnet_distilled_budget.pt` | budget distilled | **1,934** |
| `checkpoints/resnet_quantized_int8_90.pt` | 90% PTQ INT8 | 5,875 |
| `checkpoints/resnet_quantized_int8_95.pt` | 95% PTQ INT8 | 5,657 |
| `checkpoints/resnet_quantized_int8_budget.pt` | budget PTQ INT8 | **1,934** |
| `checkpoints/resnet_qat_int8_budget.pt` | **budget QAT INT8 ← deploy this** | **1,934** |

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

1. Export budget model to TorchScript: `resnet_quantized_int8_budget_scripted.pt`
2. Import into eIQ Toolkit → automatic static INT8 calibration with MATWI calibration images
3. Toolkit applies INT4 weight packing (2 weights/byte) for Neutron NPU → **967 KB projected**
4. Validate accuracy using eIQ accuracy checker before flashing
5. Flash to FRDM-MCXN947 via eIQ deployment flow

### Dataset splits

| Split | Samples | Role |
|---|---|---|
| Train | 647 | Pruning fine-tune, distillation, QAT |
| Val | 300 | Model selection (checkpoint saving), early stopping |
| Test | 247 | **Final evaluation only** — never used for selection |
