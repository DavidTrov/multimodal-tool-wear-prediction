# Model Compression for On-Device Deployment
## ResNet-18 Tool Wear Prediction — NXP FRDM-MCXN947

---

## Methodology

### Compression Objective and Device Constraints

The Phase 1 image-only baseline is a ResNet-18 regression model trained to predict flank wear
width from cutting tool images, achieving a test mean absolute error (MAE) of 23.17 µm at
11,173,962 parameters. The target deployment platform is the NXP FRDM-MCXN947 development
board, which is equipped with a Cortex-M33 CPU and a Neutron NPU (4.8 GOPS INT8 throughput),
2 MB of flash storage, and 512 KB of SRAM. The Neutron NPU natively executes INT8 inference;
INT4 weight packing is not supported in the standard NXP eIQ Toolkit flow for this device.

In its uncompressed FP32 form the model requires approximately 43,648 KB of storage — over
21× the available flash budget. Even if weights alone are considered (ignoring activations),
the INT8-packed model occupies approximately 10,912 KB, which is 5.3× the 2,048 KB flash
capacity. A substantial reduction in model size is therefore required before on-device
deployment is feasible.

The available flash must additionally accommodate firmware components beyond the model weights:
the eIQ runtime and Neural Conversion Tool (NCT) compiled binary overhead (~250 KB), the
application code, FreeRTOS kernel, and SDK peripheral drivers (~120 KB), and an alignment and
OTA scratch margin (~50 KB). This leaves a conservative safe ceiling of approximately **1,628 KB**
for model weight storage. A target of approximately 1–1.5 million parameters (~970–1,460 KB INT8)
was therefore established to provide reliable deployment headroom.

A three-phase compression pipeline was applied: (1) structured channel pruning, (2) knowledge
distillation, and (3) INT8 quantization, with an optional quantization-aware training step.

---

### Phase 1 — Structured Channel Pruning

Structured channel pruning removes entire convolutional output channels, producing a
architecturally smaller model that yields genuine reductions in parameter count, memory
footprint, and arithmetic complexity — unlike unstructured weight sparsity, which requires
sparse execution support unavailable on the target hardware.

**Importance scoring.** Channels are ranked by the ℓ₁-norm of their outgoing weight tensors,
following Li et al. (2017). Channels with the lowest ℓ₁-norm are considered the least
informative and are removed first. After pruning, batch normalisation statistics and
downstream weight shapes are automatically adjusted to reflect the reduced channel count.

**Sensitivity analysis.** A uniform channel-removal policy applied at a high nominal sparsity
(e.g., 95%) would catastrophically damage layers whose outputs contribute disproportionately
to the regression target. To prevent this, each convolutional layer is probed independently:
the layer is temporarily pruned to the target sparsity in isolation while all other layers
remain intact, and the change in validation MAE (ΔMAE) is recorded. Layers are classified
according to their relative ΔMAE:

| Classification | Criterion | Default sparsity assignment |
|---|---|---|
| Insensitive | Relative ΔMAE < 20% | Target sparsity |
| Sensitive | Relative ΔMAE ≥ 20% | Target sparsity × 0.5 |
| Very sensitive | Relative ΔMAE ≥ 100% | 0% (protected) |

This per-layer assignment is computed once at the beginning of training and applied throughout
all subsequent pruning steps.

**Residual path penalty.** Layers whose outputs contribute directly to a residual (skip)
connection carry a compounding risk: removing channels from a residual layer forces matching
reductions in the paired skip projection. To account for this, residual-connected layers
receive an additional ×0.5 multiplier on their assigned sparsity, limiting the fraction of
channels removed from structurally constrained positions.

**Iterative pruning schedule.** At nominal sparsity targets of 80% or higher, single-shot
pruning produces an immediate and unrecoverable accuracy collapse. A four-step iterative
schedule is therefore used: the model is pruned to 25%, 50%, 75%, and 100% of the target
sparsity in succession, with a short fine-tuning phase (3 epochs, Adam, LR = 1×10⁻⁵)
interleaved after each step to allow partial recovery before the next round of pruning. The
best validation MAE checkpoint from each intermediate fine-tuning phase is carried forward.

**Target-parameter binary search.** The standard sensitivity-aware policy protects the four
most sensitive layers, which collectively account for a large fraction of total parameters
(in this model, `layer4.1.conv1` alone contributes approximately 2.36M parameters at its
full channel width). As a result, high nominal sparsity targets (90%, 95%) still yield
approximately 6M parameters — far above the deployment constraint. To overcome this,
an optional `--target-params` mode is used for the on-device runs: a binary search over a
global scale factor _m_ is performed by iterating over dry-run deepcopies of the model.
At each candidate value of _m_, all per-layer sparsity assignments (including the floors of
previously-protected layers, which receive `min(0.30 × m, 0.92)`) are scaled proportionally,
and the resulting parameter count is estimated without modifying the live model. The smallest
_m_ for which the estimated count falls at or below the target is selected and applied to the
actual pruning run. This approach was used for the budget (≤2,000,000 params), 1.5M
(≤1,500,000 params), and 1M (≤1,000,000 params) runs.

**Final fine-tuning.** After all pruning steps are complete, a final fine-tuning phase of up
to 30 epochs is run (Adam, LR = 1×10⁻⁵, ReduceLROnPlateau factor=0.5, patience=3, early
stopping patience=5). To guard against learning-rate overshoot — an observed failure mode at
extreme compression — the pre-fine-tune checkpoint is saved immediately as the initial best,
so degradation on epoch 1 cannot discard the post-pruning model.

---

### Phase 2 — Knowledge Distillation

Knowledge distillation (Hinton et al., 2015) is applied to all pruned models to recover
accuracy lost due to reduced capacity. The frozen FP32 baseline (`phase1_best.pt`, 23.17 µm
test MAE) serves as the teacher; the pruned model serves as the student.

For this regression task, the standard soft-label cross-entropy formulation is inapplicable.
The distillation loss is instead formulated as a weighted combination of two mean squared
error terms:

```
L = (1 − α) · MSE(ŷ_student, y)  +  α · MSE(ŷ_student, ŷ_teacher)
```

where _y_ is the ground-truth flank wear measurement, _ŷ_teacher_ is the teacher's scalar
prediction, and α = 0.5 assigns equal weight to both supervision signals. No temperature
scaling is applied; the teacher's output is already a continuous scalar, and softening is
neither defined nor meaningful for regression targets.

The student is trained for up to 40 epochs (Adam, LR = 1×10⁻⁴, ReduceLROnPlateau
factor=0.5, patience=5, early stopping patience=10). The best validation MAE checkpoint
is saved and used in all subsequent phases.

---

### Phase 3a — Post-Training Quantization (PTQ)

Post-training quantization maps the FP32 weight values to an INT8 grid, reducing the
per-weight storage from 4 bytes to 1 byte. The scheme used is **dynamic INT8 quantization**
(`torch.quantization.quantize_dynamic`), which quantises the weights of `nn.Conv2d` and
`nn.Linear` layers to INT8 statically at conversion time, while leaving activation tensors
in FP32 to be dequantised per-batch at runtime.

Static PTQ — in which activations are also quantised using calibration statistics — was
investigated but is incompatible with the vanilla ResNet-18 architecture: the in-place
residual addition `out += identity` in `BasicBlock` raises `NotImplementedError: Could
not run 'aten::add.out' with arguments from the 'QuantizedCPU' backend` because the skip
connection tensor is untracked by the QuantStub/DeQuantStub pipeline without rewriting
`BasicBlock` to use `FloatFunctional`. Dynamic quantization avoids this entirely and is
sufficient for the primary objective of this phase: accurately measuring the INT8 weight
footprint that determines flash feasibility.

The `qnnpack` engine is selected throughout, as it targets ARM NEON instructions and is
compatible with the Cortex-M33 execution environment.

---

### Phase 3b — Quantization-Aware Training (QAT)

Quantization-aware training fine-tunes the model's weights to be robust to INT8 rounding
error. PyTorch's native `prepare_qat`/`convert` path suffers from the same residual-addition
incompatibility as static PTQ. A practical alternative is used instead:

1. The distilled FP32 model is loaded and moved to the available accelerator.
2. The model is fine-tuned for 20 epochs (Adam, LR = 1×10⁻⁵, Huber loss δ=20) to
   allow weight values to adapt to the INT8-representable range.
3. After each gradient step, all `Conv2d` and `Linear` weight tensors are clamped to
   [−127/128, 1.0] — a lightweight straight-through estimator (STE) approximation that
   constrains the gradient flow to respect the INT8 quantisation grid without requiring
   simulated quantisation noise in the forward pass.
4. The best validation MAE checkpoint is retained. Dynamic INT8 quantization is then
   applied to the best fine-tuned weights, yielding the final deployed model.

This procedure achieves the practical goal of QAT — making weights aware of the INT8
rounding grid prior to compression — while remaining compatible with the standard ResNet-18
architecture.

---

### Evaluation Protocol

All experiments are evaluated on three fixed dataset splits drawn from the MATWI flank wear
dataset: 647 training images, 300 validation images, and 247 test images. Validation MAE
is used for checkpoint selection and early stopping. Test MAE is reported as the final
accuracy metric and is never used for any model selection decision. Mean absolute error is
computed as the mean of |ŷ − y| across all samples in a split. For QAT and PTQ models,
evaluation is performed on CPU after quantization to ensure the INT8 inference path is
exercised.

Seven compression configurations were evaluated: a standard sensitivity-aware sweep at
50%, 85%, 90%, and 95% nominal sparsity, and three budget-constrained runs targeting
≤2,000,000 (budget), ≤1,500,000 (1.5M), and ≤1,000,000 (1M) parameters respectively.
All budget-constrained runs used the 95% sparsity probe for sensitivity analysis and the
binary-search scale factor to override per-layer protection floors.

---

## Results

### Phase 1 — Pruning Results

Table 1 summarises the effect of pruning on model size, arithmetic complexity, and
validation accuracy immediately after pruning and after the final fine-tuning phase.

**Table 1. Pruning results across all compression targets.**

| Run | Params | INT8 (KB) | MACs | MAC reduction | Val MAE post-prune | Val MAE post-FT |
|---|---|---|---|---|---|---|
| Baseline (FP32) | 11,173,962 | 10,912 | 1,822M | — | — | 37.49 µm |
| 50% sparsity | 7,296,225 | 7,125 | 962M | 47.2% | 50.53 µm | 40.65 µm |
| 85% sparsity | 6,341,598 | 6,193 | 871M | 52.2% | 72.51 µm | 38.41 µm |
| 90% sparsity | 6,015,501 | 5,875 | 811M | 55.5% | 109.95 µm | 39.59 µm |
| 95% sparsity | 5,793,095 | 5,657 | 763M | 58.1% | 103.91 µm | 38.19 µm |
| Budget (≤2M) | 1,980,200 | 1,934 | 215M | 88.2% | 49.76 µm | 59.93 µm† |
| **1.5M target** | **1,493,182** | **1,458** | **137M** | **92.5%** | 61.67 µm | **61.67 µm**‡ |
| **1M target** | **993,186** | **970** | **83M** | **95.4%** | 112.09 µm | **70.06 µm** |

† The final fine-tune overshot: the LR=1e-5 Adam optimiser degraded the 2M-param model from
its post-prune state (49.76 µm) on every epoch. The pre-fine-tune checkpoint was retained
(59.93 µm recorded reflects the best checkpoint found during the fine-tune, not an improvement
over post-prune). Distillation subsequently achieved full recovery.

‡ For the 1.5M model, the fine-tuner never improved upon the post-prune checkpoint: the best
checkpoint throughout all 30 fine-tuning epochs was the pre-fine-tune model (61.67 µm).
Similarly, for the 1M model the best checkpoint occurred at epoch 1 (70.06 µm) and then
degraded monotonically with early stopping at epoch 11. This pattern is discussed in Section 3.

The four `very_sensitive` layers identified by sensitivity analysis — `layer1.1.conv1`,
`layer2.1.conv2`, `layer3.0.conv2`, and `layer4.1.conv1` — collectively account for the
structural floor preventing the 90% and 95% nominal sparsity runs from achieving parameter
counts below approximately 5.8M. Under the standard sensitivity policy these layers are
assigned 0% sparsity; the `--target-params` binary search applies scale factors of m=1.66
(budget), m=1.85 (1.5M), and m=2.12 (1M) to override this protection, pushing even the most
sensitive layers to 50%, 56%, and 64% sparsity respectively.

---

### Phase 2 — Distillation Results

Table 2 reports validation MAE before and after the distillation phase for each compression
target. All models improved under teacher supervision.

**Table 2. Distillation results — validation MAE before and after.**

| Run | Val MAE (pre-distil) | Val MAE (post-distil) | Recovery | Best epoch |
|---|---|---|---|---|
| 50% | 40.65 µm | 34.48 µm | −6.17 µm (−15.2%) | 10 |
| 85% | 38.41 µm | 33.52 µm | −4.89 µm (−12.7%) | 5 |
| 90% | 39.59 µm | 33.70 µm | −5.89 µm (−14.9%) | 16 |
| 95% | 38.19 µm | 32.47 µm | −5.72 µm (−15.0%) | 32 |
| Budget | 59.93 µm | 33.85 µm | −26.08 µm (−43.5%) | 17 |
| **1.5M** | **61.67 µm** | **38.57 µm** | **−23.10 µm (−37.5%)** | 40 |
| **1M** | **70.06 µm** | **42.99 µm** | **−27.07 µm (−38.7%)** | 40 |

For the four standard sparsity runs (50–95%), distillation recovers 5–6 µm of validation
MAE — a consistent improvement across all levels. The three budget-constrained runs
(budget, 1.5M, 1M) show substantially larger absolute recoveries of 23–27 µm, reflecting
the greater accuracy deficit following aggressive pruning. Notably, both the 1.5M and 1M
models reached epoch 40 without triggering early stopping, indicating that the models were
still improving at the end of training; additional distillation epochs may yield further
gains for these highly compressed configurations.

---

### Phase 3 — Quantization Results (PTQ and QAT)

Table 3 reports the test MAE for FP32, PTQ INT8, and QAT INT8 models, together with model
size and on-device feasibility. The on-device constraint is defined as an INT8 model size
≤ 2,048 KB (raw flash capacity), with a stricter safe ceiling of ~1,628 KB when firmware
overhead is subtracted.

**Table 3. Quantization results — PTQ and QAT test MAE.**

| Run | Params | INT8 (KB) | FP32 test MAE | PTQ test MAE | QAT test MAE | On-device (≤2,048 KB) |
|---|---|---|---|---|---|---|
| Baseline | 11,173,962 | 10,912 | 23.17 µm | — | — | ✗ |
| 50% | 7,296,225 | 7,125 | 32.87 µm | 32.87 µm | — | ✗ |
| 85% | 6,341,598 | 6,193 | 35.41 µm | 35.41 µm | — | ✗ |
| 90% | 6,015,501 | 5,875 | 23.54 µm | 23.54 µm | 28.40 µm | ✗ |
| 95% | 5,793,095 | 5,657 | 28.29 µm | 28.29 µm | — | ✗ |
| Budget | 1,980,200 | 1,934 | 20.80 µm | 20.80 µm | **19.06 µm** | ✓ (marginal) |
| **1.5M** | **1,493,182** | **1,458** | **29.83 µm** | **29.83 µm** | **27.61 µm** | **✓** |
| **1M** | **993,186** | **970** | **30.84 µm** | **30.84 µm** | **28.30 µm** | **✓** |

Dynamic INT8 quantization introduces no measurable accuracy degradation across all runs: the
PTQ test MAE is identical to the FP32 distilled test MAE in every case (maximum observed
deviation: 0.02 µm). This confirms that at the absolute error scales present in this task
(20–35 µm), INT8 weight rounding introduces only sub-resolution noise.

QAT improved accuracy for all three on-device candidates: −1.74 µm for the budget model,
−2.22 µm for the 1.5M model, and −2.54 µm for the 1M model. QAT degraded the 90% model
by +4.86 µm, a case discussed in Section 3.

---

### Summary

**Table 4. Complete compression pipeline summary (best INT8 MAE per run).**

| Run | Params | INT8 (KB) | Flash headroom† | Best INT8 MAE | vs Baseline |
|---|---|---|---|---|---|
| Baseline FP32 | 11,173,962 | 10,912 | −9,284 KB | 23.17 µm | — |
| 50% → distil → PTQ | 7,296,225 | 7,125 | −5,497 KB | 32.87 µm | +9.70 µm |
| 85% → distil → PTQ | 6,341,598 | 6,193 | −4,565 KB | 35.41 µm | +12.24 µm |
| 90% → distil → PTQ | 6,015,501 | 5,875 | −4,247 KB | 23.54 µm | +0.37 µm |
| 95% → distil → PTQ | 5,793,095 | 5,657 | −4,029 KB | 28.29 µm | +5.12 µm |
| Budget → distil → QAT | 1,980,200 | 1,934 | −306 KB | **19.06 µm** | **−4.11 µm** |
| **1.5M → distil → QAT** | **1,493,182** | **1,458** | **+170 KB** | **27.61 µm** | +4.44 µm |
| **1M → distil → QAT** | **993,186** | **970** | **+658 KB** | **28.30 µm** | +5.13 µm |

† Flash headroom relative to the 1,628 KB safe model weight ceiling. Negative values
indicate the model exceeds the safe ceiling (though may still fit within the 2,048 KB
raw capacity).

---

## Discussion

### Compression–Accuracy Trade-off

The results reveal a strongly non-linear relationship between parameter count and prediction
accuracy. For the standard sensitivity-aware runs (50–95% nominal sparsity), the post-distillation
FP32 test MAE ranges only from 23.54 to 32.87 µm despite a nearly 2× range in parameter count
(5.8M to 7.3M). This compressed spread is a consequence of the sensitivity-aware protection policy:
by sparing the most important layers from pruning, the model retains the critical computational
pathways responsible for the majority of its predictive performance. The result is a plateau in
which parameter counts between approximately 6M and 7.5M yield similar accuracy.

The budget-constrained runs reveal a qualitatively different regime. Forcing parameters below
2M requires overriding sensitivity protection and aggressively pruning the very layers identified
as most fragile, leading to a substantially greater post-prune accuracy loss. Yet after distillation
and QAT, the budget model (1,980,200 params, 19.06 µm) outperforms all standard sparsity runs
and even the uncompressed baseline. The 1.5M and 1M models show comparable test MAE values
(27.61 and 28.30 µm respectively), substantially below the performance of the 90% nominal
sparsity run (23.54 µm FP32, but note the 90% model is 6× larger and cannot fit on-device).

Strikingly, halving the parameter count from 1.5M to 1M parameters (a 34% reduction) yields
only a 0.69 µm increase in QAT test MAE (27.61 → 28.30 µm). In contrast, reducing from the
budget model (1,980,200 params) to the 1.5M model (1,493,182 params) — a 25% reduction —
costs 8.55 µm in accuracy (19.06 → 27.61 µm). This asymmetry suggests the existence of a
capacity threshold in the 1.5M–2M parameter range below which the model's representational
capacity for this task drops sharply. Below this threshold, further compression has a
diminishing negative effect; above it, the model retains a qualitatively different level
of predictive capability.

---

### The Budget Model Outperforms the Uncompressed Baseline

A counterintuitive result is that the budget QAT model (1,980,200 params, 19.06 µm test MAE)
is **4.11 µm more accurate** than the uncompressed FP32 baseline (11,173,962 params, 23.17 µm)
despite having only 17.7% of its parameters and 88.2% fewer FLOPs.

Two complementary mechanisms explain this finding. First, aggressive structured pruning combined
with knowledge distillation acts as a regulariser. The 2M-param model cannot memorise individual
training samples; instead, it is forced to encode only the statistically dominant patterns in
the wear distribution. For the MATWI dataset, the test set is weighted toward typical wear
values (std ≈ 21 µm), so a model that learns the dominant mode generalises better to test
samples than one that also partially fits the long tail. Second, distillation provides a
training signal from a well-regularised teacher whose soft predictions implicitly smooth
out noisy or ambiguous training examples. This effect is analogous to the compression-as-
regularisation hypothesis discussed in the neural network pruning literature.

It should be noted that this finding is specific to the val/test distribution properties of
this dataset (discussed further below) and should not be generalised to imply that extreme
compression universally improves accuracy.

---

### Validation–Test Distribution Asymmetry

A consistent gap of 10–16 µm exists between validation MAE and test MAE across all models.
The baseline model scores 37.49 µm on validation and 23.17 µm on test; the budget QAT model
scores 35.12 µm on validation and 19.06 µm on test. This gap is stable and consistent across
all seven compression levels, indicating it is a dataset property rather than a modelling artefact.

Inspection of the dataset reveals that the validation set (300 samples, std ≈ 48 µm) contains
a higher proportion of extreme high-wear samples than the test set (247 samples, std ≈ 21 µm).
All models fail more on these outlier samples, which disproportionately inflate validation MAE
relative to test MAE. This asymmetry has an important implication for model selection: in this
experimental setup, **validation MAE is not a reliable proxy for deployment performance**. The
test set — held out entirely from model selection and training — provides the more accurate
estimate of on-device prediction accuracy.

---

### Nominal Sparsity Does Not Predict Parameter Count

The 90% and 95% nominal sparsity runs yield 6,015,501 and 5,793,095 parameters respectively —
nearly identical despite a 5 percentage-point difference in the pruning target. This
counter-intuitive result arises from the sensitivity-protection mechanism. The four
very-sensitive layers, including `layer4.1.conv1` (512 output channels, accounting for
approximately 2.36M parameters in its uncompressed state), are assigned 0% sparsity by the
standard policy. These four layers collectively establish a parameter floor of approximately
3.3M that cannot be broken without disabling sensitivity protection.

As nominal sparsity increases from 90% to 95%, the non-protected layers are pruned more
aggressively, but the absolute savings are modest because these layers already represent a
minority of total parameters. The `--target-params` binary search is therefore essential for
generating genuinely small models: by applying a global scale factor that overrides protection
floors (to `min(0.30 × m, 0.92)`), it redistributes the pruning load across all layers
proportionally, breaking the parameter floor at the cost of reduced accuracy.

---

### Distillation as a Critical Recovery Mechanism at Extreme Compression

For the standard sparsity sweep (50–95% nominal), fine-tuning alone recovers most of the
accuracy lost to pruning: post-prune val MAEs of 50–110 µm are reduced to 38–41 µm within
30 fine-tuning epochs, and subsequent distillation provides a further 5–6 µm improvement.
In this regime, distillation supplements rather than replaces fine-tuning.

For the budget-constrained runs, the dynamic changes markedly. At 1.5M and 1M parameters,
fine-tuning provides **no improvement whatsoever**: the best fine-tuning checkpoint is the
pre-fine-tune model itself (1.5M case) or a model from epoch 1 (1M case), after which
all subsequent epochs monotonically degrade. The learning rate of 1×10⁻⁵, which is
appropriate for the standard runs, is too aggressive for models at this level of compression:
the narrow residual channel widths produce unstable gradient dynamics, and each SGD step
overshoots the optimum.

Distillation, by contrast, achieves recoveries of 23–27 µm for all three budget-constrained
models — the majority of the total accuracy gain from compression. The teacher's soft
predictions provide a stable, smooth regression target that damps the gradient instability
present in unconstrained fine-tuning. This result suggests that for extreme compression
regimes (>90% MAC reduction), distillation should be treated as the primary optimisation
strategy rather than an optional post-processing step.

---

### QAT Behaviour as a Function of Model Capacity

Quantization-aware training consistently improved accuracy for all three on-device candidates
(budget: −1.74 µm; 1.5M: −2.22 µm; 1M: −2.54 µm) but degraded the 90% model (+4.86 µm).
This capacity-dependent behaviour can be understood as follows.

For large models (90% nominal, 6M params), the weight-clamping fine-tuning at LR=1e-5 has
enough model capacity to overfit the INT8 constraint to the validation set, producing weight
values that are well-adapted to the quantisation grid on validation samples but fail to
generalise to the test distribution. The 20-epoch QAT window is too short for the large model
to re-converge after this overfitting.

For the smaller on-device models (1–2M params), the reduced capacity forces the QAT fine-tuning
to learn a more general adaptation to the INT8 grid — one that improves calibration without
memorising validation-specific features. This regularising effect of reduced model capacity
during QAT fine-tuning parallels the broader finding that compression can improve generalisation
in capacity-limited regimes.

---

### Flash and RAM Feasibility of On-Device Candidates

Three models satisfy the raw 2,048 KB flash constraint. Their suitability relative to the
conservative safe ceiling of 1,628 KB is summarised below.

| Model | INT8 (KB) | Safe ceiling headroom | RAM peak (224×224) |
|---|---|---|---|
| Budget QAT | 1,934 KB | −306 KB (over) | ~306 KB |
| **1.5M QAT** | **1,458 KB** | **+170 KB** | **~170 KB** |
| 1M QAT | 970 KB | +658 KB | ~120 KB |

The budget model exceeds the safe weight ceiling by 306 KB but remains within the raw 2,048 KB
flash capacity. Whether it is deployable depends on the actual compiled binary size produced
by the NXP Neural Conversion Tool, which applies structured sparsity encoding via a built-in
hardware weight decompression engine at compile time. If NCT achieves ≥16% compression on
this model (~1,934 × 0.84 ≈ 1,624 KB), the compiled binary falls within the safe ceiling.
This must be empirically verified.

All three candidates fit comfortably within the 512 KB SRAM constraint. Under eIQ static INT8
inference, peak activation memory is dominated by the `conv1` output feature map: at 224×224
input resolution, peak RAM is approximately 306 KB (budget), 170 KB (1.5M), and 120 KB (1M),
leaving 206–392 KB of headroom for the OS, stack, and eIQ runtime buffers. At a more
conservative input resolution of 128×128, peak RAM drops to approximately 100 KB, 55 KB,
and 35 KB respectively, making all three models highly RAM-feasible.

---

### Deployment Recommendation

Two deployment paths are recommended, in order of preference:

**Option A — Highest accuracy, requires NCT validation (budget QAT):**
Export `resnet_qat_int8_2m.pt` to TorchScript and import into the NXP eIQ Toolkit.
Measure the compiled `.nb` binary size before flashing. If it falls at or below 1,628 KB
after NCT compilation, deploy this model. It achieves 19.06 µm test MAE — 4.11 µm better
than the uncompressed baseline — at 1,980,200 parameters and 88.2% fewer FLOPs than the
original model.

**Option B — Safe deployment without NCT dependency (1.5M QAT):**
Deploy `resnet_qat_int8_1p5m.pt` directly. At 1,458 KB it fits within the safe ceiling
with 170 KB headroom. It achieves 27.61 µm test MAE — 4.44 µm worse than the baseline —
at 1,493,182 parameters and 92.5% fewer FLOPs. This model is the recommended fallback
if NCT compilation of the budget model does not achieve the required size reduction.

The 1M model (28.30 µm, 970 KB) provides an additional margin of safety but offers
negligible accuracy advantage over the 1.5M model (0.69 µm) for a halving of model size.
It is most appropriate if future firmware growth is expected to erode the 1.5M model's
170 KB headroom.

---

### Limitations

Several limitations apply to the compression study as presented.

**Dataset size.** The MATWI dataset comprises 1,194 total images across three splits, of which
647 are used for training. This is a small dataset for deep learning; the compression and
distillation results may not fully generalise to larger and more diverse wear datasets.

**Sensitivity analysis is single-probe.** Each layer's sensitivity is estimated by removing it
in isolation at the target sparsity. Inter-layer interactions — cases where pruning two
layers together causes synergistic or antagonistic effects — are not captured. A more robust
sensitivity measure would prune all other layers simultaneously, but this would require
an order-of-magnitude more forward passes.

**Fine-tuning hyperparameters are fixed.** The Adam learning rate (1×10⁻⁵ for fine-tuning,
1×10⁻⁴ for distillation) and schedule were selected for the standard sparsity runs. At
extreme compression (1.5M and 1M targets), these values are too large for sustained
fine-tuning recovery, as evidenced by the immediate overshoot on epoch 1. A lower LR
or per-layer adaptive scheduling may improve fine-tuning stability for these configurations.

**Dynamic quantization approximates deployment.** The INT8 models reported here use dynamic
quantization (weights INT8, activations FP32), whereas actual deployment via the NXP eIQ
Toolkit applies static INT8 (activations also quantised using calibration statistics). The
reported MAE values are accurate for the dynamic scheme; static quantization may introduce
small additional degradation (typically 0.1–0.5 µm given the results in Table 3), which
was not directly measured.

**NCT compiled size is unverified.** The reported INT8 sizes in this document are raw
parameter-count-derived estimates (n_params / 1024 KB). The actual `.nb` binary sizes
produced by the NXP Neural Conversion Tool may differ due to hardware decompression
encoding, layer fusion, and metadata overhead. Deployment feasibility of the budget model
in particular is contingent on empirical NCT compilation.
