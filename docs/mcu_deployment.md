# MCU Deployment: Tool Wear Prediction on NXP FRDM-MCXN947

End-to-end documentation for deploying the full pipeline — CWT scalogram
preprocessing + CNN wear regression — on the NXP FRDM-MCXN947
(Cortex-M33 @ 150 MHz, 512 KB SRAM, 2 MB flash, eIQ Neutron NPU).

---

## Table of Contents

- [Hardware Target](#hardware-target)
- [Pipeline Overview](#pipeline-overview)
- [Model Selection and Memory Analysis](#model-selection-and-memory-analysis)
- [Toolchain](#toolchain)
- [Step 1 — Export to ONNX](#step-1--export-to-onnx)
- [Step 2 — Build Calibration Dataset](#step-2--build-calibration-dataset)
- [Step 3 — INT8 Quantization via ONNX2Quant](#step-3--int8-quantization-via-onnx2quant)
- [Step 4 — Convert to TFLite via onnx2tflite](#step-4--convert-to-tflite-via-onnx2tflite)
- [Step 5 — Generate C Array for Flash](#step-5--generate-c-array-for-flash)
- [Step 6 — MCUXpresso Project](#step-6--mcuxpresso-project)
- [Step 7 — NPU Acceleration (future)](#step-7--npu-acceleration-future)
- [Accuracy Results](#accuracy-results)
- [Memory Budget](#memory-budget)
- [GroupNorm Compatibility](#groupnorm-compatibility)
- [File Inventory](#file-inventory)

---

## Hardware Target

| Resource | Specification |
|---|---|
| MCU | NXP MCXN947, dual Cortex-M33 @ 150 MHz |
| SRAM | 512 KB |
| Flash | 2 MB |
| FPU | Single-precision only (no hardware double) |
| NPU | eIQ Neutron N1-16 (on-chip neural accelerator) |
| Board | FRDM-MCXN947 |

The single-precision FPU constraint affects the CWT preprocessing: all signal
processing in `cwt_mcu.c` uses `float` (float32), not `double`.

---

## Pipeline Overview

The pipeline runs sequentially to stay within the 512 KB SRAM budget. CWT and
inference cannot overlap because CWT needs ~396 KB for the signal buffer, and
the TFLite tensor arena needs that same memory region once CWT completes.

```
CSV sensor recording (5 channels × up to 99,000 samples)
        │
        ▼  cwt_mcu.c — CMSIS-DSP, direct time-domain CWT
[Phase 1] CWT scalogram → (5, 64, 64) float32
        │  free signal buffer (~396 KB)
        ▼  TFLite Micro — reuses freed signal buffer as tensor arena
[Phase 2] CNN inference → wear prediction (µm)
```

Estimated timing on Cortex-M33 @ 150 MHz (CPU only):

| Phase | Time |
|---|---|
| CWT (5 channels, ~50K-sample signal) | ~100 ms |
| CNN inference (INT8) | ~15 ms |
| **Total** | **~115 ms** |

With the eIQ Neutron NPU handling CNN inference: ~101 ms total.

---

## Model Selection and Memory Analysis

### Candidate models

Two sensor CNN architectures were trained on (5, 64, 64) CWT scalograms:

| Model | Params | FP32 weights | INT8 weights | Best test MAE |
|---|---|---|---|---|
| SensorCNNRegressor (baseline) | 61 K | 238 KB | ~61 KB | 30.36 µm |
| **MultiScaleSensorCNN** | **244 K** | **955 KB** | **~239 KB** | **~28 µm** |

### Why MultiScaleSensorCNN fits despite larger activations

An initial analysis incorrectly concluded that the MultiScaleSensorCNN could
not fit in 512 KB SRAM. The error was adding model weight size to the SRAM
requirement. Model weights are stored as a `const` array in **flash** and are
never copied to SRAM. Only activations use SRAM (the TFLite tensor arena).

**Corrected SRAM analysis (MultiScaleSensorCNN, INT8):**

The model's inception-style entry block computes three parallel conv paths and
concatenates them:

```
path_1x1 → (1, 16, 64, 64) INT8 = 64 KB  ┐
path_3x3 → (1, 24, 64, 64) INT8 = 96 KB  ├─ must coexist during concat
path_5x5 → (1,  8, 64, 64) INT8 = 32 KB  ┘
concat   → (1, 48, 64, 64) INT8 = 192 KB  (output being written)
──────────────────────────────────────────────────────────────────
Peak activation arena:                   384 KB
TFLite runtime overhead:                 ~25 KB
Quantized input scalogram (1,5,64,64):   ~20 KB
Stack:                                   ~16 KB
────────────────────────────────────────────────
Total SRAM:                              ~445 KB < 512 KB ✓
```

**Flash budget:**

| Component | Flash |
|---|---|
| Firmware + CWT code | ~40 KB |
| CMSIS-DSP IFFT twiddle tables (const) | ~16 KB |
| TFLite Micro runtime | ~60 KB |
| Model weights INT8 (const array) | ~239 KB |
| **Total** | **~355 KB of 2 MB** ✓ |

### Selected model

`sensor/multiscale/checkpoints/phase4_multiscale_sgdm_best_25.pt` —
MultiScaleSensorCNN with CBAM attention, ResBlock, GroupNorm(8), trained with
SGDM (momentum 0.9, weight_decay 5e-3), HuberLoss(δ=20). 244,463 parameters.

> **Note (post-restructure):** the epoch-tagged `_25` checkpoint was pruned during
> the repo reorganisation; the surviving training checkpoint is
> `sensor/multiscale/checkpoints/phase4_multiscale_sgdm_best.pt`. The already-exported
> ONNX/TFLite artifacts under `sensor/deployment/onnx/` (named `..._25_...`) were
> produced from that `_25` snapshot and remain the deployment source of truth.

---

## Toolchain

NXP does not provide a cloud service equivalent to ST's Edge AI Developer Cloud
for Cortex-M33 + Neutron NPU targets. The **eIQ AI Hub** (launched Jan 2026)
is primarily scoped to i.MX 8/9 application processors and the Ara discrete
NPU — not the MCXN947 Neutron NPU. The correct pipeline uses three sequential
**local** tools from the NXP open-source `eiq-onnx2tflite` package:

```
PyTorch (.pt)
    │
    ▼  torch.onnx.export()                 [fusion/deployment/export_onnx.py]
ONNX FP32 (.onnx, opset 13)
    │
    ▼  onnx2quant  (NXP eiq-onnx2tflite)   [pip install from GitHub release]
ONNX INT8 QDQ (.onnx, with QuantizeLinear/DequantizeLinear nodes)
    │
    ▼  onnx2tflite (NXP eiq-onnx2tflite)
TFLite INT8 (.tflite, float32 I/O)
    │
    ▼  neutron-converter (eIQ Toolkit)     [requires NXP local install]
NPU-optimised TFLite + model_data.h
```

Install the open-source package:
```bash
pip install https://github.com/NXP/eiq-onnx2tflite/releases/download/0.10.0/eiq_onnx2tflite-0.10.0-py3-none-any.whl
```

---

## Step 1 — Export to ONNX

**Script:** `fusion/deployment/export_onnx.py`

```bash
python fusion/deployment/export_onnx.py
# Output: sensor/deployment/onnx/phase4_multiscale_sgdm_best_25_op13.onnx
```

### Key decisions

**Opset 13, not 18.**
PyTorch 2.x defaults to opset 18 with the dynamo exporter. Opset 18 moved the
`ReduceMean` axes argument from an attribute to an input tensor. The
`onnx2tflite` converter does not handle this form — it expects INT64 axes as an
attribute. Exporting at opset 13 (using the legacy TorchScript exporter,
`dynamo=False`) keeps `ReduceMean` axes as an attribute and produces a graph
that converts cleanly.

**dynamo=False.**
The dynamo exporter produces a cleaner graph for opset 18 (34 nodes vs 133
nodes) but is incompatible with `onnx2tflite`'s ReduceMean handling at that
opset. The legacy exporter (`dynamo=False`) at opset 13 is compatible
throughout the pipeline.

**GroupNorm decomposition.**
PyTorch's GroupNorm(`G=8, C`) exports at opset 13 as:
```
Reshape(NCHW → N·G, C/G, H, W)
  → InstanceNormalization
  → Reshape back to NCHW
  → Mul(γ) + Add(β)
```
All are standard ONNX ops fully supported by `onnx2quant` and `onnx2tflite`.

**Op types in the exported graph:**

| Op | Count | Source |
|---|---|---|
| Conv | 11 | All conv layers |
| Reshape | 21 | GroupNorm + entry path reshapes |
| InstanceNormalization | 10 | GroupNorm decomposition |
| Mul / Add | 12 / 11 | GroupNorm γ/β + ResBlock skip + CBAM |
| Relu | 11 | Activations |
| Concat | 2 | Inception entry concat + CBAM |
| MaxPool | 3 | Spatial downsampling |
| GlobalAveragePool | 2 | AdaptiveAvgPool2d |
| Gemm | 3 | Final head Linear |
| Sigmoid | 2 | CBAM channel + spatial attention gates |
| ReduceMean | 1 | CBAM channel attention global average |
| ReduceMax | 1 | CBAM spatial attention channel max |
| **Total** | **133** | |

**Numerical validation.**
50 test scalograms were run through both PyTorch and ONNX Runtime:
- Max |PT − ONNX| = **5.3×10⁻⁵ µm** (float32 rounding only)
- MAE delta = **0.00 µm**

### Output

`sensor/deployment/onnx/phase4_multiscale_sgdm_best_25_op13.onnx` — 978 KB,
self-contained (no `.onnx.data` sidecar).

---

## Step 2 — Build Calibration Dataset

**Script:** `fusion/deployment/build_calibration_data.py`

```bash
python fusion/deployment/build_calibration_data.py
# Output: fusion/deployment/calibration_data/  (100 × .npy files)
```

### Why calibration matters

INT8 post-training quantization works by observing activation ranges at each
layer and choosing per-tensor or per-channel scale/zero-point to cover them.
The chosen scale directly determines quantization error: too wide → low
resolution; too narrow → clipping.

If calibration samples all have similar wear values (e.g., all fresh tools),
the quantizer sees a narrow activation range for the worn-tool regime, causing
clipping and accuracy loss on exactly the samples that matter most for
predicting heavy wear.

### Stratified sampling

The training set (647 samples) spans wear 15–750 µm (median 90 µm, skewed
toward lower values). 100 samples are selected by dividing the wear range into
10 equal-count bins and drawing 10 samples from each bin at random (seed=42).

```
Bin  0 ( 15– 30 µm): 10 samples   ← fresh tools
Bin  1 ( 30– 45 µm): 10 samples
Bin  2 ( 45– 60 µm): 10 samples
...
Bin  9 (175–750 µm): 10 samples   ← heavily worn tools
```

This guarantees the full activation range is observed, including the
high-wear regime where predictions are most critical.

**Calibration set summary:** min=15 µm, median=90 µm, max=420 µm, std=93 µm.

**Format:** one `.npy` file per sample, shape `(1, 5, 64, 64)`, float32. The
`onnx2quant` CLI expects `--calibration-dataset-mapping input_name;directory/`.

---

## Step 3 — INT8 Quantization via ONNX2Quant

```bash
python -m onnx2quant \
  sensor/deployment/onnx/phase4_multiscale_sgdm_best_25_op13.onnx \
  -c "scalogram;fusion/deployment/calibration_data/" \
  --per-channel \
  -o sensor/deployment/onnx/phase4_multiscale_sgdm_best_25_op13_int8.onnx
```

### What ONNX2Quant does

ONNX2Quant inserts `QuantizeLinear` and `DequantizeLinear` (QDQ) nodes around
each quantizable operator. Unlike standard ONNX quantization tools, it places
these in the pattern expected by TFLite — making the subsequent `onnx2tflite`
step aware of quantization parameters for each tensor and able to produce a
properly INT8-quantized `.tflite` model rather than a float32 model with
post-export quantization.

`--per-channel` enables per-output-channel quantization for Conv and Gemm
weight tensors. This is more accurate than per-tensor quantization at the cost
of slightly larger scale tables. The Neutron NPU supports per-channel
quantization natively.

### Result

Input FP32 ONNX: 978 KB → Output INT8 QDQ ONNX: **421 KB** (57% size
reduction). 233 QDQ nodes inserted across 133 original nodes.

Accuracy on val and test sets measured by running both FP32 and INT8 ONNX
models through ONNX Runtime on the full splits:

| Split | FP32 MAE | INT8 MAE | Δ MAE | Max per-sample diff |
|---|---|---|---|---|
| Val (300) | 45.70 µm | 45.15 µm | **−0.55 µm** | 9.53 µm |
| Test (247) | 24.96 µm | 24.70 µm | **−0.25 µm** | 9.19 µm |

The negative delta indicates INT8 slightly outperforms FP32 on these splits —
this is within noise (100 calibration samples provide slightly different
statistics than the full training set, acting as mild regularization).
The key result is that quantization does not cause measurable accuracy
degradation.

---

## Step 4 — Convert to TFLite via onnx2tflite

```bash
python -m onnx2tflite \
  sensor/deployment/onnx/phase4_multiscale_sgdm_best_25_op13_int8.onnx \
  --qdq-aware-conversion \
  --keep-io-tensors-format \
  -o sensor/deployment/onnx/phase4_multiscale_sgdm_best_25_int8.tflite
```

### Flags explained

`--qdq-aware-conversion`: Instructs the converter to use the QDQ nodes inserted
by `onnx2quant` to produce a properly INT8-quantized TFLite flatbuffer, rather
than a float32 graph. Without this flag the QDQ nodes would be preserved as
QuantizeLinear/DequantizeLinear ops instead of being folded into TFLite's
native quantization scheme.

`--keep-io-tensors-format`: Preserves the NCHW (channels-first) layout for the
input tensor `(1, 5, 64, 64)`. TFLite normally converts to NHWC internally.
Keeping NCHW means the MCU application feeds the scalogram directly into the
TFLite input tensor without a transpose step — the CWT output is already in
channel-first order `(5, 64, 64)`.

### The ReduceMean / opset-18 issue

The first export attempt used opset 18 (PyTorch 2.x default). ONNX opset 18
moved `ReduceMean`'s axes argument from an attribute to a second input tensor.
`onnx2tflite` 0.10.0 parses the axes as an attribute and fails with:

```
[ERROR] ONNX `ReduceMean` has `axes` of type `INT32`, instead of INT64
```

The fix was to re-export at **opset 13**, where `ReduceMean` retains attribute-
based axes. The opset 13 legacy exporter also avoids symbolic shape nodes
(`Shape`, `Gather`) that can complicate the TFLite conversion of GroupNorm
reshapes.

### Output

`sensor/deployment/onnx/phase4_multiscale_sgdm_best_25_int8.tflite` — **308 KB**,
float32 input/output, NCHW layout.

The warnings about "re-quantizing tensors to match Concat output q-params" are
expected: the three inception paths have slightly different activation
distributions, so their quantization scales differ. `onnx2tflite` re-quantizes
path outputs to a common scale at the Concat op. This is correct TFLite
behavior and does not introduce meaningful accuracy loss.

### Full-chain accuracy validation

FP32 ONNX → INT8 ONNX → TFLite INT8, measured on the full splits using
ai-edge-litert (Google's TFLite Python runtime):

| Split | FP32 MAE | INT8 ONNX MAE | TFLite MAE | TFLite Δ vs FP32 |
|---|---|---|---|---|
| Val (300) | 45.70 µm | 45.15 µm | **45.18 µm** | **−0.52 µm** |
| Test (247) | 24.96 µm | 24.70 µm | **24.74 µm** | **−0.21 µm** |

Max per-sample difference between INT8 ONNX and TFLite: 3.3 µm (rounding in
quantized representation, not systematic error).

**Conclusion:** The TFLite model preserves the FP32 model's accuracy to within
the noise of the evaluation. There is no measurable accuracy cost from
quantization.

---

## Step 5 — Generate C Array for Flash

The validated TFLite model is embedded in the MCUXpresso project as a C
`const` array in flash. `xxd` produces a header that the linker places in
the `.rodata` section — the model bytes are never copied to SRAM.

```bash
cd sensor/deployment/onnx
xxd -i phase4_multiscale_sgdm_best_25_int8.tflite \
    > phase4_multiscale_sgdm_best_25_model_data.h
```

The generated header contains:
```c
unsigned char phase4_multiscale_sgdm_best_25_int8_tflite[] = { 0x1c, 0x00, ... };
unsigned int  phase4_multiscale_sgdm_best_25_int8_tflite_len = 315392;
```

Rename the symbols to something shorter for use in application code:
```bash
sed -i 's/phase4_multiscale_sgdm_best_25_int8_tflite/g_model_data/g' \
    phase4_multiscale_sgdm_best_25_model_data.h
```

Add `phase4_multiscale_sgdm_best_25_model_data.h` to the MCUXpresso project
source tree. The model runs on TFLite Micro using CMSIS-NN-accelerated kernels
on the Cortex-M33 CPU (Conv2D, MaxPool2D, FullyConnected all use SIMD integer
math via CMSIS-NN automatically). Estimated inference time: **~15 ms**.

---

## Step 6 — MCUXpresso Project

The full application source lives in
`fusion/deployment/mcu_project/`.  Four files are complete and
ready to be added to an MCUXpresso managed-build project or compiled via the
provided `CMakeLists.txt`:

| File | Role |
|---|---|
| `main.c` | Board init, UART CSV receiver, CWT + inference orchestration |
| `sensor_pipeline.h` | Plain-C API (pool layout, status codes, all four functions) |
| `sensor_pipeline.cpp` | TFLite Micro implementation — pool carving, 19-op resolver, placement-new interpreter |
| `send_csv_uart.py` | Host script — applies cutting mask, streams CSV to MCU over UART, reads prediction |
| `CMakeLists.txt` | CMake build file (arm-none-eabi-gcc, SDK components, pyOCD/J-Link flash targets) |

### Pool memory layout (as implemented)

The caller provides a single `static uint8_t g_pool[SENSOR_PIPELINE_POOL_BYTES]`
(508 KB). `sensor_pipeline.cpp` carves it into three regions:

```
Offset 0          │ signal buffer     │  396 KB  MAX_SIGNAL_SAMPLES × float32
Offset 396 KB     │ scalogram (5×64²) │   80 KB  persists across both phases
Offset 476 KB     │ CWT workspace     │   16 KB  2 × CWT_MCU_N_KER floats
                  └───────────────────┘  = 492 KB active during CWT

After CWT (signal + workspace freed):
Offset 0          │ TFLite tensor arena│  412 KB  signal_bytes + wksp_bytes
Offset 396 KB     │ scalogram (persists│   80 KB  read by inference pass
```

### TFLite Micro op resolver (19 ops, exact)

The op list was determined by parsing the flatbuffer `operator_codes` table of
`phase4_multiscale_sgdm_best_25_int8.tflite` directly (not inferred from ONNX):

```cpp
// sensor_pipeline.cpp — tflite::MicroMutableOpResolver<19> s_resolver
s_resolver.AddQuantize();          // float32 → INT8 at model entry
s_resolver.AddConv2D();
s_resolver.AddReshape();
s_resolver.AddTranspose();
s_resolver.AddDequantize();        // INT8 → float32 at model exit
s_resolver.AddRsqrt();             // GroupNorm normalisation
s_resolver.AddMean();              // GroupNorm mean / ReduceMean
s_resolver.AddSquaredDifference(); // GroupNorm variance
s_resolver.AddAdd();
s_resolver.AddSub();
s_resolver.AddMul();               // GroupNorm γ scale + attention gates
s_resolver.AddRelu();
s_resolver.AddConcatenation();     // Inception multi-scale concat
s_resolver.AddMaxPool2D();
s_resolver.AddAveragePool2D();
s_resolver.AddFullyConnected();
s_resolver.AddLogistic();          // Sigmoid — attention gates
s_resolver.AddReduceMax();
s_resolver.AddSum();
```

The model input is **float32** (scale=0, zp=0). The `QUANTIZE` op is the first
internal op and converts float32 → INT8. No manual quantization is needed when
feeding the model; `sensor_pipeline_infer` performs a plain `memcpy` from the
scalogram into the input tensor.

### UART protocol

```
MCU  →  "READY\r\n"
Host →  "acc,acoustic,fx,fy,fz\n"   (one row per sample, float values)
Host →  ...
Host →  "END\n"
MCU  →  "Predicted tool wear: XX.X um\r\n"
```

The host script `send_csv_uart.py` applies the same cutting mask as
`cwt_mcu_process_channel` before transmitting, so only the cutting-segment rows
are sent. With the default chunk-size and inter-chunk sleep the script handles
UART back-pressure without overflowing the MCU RX buffer.

### Build flags (MCUXpresso project properties → C/C++ Build → Settings)

```
-DARM_MATH_CM33            # CMSIS-DSP for Cortex-M33
-DTF_LITE_STATIC_MEMORY    # TFLite Micro: no dynamic allocation
-O2 -ffast-math            # FP performance (safe for this pipeline)
```

Do **not** define `DESKTOP_SHIM` — `cwt_mcu.c` then includes the real
`arm_math.h` and links against the native CMSIS-DSP library.

### MCUXpresso IDE setup checklist

1. **New C/C++ Project** → SDK-based → FRDM-MCXN947, Cortex-M33 core 0
2. **Add SDK components** in the component selector:
   - *CMSIS DSP Library* (arm_math.h, CFFT, FIR, ...)
   - *eIQ — TensorFlow Lite Micro* (micro_interpreter, op kernels)
   - *LPUART driver* + *Debug Console*
3. **Copy source files** into the project:
   - `main.c`, `sensor_pipeline.h`, `sensor_pipeline.cpp`
   - `cwt_mcu.c`, `cwt_mcu.h`  (from `sensor/deployment/cwt_c/`)
   - `phase4_multiscale_sgdm_best_25_model_data.h`  (from `sensor/deployment/onnx/`)
4. **Set build flags** (above) in project Properties → C/C++ Build → Settings
5. **Build** → should produce `tool_wear.axf` under `Debug/` or `Release/`
6. **Flash** via *Run → Debug* (J-Link on-board) or `pyocd flash -t mcxn947`

### End-to-end validation

```bash
# On host machine — connect FRDM-MCXN947 via USB
python fusion/deployment/mcu_project/send_csv_uart.py \
    --port /dev/tty.usbmodemXXXX \
    --csv  data/raw/Set4/sensordata/<recording>.csv \
    --baud 115200
```

Expected output (example):
```
Cutting segment: 12453 samples (7.7 s at 1625 Hz)
Opening /dev/tty.usbmodemXXXX at 115200 baud...
Waiting for MCU READY...
  MCU: === Tool Wear Prediction Pipeline ===
  MCU: READY
  Sent 12453 rows + END
Waiting for prediction (timeout 120 s)...
  MCU: Received 12453 samples
  MCU: CWT channel 1/5...
  MCU: CWT channel 2/5...  ...
  MCU: Initialising TFLite Micro...
  MCU: [pipeline] Arena used: 384 KB / 412 KB
  MCU: Running inference...
  MCU: ========================================
  MCU: Predicted tool wear: 127.3 um
  MCU: ========================================

=== Result ===
Predicted tool wear: 127.3 um
```

---

## Accuracy Results

Full accuracy chain measured on the MATWI test split (247 samples):

| Stage | Test MAE | Δ vs PyTorch FP32 |
|---|---|---|
| PyTorch FP32 (reference) | 24.96 µm | — |
| ONNX FP32 | 24.96 µm | 0.00 µm |
| ONNX INT8 (ONNX2Quant) | 24.70 µm | −0.25 µm |
| TFLite INT8 (onnx2tflite) | 24.74 µm | **−0.21 µm** |

The quantization pipeline introduces no measurable accuracy degradation.
The slight improvement (−0.21 µm) is within evaluation noise.

### TFLite INT8 per-sample validation

`fusion/deployment/validate_tflite.py` runs both models on the
same inputs and compares predictions sample-by-sample. Results on the test
split (247 samples):

| Metric | Value | Threshold | Status |
|---|---|---|---|
| PyTorch FP32 MAE | 24.96 µm | — | — |
| TFLite INT8 MAE | 24.74 µm | — | — |
| MAE delta (TFLite − PyTorch) | 0.21 µm | < 5 µm | ✅ PASS |
| Max per-sample diff | 10.29 µm | < 20 µm | ✅ PASS |
| p95 per-sample diff | 6.00 µm | — | — |
| Systematic bias (mean diff) | −3.26 µm | — | — |
| Prediction correlation | 0.9983 | — | — |

The TFLite model has a small systematic bias of −3.3 µm (slight underestimation
of wear). This is a known effect of symmetric INT8 quantization on regression
outputs and is well within the 5 µm acceptance threshold. The 0.9983
correlation confirms the ranking of samples is virtually identical to FP32.

Results saved to `fusion/deployment/results/validate_tflite_results.json`.

**Model size progression:**

| Format | Size |
|---|---|
| PyTorch .pt | 955 KB (includes optimizer state) |
| ONNX FP32 | 978 KB |
| ONNX INT8 QDQ | 421 KB |
| TFLite INT8 | **308 KB** |
| Expected after Neutron Converter | ~280–320 KB |

---

## Memory Budget

### SRAM

```
                ┌──────────────── 512 KB SRAM ─────────────────┐

Phase 1 (CWT)  │ signal (1 ch)   │ scalogram  │ wksp  │ stack │
               │    396 KB       │   80 KB    │ 16 KB │ 16 KB │
               └─────────────────┴────────────┴───────┴───────┘
               Peak: ~508 KB ✓

               ← free signal + workspace (412 KB) ──────────────

Phase 2        │ tensor arena    │ scalogram  │  rt   │ stack │
(Inference)    │   ≤412 KB       │   80 KB    │ 25 KB │ 16 KB │
               │ (reused region) │ (persists) │       │       │
               └─────────────────┴────────────┴───────┴───────┘
               Peak: ~445 KB ✓  (arena peak: ~384 KB at inception concat)
```

### Flash

| Component | Flash |
|---|---|
| Firmware + application code | ~40 KB |
| CWT code (`cwt_mcu.c`) | ~16 KB |
| CMSIS-DSP twiddle tables (const) | ~16 KB |
| TFLite Micro runtime | ~60 KB |
| Model weights (INT8, const) | ~239 KB |
| **Total** | **~371 KB of 2 MB** ✓ |

---

## Step 7 — NPU Acceleration (future)

The current deployment runs inference on the Cortex-M33 CPU using CMSIS-NN
integer kernels (~15 ms). The FRDM-MCXN947's eIQ Neutron NPU can reduce this
to under 1 ms, but the Neutron Converter tool is only available for
Windows/Linux (not macOS) as part of the NXP eIQ Toolkit desktop application.

This step should be completed on a Windows/Linux machine or in a Linux Docker
container before final deployment.

### When to do this

After the CPU-based pipeline is validated end-to-end on the board (data flows
correctly, predictions are reasonable), swap in the NPU-compiled model to
reduce inference latency from ~15 ms to <1 ms, bringing total pipeline time
below 102 ms.

### How to do it

**Option A — Docker on macOS** (if Docker Desktop is installed):

```bash
docker run -it --rm \
  -v "$(pwd)/sensor/deployment/onnx:/models" \
  ubuntu:22.04 bash

# Inside container — download eIQ Toolkit Linux installer from
# https://www.nxp.com/design/design-center/software/eiq-ai-development-environment/eiq-toolkit-for-end-to-end-model-development-and-deployment:EIQ-TOOLKIT
# (free NXP account required)
./eiq-toolkit-installer.run --cli-only
neutron-converter \
  --input  /models/phase4_multiscale_sgdm_best_25_int8.tflite \
  --output /models/phase4_multiscale_sgdm_best_25_npu.tflite \
  --target imxrt700 \
  --dump-header-file-output
```

**Option B — GitHub Actions Linux runner**:

```yaml
# .github/workflows/neutron_convert.yml
jobs:
  convert:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - name: Install eIQ Toolkit CLI
        run: |
          wget -q "${{ secrets.EIQ_TOOLKIT_LINUX_URL }}" -O installer.run
          chmod +x installer.run && ./installer.run --cli-only
      - name: Run Neutron Converter
        run: |
          neutron-converter \
            --input  sensor/deployment/onnx/phase4_multiscale_sgdm_best_25_int8.tflite \
            --output sensor/deployment/onnx/phase4_multiscale_sgdm_best_25_npu.tflite \
            --target imxrt700 --dump-header-file-output
      - uses: actions/upload-artifact@v4
        with:
          name: npu-model
          path: sensor/deployment/onnx/phase4_multiscale_sgdm_best_25_npu.*
```

**Option C — Native Windows/Linux machine**: run the commands from the
original Step 5 section directly.

### After conversion

Replace the model header in the MCUXpresso project:

```bash
# Rename symbols in the NPU-compiled header
sed -i 's/phase4_multiscale_sgdm_best_25_npu_tflite/g_model_data/g' \
    phase4_multiscale_sgdm_best_25_npu.h
```

Swap `phase4_multiscale_sgdm_best_25_model_data.h` for
`phase4_multiscale_sgdm_best_25_npu.h` in the MCUXpresso project — no other
code changes are needed. The TFLite Micro API is identical; the NPU kernels
are dispatched transparently.

### Expected NPU coverage for MultiScaleSensorCNN

| Op | NPU | Notes |
|---|---|---|
| Conv2D (INT8 per-channel) | ✅ Full | Dominates compute (~95% of FLOPs) |
| MaxPool2D | ✅ Full | |
| GlobalAveragePool2D | ✅ Full | |
| FullyConnected (Gemm) | ✅ Full | |
| ReLU | ✅ Full (fused into Conv) | |
| Concat | ⚠️ CPU | Memory reshape, negligible cost |
| GroupNorm (Reshape + InstanceNorm + Mul + Add) | ⚠️ CPU | Fast on small tensors |
| Sigmoid, ReduceMax (CBAM) | ⚠️ CPU | Fast on small tensors |

The Conv2D layers dominate. NPU acceleration of those alone brings inference
from ~15 ms to under 1 ms.

---

## GroupNorm Compatibility

`MultiScaleSensorCNN` uses `nn.GroupNorm(8, C)` after each convolution.
TFLite has no native GroupNorm op. PyTorch's ONNX exporter at opset 13
automatically decomposes it into standard ops:

```
Reshape  →  InstanceNormalization  →  Reshape  →  Mul(γ)  →  Add(β)
```

`onnx2tflite` maps these to primitive TFLite ops (Reshape, Mean, Sub, Mul,
Add) which CMSIS-NN can execute. The Neutron NPU does not accelerate
InstanceNormalization, so GroupNorm falls back to the Cortex-M33 CPU — but
GroupNorm ops are fast (small tensors after pooling) and contribute <1 ms to
total inference time.

An alternative would be retraining with `nn.BatchNorm2d`. At inference time
BatchNorm uses fixed running statistics and folds into the preceding conv
weights (zero runtime cost; natively supported by TFLite and the NPU).
Training used GroupNorm because BatchNorm degrades accuracy under SGDM at
small batch sizes (Phase 4 experiment 12: BN=40.45 µm vs GN=30.36 µm).
If GroupNorm causes unexpected issues on the MCU, retraining with BN for
the deployment branch is the cleanest fix.

---

## File Inventory

| File | Purpose | Status |
|---|---|---|
| `sensor/deployment/cwt_c/cwt_mcu.c` | CWT preprocessing (CMSIS-DSP) | ✅ Complete, validated 50/50 |
| `sensor/deployment/cwt_c/cwt_mcu.h` | CWT public API | ✅ Complete |
| `sensor/deployment/cwt_c/cmsis_shim.h` | Desktop compatibility layer | ✅ Complete |
| `sensor/deployment/cwt_c/main_mcu.c` | Desktop CLI wrapper for CWT | ✅ Complete |
| `sensor/deployment/cwt_c/validate_mcu.py` | CWT validation vs Python reference | ✅ 50/50 pass |
| `fusion/deployment/export_onnx.py` | PyTorch → ONNX export | ✅ Complete |
| `fusion/deployment/build_calibration_data.py` | Calibration dataset (100 samples) | ✅ Complete |
| `fusion/deployment/validate_tflite.py` | Per-sample TFLite vs PyTorch validation | ✅ PASS (MAE delta 0.21 µm) |
| `sensor/deployment/onnx/phase4_multiscale_sgdm_best_25_op13.onnx` | FP32 ONNX (opset 13) | ✅ 978 KB |
| `sensor/deployment/onnx/phase4_multiscale_sgdm_best_25_op13_int8.onnx` | INT8 QDQ ONNX | ✅ 421 KB |
| `sensor/deployment/onnx/phase4_multiscale_sgdm_best_25_int8.tflite` | TFLite INT8 | ✅ 308 KB |
| `sensor/deployment/onnx/phase4_multiscale_sgdm_best_25_model_data.h` | Model as C array for flash (CPU) | ✅ Generated via xxd |
| `fusion/deployment/mcu_project/sensor_pipeline.h` | CWT + TFLite Micro pipeline — plain-C API | ✅ Complete |
| `fusion/deployment/mcu_project/sensor_pipeline.cpp` | Pipeline implementation — 19-op resolver, placement-new | ✅ Complete |
| `fusion/deployment/mcu_project/main.c` | FRDM-MCXN947 entry point — UART CSV receive + orchestration | ✅ Complete |
| `fusion/deployment/mcu_project/send_csv_uart.py` | Host script — cutting mask + UART streaming | ✅ Complete |
| `fusion/deployment/mcu_project/CMakeLists.txt` | CMake build (arm-none-eabi-gcc, pyOCD/J-Link targets) | ✅ Complete |
| `sensor/deployment/onnx/phase4_multiscale_sgdm_best_25_npu.tflite` | NPU-compiled TFLite | ⏳ Step 7 — requires eIQ Toolkit on Linux/Windows |
| `sensor/deployment/onnx/phase4_multiscale_sgdm_best_25_npu.h` | NPU model as C array | ⏳ Step 7 — generated by Neutron Converter |
