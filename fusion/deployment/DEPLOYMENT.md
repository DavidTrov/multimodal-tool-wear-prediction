# On-Device Deployment Guide — FRDM-MCXN947
## Tool Wear Prediction: Fusion, Sensor-Only, and Image-Only Models

**Target hardware:** NXP FRDM-MCXN947 (Cortex-M33 @ 150 MHz, 512 KB SRAM, 2 MB flash)  
**SDK:** MCUXpresso SDK v2.x with eIQ middleware  
**Runtime:** TensorFlow Lite for Microcontrollers (TFLM)  
**Date:** May 2026

---

## Table of Contents

1. [Hardware Overview](#1-hardware-overview)
2. [Software Stack](#2-software-stack)
3. [Model Conversion Pipeline](#3-model-conversion-pipeline)
   - 3.1 [PyTorch → ONNX](#31-pytorch--onnx)
   - 3.2 [ONNX axes deduplication](#32-onnx-axes-deduplication)
   - 3.3 [ONNX → TFLite (onnx2tf)](#33-onnx--tflite-onnx2tf)
   - 3.4 [Choosing the right quantisation variant](#34-choosing-the-right-quantisation-variant)
   - 3.5 [TFLite → C header (xxd)](#35-tflite--c-header-xxd)
4. [Memory Budget and Layout](#4-memory-budget-and-layout)
   - 4.1 [MCXN947 memory map](#41-mcxn947-memory-map)
   - 4.2 [Fusion model budget](#42-fusion-model-budget)
   - 4.3 [Sensor-only model budget](#43-sensor-only-model-budget)
   - 4.4 [Image-only model budget](#44-image-only-model-budget)
5. [MCUXpresso IDE Setup](#5-mcuxpresso-ide-setup)
   - 5.1 [Creating the project](#51-creating-the-project)
   - 5.2 [Adding TFLM and CMSIS-DSP](#52-adding-tflm-and-cmsis-dsp)
   - 5.3 [Preprocessor defines and compiler flags](#53-preprocessor-defines-and-compiler-flags)
   - 5.4 [Disabling managed linker scripts](#54-disabling-managed-linker-scripts)
6. [Linker Script Modifications](#6-linker-script-modifications)
   - 6.1 [Why the default scripts fail](#61-why-the-default-scripts-fail)
   - 6.2 [Expanding PROGRAM_FLASH0 to 2 MB](#62-expanding-program_flash0-to-2-mb)
   - 6.3 [Expanding SRAM to 416 KB (merging SRAMH)](#63-expanding-sram-to-416-kb-merging-sramh)
   - 6.4 [Reducing heap size](#64-reducing-heap-size)
   - 6.5 [Final memory.ld](#65-final-memoryld)
7. [Firmware Source Files](#7-firmware-source-files)
   - 7.1 [sensor_pipeline.h](#71-sensor_pipelineh)
   - 7.2 [sensor_pipeline.cpp](#72-sensor_pipelinecpp)
   - 7.3 [main.c](#73-mainc)
   - 7.4 [CWT: cwt_mcu.h / cwt_mcu.c](#74-cwt-cwt_mcuhcwt_mcuc)
8. [UART Protocol and Host Script](#8-uart-protocol-and-host-script)
9. [Building and Flashing](#9-building-and-flashing)
   - 9.1 [Build](#91-build)
   - 9.2 [Flashing with the GUI Flash Tool](#92-flashing-with-the-gui-flash-tool)
   - 9.3 [Debug configuration flash limit](#93-debug-configuration-flash-limit)
   - 9.4 [Serial terminal](#94-serial-terminal)
10. [Known Issues and Workarounds](#10-known-issues-and-workarounds)
11. [Adapting for Sensor-Only and Image-Only Models](#11-adapting-for-sensor-only-and-image-only-models)
12. [End-to-End Validation](#12-end-to-end-validation)

---

## 1. Hardware Overview

| Feature | Value |
|---|---|
| SoC | NXP MCXN947 |
| CPU | Dual Cortex-M33 @ 150 MHz (only core0 used here) |
| Flash | 2 MB contiguous at 0x0000_0000–0x001F_FFFF |
| SRAM | 512 KB total; SRAM0–7 contiguous at 0x2000_0000 |
| SRAMX | 96 KB "code RAM" at 0x0400_0000 |
| Debug I/F | CMSIS-DAP via on-board LPC55S69 |
| UART | LPUART0 → USB virtual COM (115200 baud default) |

**Physical SRAM layout relevant to this project:**

```
0x20000000  ┌──────────────────────────────────────┐
            │  SRAM0–6 : 384 KB  (original "SRAM")  │
0x20060000  ├──────────────────────────────────────┤
            │  SRAMH   :  32 KB  (contiguous)       │ ← merged into SRAM
0x20068000  └──────────────────────────────────────┘
            Total: 416 KB physically contiguous

0x04000000  ┌──────────────────────────────────────┐
            │  SRAMX   :  96 KB                     │ ← scalogram lives here
0x04018000  └──────────────────────────────────────┘
```

The SRAMH region is physically contiguous with SRAM (no gap between 0x2005_FFFF and 0x2006_0000) so it can be merged into a single 416 KB SRAM region in the linker script.

---

## 2. Software Stack

| Layer | Component | Notes |
|---|---|---|
| IDE | MCUXpresso IDE v25.6 | Eclipse-based; manages build + flash |
| SDK | MCUXpresso SDK v2.x for MCXN947 | Board support, clock config, LPUART driver |
| ML runtime | TensorFlow Lite for Microcontrollers (TFLM) | Bundled as `tflm_lib` prebuilt static library in the eIQ middleware |
| DSP | CMSIS-DSP → replaced by kiss_fft | `DESKTOP_SHIM` macro enables the substitution |
| Conversion | onnx2tf | Python: `pip install onnx2tf` |
| Conversion | torch.onnx | Bundled with PyTorch |
| Host script | pyserial, Pillow, numpy, pandas | Python host-side UART driver |

### TFLM in MCUXpresso

The eIQ middleware ships a prebuilt `tflm_lib.a` (or similar) that already includes:
- TFLM interpreter, allocator, tensor arena
- All standard builtin ops
- **kiss_fft** (used when `DESKTOP_SHIM` is defined — replaces CMSIS-DSP FFT)

Because kiss_fft is already inside `tflm_lib`, **do not add `kiss_fft.c` to your project**. Adding it causes duplicate symbol errors. The include path for `kiss_fft.h` is provided by the eIQ SDK include paths (`-I.../third_party/kissfft`), so `#include "kiss_fft.h"` (without subdirectory prefix) works directly.

---

## 3. Model Conversion Pipeline

### 3.1 PyTorch → ONNX

The fusion model (distilled ResNet image encoder + MultiScaleSensorCNN scalogram encoder) is exported from PyTorch to ONNX opset 18:

```python
# experiments/phase6_deployment/export_onnx.py  (or convert_tflite.py Step 1)

class Wrapper(torch.nn.Module):
    def forward(self, image, scalogram):
        out = self.m(image, scalogram)
        return out[0] if isinstance(out, tuple) else out

wrapper = Wrapper(model)
wrapper.eval()

torch.onnx.export(
    wrapper, (img_dummy, scal_dummy),
    "checkpoints/fusion_fp32.onnx",
    opset_version=18,
    input_names=["image", "scalogram"],
    output_names=["prediction"],
    dynamic_axes={"image": {0: "batch"}, "scalogram": {0: "batch"}, "prediction": {0: "batch"}},
)
```

For the sensor-only model (`MultiScaleSensorCNN`) the export is simpler — single input `scalogram [1, 5, 64, 64]`. For the image-only ResNet variant, single input `image [1, 3, 224, 224]`.

**Verify the export numerically before proceeding:**

```python
sess = onnxruntime.InferenceSession("checkpoints/fusion_fp32.onnx")
onnx_out = sess.run(None, {"image": img_np, "scalogram": scal_np})[0]
assert abs(onnx_out - pt_out) < 1e-3, "ONNX/PyTorch mismatch"
```

### 3.2 ONNX axes deduplication

PyTorch's ONNX exporter (opset 18) sometimes emits `ReduceMean` / `ReduceMax` nodes that share a single `axes` initializer tensor. The `onnx2tf` converter (and NXP's eiq-onnx2tflite) misparsed these shared axes tensors as INT32, producing broken models. The fix is to deduplicate them so each Reduce node owns its own axes tensor:

```python
# experiments/compression/cwt/phase4_static_quant/convert_tflite.py — Step 2
for node in model.graph.node:
    if node.op_type in ("ReduceMean", "ReduceMax") and len(node.input) >= 2:
        axes_name = node.input[1]
        if axes_name in used_names:
            # clone the initializer under a new name and redirect the node
            new_init = numpy_helper.from_array(arr, name=f"{axes_name}_{node.name}")
            model.graph.initializer.append(new_init)
            node.input[1] = new_init.name
        else:
            used_names.add(axes_name)
```

This step is only needed if the model contains `ReduceMean` or `ReduceMax` with shared axes — inspect `onnx.load(...)` if unsure.

### 3.3 ONNX → TFLite (onnx2tf)

```bash
pip install onnx2tf tensorflow
```

```python
# experiments/compression/cwt/phase4_static_quant/convert_tflite.py — Step 4
import onnx2tf

onnx2tf.convert(
    input_onnx_file_path="checkpoints/fusion_fp32_dedup.onnx",
    output_folder_path="checkpoints/fusion_tflite",
    non_verbose=True,
    output_integer_quantized_tflite=True,
    quant_type="per-channel",
    custom_input_op_name_np_data_path=[
        ["image",     "checkpoints/calib_data/calib_images.npy"],
        ["scalogram", "checkpoints/calib_data/calib_scalograms.npy"],
    ],
)
```

**Calibration data** is a numpy `.npy` file of representative inputs (50–100 samples in NHWC layout for images, NHWC for scalograms). `onnx2tf` uses it to determine per-channel INT8 scale/zero_point for each activation tensor.

> **Data layout note:** PyTorch uses NCHW internally, but onnx2tf converts the graph to NHWC (TFLite's default). The output TFLite model expects:
> - `image`:     `[1, 224, 224, 3]` INT8 NHWC
> - `scalogram`: `[1,  64,  64, 5]` INT8 NHWC
>
> The host script and MCU firmware both apply the NCHW→NHWC transpose explicitly.

### 3.4 Choosing the right quantisation variant

`onnx2tf` produces several TFLite files. Only one is suitable for TFLM on a microcontroller:

| File suffix | Weights | I/O tensors | Peak SRAM | Use for MCU? |
|---|---|---|---|---|
| `_float32.tflite` | FP32 | FP32 | ~600 KB | ✗ too large |
| `_float16.tflite` | FP16 | FP32 | ~600 KB | ✗ no FP16 hardware |
| `_integer_quant.tflite` | INT8 | **FP32** | ~370 KB | ✗ I/O conversion overhead |
| **`_full_integer_quant.tflite`** | **INT8** | **INT8** | **~370 KB** | **✓ use this one** |

**Use `_full_integer_quant.tflite`**: INT8 weights AND INT8 I/O tensors. The image input tensor is 147 KB as INT8 vs 602 KB as FP32 — essential to fit in the tensor arena.

Inspect input quantisation params after conversion (needed for the host script):

```python
import tensorflow as tf
interp = tf.lite.Interpreter("checkpoints/fusion_tflite/fusion_fp32_dedup_full_integer_quant.tflite")
interp.allocate_tensors()
for d in interp.get_input_details():
    print(d["index"], d["name"], d["shape"], d["dtype"], d["quantization"])
# Expected output:
# 0  image      [1 224 224 3]  int8   (scale=0.0078125, zero_point=0)
# 1  scalogram  [1  64  64 5]  int8   (scale=...,       zero_point=...)
```

### 3.5 TFLite → C header (xxd)

```bash
# From the thesis root:
xxd -i "checkpoints/fusion_tflite/fusion_fp32_dedup_full_integer_quant.tflite" \
    > experiments/phase6_deployment/mcu_project/fusion_model_data.h

# Rename identifiers to match the firmware:
sed -i '' \
  's/unsigned char checkpoints_fusion_tflite_[a-z_]*/const uint8_t g_model_data[] __attribute__((aligned(4))) =/' \
  experiments/phase6_deployment/mcu_project/fusion_model_data.h

sed -i '' \
  's/unsigned int checkpoints_fusion_tflite_[a-z_]*/const uint32_t g_model_data_len =/' \
  experiments/phase6_deployment/mcu_project/fusion_model_data.h
```

The resulting header declares `g_model_data[]` (1,259,584 bytes = 1.2 MB for the fusion model) and `g_model_data_len`. It must be added to the MCUXpresso project source directory and included in `main.c` as `#include "fusion_model_data.h"`.

> **Alignment note:** The `__attribute__((aligned(4)))` ensures 4-byte alignment of the model flatbuffer, which TFLM requires. Without it you may get a bus fault on the first `GetModel()` call.

---

## 4. Memory Budget and Layout

### 4.1 MCXN947 memory map

```
Address         Region          Size    Purpose
──────────────────────────────────────────────────────────────────────
0x00000000      PROGRAM_FLASH0  2 MB    Code + .rodata + model weights
0x04000000      SRAMX           96 KB   Scalogram buffer (CWT output)
0x20000000      SRAM (merged)   416 KB  Pool: signal buf + TFLite arena
                                        + stack + heap
```

### 4.2 Fusion model budget

The fusion model (`fusion_fp32_dedup_full_integer_quant.tflite`) has a compressed ResNet image encoder (conv1: 13 output channels, not 64) and a MultiScaleSensorCNN scalogram encoder.

**Flash:**

| Component | Size |
|---|---|
| Code + SDK BSP | ~176 KB |
| `g_model_data` (1.2 MB TFLite flatbuffer) | 1,230 KB |
| **Total** | **~1,406 KB / 2,048 KB (69%)** ✓ |

**SRAM (416 KB):**

| Allocation | Size | Location |
|---|---|---|
| `g_pool`: signal buffer (96,000 × 4 B) | 375 KB | main SRAM (BSS) |
| `g_pool`: CWT workspace (2 × 2048 × 4 B) | 16 KB | main SRAM (BSS) |
| `g_pool` total (= tensor arena after CWT) | **391 KB** | main SRAM |
| Other BSS + data | ~2 KB | main SRAM |
| Heap | 4 KB | main SRAM |
| Stack | 8 KB | main SRAM (top) |
| **main SRAM total** | **~405 KB / 416 KB (97%)** ✓ | |
| `g_scalogram` (5 × 64 × 64 × 4 B) | **80 KB** | **SRAMX** |

**TFLite tensor arena peak:** ~370 KB (fits within the 391 KB pool with 21 KB margin).

### 4.3 Sensor-only model budget

The sensor-only `MultiScaleSensorCNN` (~244K parameters, INT8) is much smaller:

| Component | Size |
|---|---|
| Model flatbuffer (INT8) | ~239 KB |
| Code + BSP | ~100 KB |
| **Total flash** | **~339 KB / 2,048 KB** ✓ |

**SRAM:**  
Arena peak at the inception entry concat: ~445 KB at FP32. With INT8 full-integer quant, peak activations ≈ 3× smaller → ~150 KB arena. Fits comfortably in the original 384 KB SRAM (no SRAMH merge needed).

- No SRAMX usage required.
- `g_pool` can be reduced (e.g., 256 KB arena is sufficient).
- `g_scalogram` may remain in main SRAM.

### 4.4 Image-only model budget

The compressed ResNet image encoder (conv1: 13 channels) without the scalogram branch:

| Component | Size |
|---|---|
| Model flatbuffer (INT8) | ~400–600 KB |
| Code + BSP | ~100 KB |
| **Total flash** | **~500–700 KB** ✓ |

**SRAM:**  
Peak activation at conv1 output: 13 × 112 × 112 = 159 KB (INT8). No scalogram buffer needed. Arena of ~200 KB is sufficient. Fits in 384 KB SRAM without any SRAM expansion.

No changes to the linker script are required for the sensor-only or image-only models — the default 1 MB flash / 384 KB SRAM limits are sufficient.

---

## 5. MCUXpresso IDE Setup

### 5.1 Creating the project

Start from an existing NXP SDK example project for the MCXN947 that already has TFLM/eIQ configured (e.g., the **cifar10** or **label_image** example from the SDK). This ensures:
- The correct startup file (`startup_mcxn947_cm33_core0.c` / `.s`) is present
- The correct linker script templates exist
- TFLM library is linked
- Board support headers (`board.h`, `clock_config.h`, `pin_mux.h`) are set up

**Do not start from scratch** — setting up TFLM manually is error-prone and the eIQ SDK examples have the library paths, op resolvers, and startup code pre-configured.

### 5.2 Adding TFLM and CMSIS-DSP

The eIQ SDK example already links `tflm_lib`. For the CWT module (`cwt_mcu.c`), which requires FFT:

**Option A — DESKTOP_SHIM (recommended):** Define `DESKTOP_SHIM` so `cwt_mcu.c` uses kiss_fft instead of CMSIS-DSP. Since `tflm_lib` already bundles kiss_fft, **no additional source files are needed**. The `cmsis_shim.h` header replaces the CMSIS FFT API with kiss_fft calls:

```c
// In cmsis_shim.h — use bare filename, not subdirectory path
#include "kiss_fft.h"    // ← correct: tflm_lib adds -I.../third_party/kissfft
// NOT: #include "kiss_fft/kiss_fft.h"   ← wrong, causes "file not found"
```

**Option B — native CMSIS-DSP:** Add the CMSIS-DSP library (comes with the MCUXpresso SDK under `CMSIS/DSP/`) and define `ARM_MATH_CM33` instead of `DESKTOP_SHIM`. Produces faster FFT on hardware but requires more setup.

### 5.3 Preprocessor defines and compiler flags

In **Project Properties → C/C++ Build → Settings → MCU C Compiler → Preprocessor** (and the C++ equivalent):

```
ARM_MATH_CM33
DESKTOP_SHIM
TF_LITE_STATIC_MEMORY
CPU_MCXN947VDF_cm33_core0
```

Compiler optimisation flags:
```
-O2 -ffast-math
```

> `TF_LITE_STATIC_MEMORY` disables TFLM's dynamic allocation paths — essential for microcontrollers.

### 5.4 Disabling managed linker scripts

**This is the most critical IDE step for the fusion model.**

By default, MCUXpresso regenerates `Debug/*.ld` from internal device templates on every build, overwriting any manual edits. To prevent this:

Open `.cproject` in a text editor and set `value="false"` for **both** linker entries in the Debug configuration (there are separate entries for the C and C++ linker):

```xml
<!-- C++ linker (around line 324): -->
<option id="com.crt.advproject.link.cpp.manage.XXXXXXXXX"
        name="Manage linker script"
        superClass="com.crt.advproject.link.cpp.manage"
        value="false"   ← change from "true"
        valueType="boolean"/>

<!-- C linker (around line 381): -->
<option id="com.crt.advproject.link.manage.XXXXXXXXX"
        name="Manage linker script"
        superClass="com.crt.advproject.link.manage"
        value="false"   ← change from "true"
        valueType="boolean"/>
```

Both entries must be set to `false`. If only the C++ linker is disabled, the C linker will still regenerate the script.

After editing `.cproject`, **close and reopen the project in MCUXpresso** to force a reload. Then do a **Project → Clean → Build** (not just Build).

> **Important:** The `<memoryInstance>` XML elements in `.cproject` control IDE display and debugger awareness only — they do **not** control what the linker script generates. Changing flash/SRAM sizes there has no effect on the build.

---

## 6. Linker Script Modifications

The project uses a two-file linker script structure generated by MCUXpresso:

- `Debug/<project>_Debug_memory.ld` — memory region definitions (edit this)
- `Debug/<project>_Debug.ld` — section placement (usually unchanged)

### 6.1 Why the default scripts fail

The default `memory.ld` for MCXN947 defines:
```
PROGRAM_FLASH0 : ORIGIN = 0x0,          LENGTH = 0x100000   /* 1 MB  */
SRAM           : ORIGIN = 0x20000000,   LENGTH = 0x60000    /* 384 KB */
```

The fusion model requires:
- **Flash:** 1.4 MB (model flatbuffer 1.2 MB + code 176 KB) → overflows 1 MB by ~400 KB
- **SRAM:** 405 KB (pool + scalogram if in SRAM) → overflows 384 KB

### 6.2 Expanding PROGRAM_FLASH0 to 2 MB

The MCXN947 has 2 MB contiguous flash at 0x0–0x1FFFFF. Simply increase the length:

```ld
PROGRAM_FLASH0 (rx) : ORIGIN = 0x0, LENGTH = 0x200000   /* 2M bytes */
```

The `PROGRAM_FLASH1` region (originally at 0x100000 for the second MB) now overlaps with PROGRAM_FLASH0. This is harmless as long as no code explicitly targets `.text.$Flash2` sections — those sections remain empty in typical projects.

### 6.3 Expanding SRAM to 416 KB (merging SRAMH)

SRAMH (32 KB, originally at 0x20060000) is physically contiguous with SRAM. Merge it by extending the SRAM length and zeroing SRAMH:

```ld
SRAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 0x68000   /* 416K bytes — merged SRAM+SRAMH */
SRAMH (rwx) : ORIGIN = 0x20060000, LENGTH = 0x0       /* 0 bytes — absorbed into SRAM */
```

Update all `__top_*` symbols accordingly (they are used by the startup code to initialise BSS):

```ld
__top_SRAM    = 0x20000000 + 0x68000;
__top_RAM     = 0x20000000 + 0x68000;
__top_SRAMH   = 0x20060000 + 0x0;
__top_RAM3    = 0x20060000 + 0x0;
```

### 6.4 Reducing heap size

The default heap is 32 KB (`0x8000`). For a UART-only pipeline with no `malloc` calls, 4 KB (`0x1000`) is sufficient. Reduce it in `.cproject` (the heap size **is** read from the project settings by the managed linker — reducing it here saves the exact bytes without touching `Debug.ld`):

```xml
<option ... name="Heap size" value="0x1000" .../>
```

This saves 28 KB of SRAM.

### 6.5 Final memory.ld

```ld
/*
 * Manually configured memory layout for FRDM-MCXN947 fusion model deployment.
 * DO NOT regenerate via MCUXpresso IDE — "Manage linker script" is disabled.
 *
 * Changes from default:
 *   PROGRAM_FLASH0: 1 MB → 2 MB  (chip has 2 MB contiguous flash)
 *   SRAM:         384 KB → 416 KB  (absorbs SRAMH, physically contiguous)
 *   SRAMH:         32 KB → 0 KB   (covered by expanded SRAM)
 */

MEMORY
{
  PROGRAM_FLASH0 (rx) : ORIGIN = 0x0,          LENGTH = 0x200000  /* 2M bytes */
  PROGRAM_FLASH1 (rx) : ORIGIN = 0x100000,     LENGTH = 0x100000  /* overlaps FLASH0; compat only */
  SRAM (rwx)          : ORIGIN = 0x20000000,   LENGTH = 0x68000   /* 416K bytes */
  SRAMX (rwx)         : ORIGIN = 0x4000000,    LENGTH = 0x18000   /* 96K bytes */
  SRAMH (rwx)         : ORIGIN = 0x20060000,   LENGTH = 0x0       /* 0 bytes — absorbed */
  USB_RAM (rwx)       : ORIGIN = 0x400ba000,   LENGTH = 0x1000    /* 4K bytes */
}

__base_PROGRAM_FLASH0 = 0x0;
__base_Flash          = 0x0;
__top_PROGRAM_FLASH0  = 0x0 + 0x200000;
__top_Flash           = 0x0 + 0x200000;

__base_PROGRAM_FLASH1 = 0x100000;
__base_Flash2         = 0x100000;
__top_PROGRAM_FLASH1  = 0x100000 + 0x100000;
__top_Flash2          = 0x100000 + 0x100000;

__base_SRAM           = 0x20000000;
__base_RAM            = 0x20000000;
__top_SRAM            = 0x20000000 + 0x68000;
__top_RAM             = 0x20000000 + 0x68000;

__base_SRAMX          = 0x4000000;
__base_RAM2           = 0x4000000;
__top_SRAMX           = 0x4000000 + 0x18000;
__top_RAM2            = 0x4000000 + 0x18000;

__base_SRAMH          = 0x20060000;
__base_RAM3           = 0x20060000;
__top_SRAMH           = 0x20060000 + 0x0;
__top_RAM3            = 0x20060000 + 0x0;

__base_USB_RAM        = 0x400ba000;
__base_RAM4           = 0x400ba000;
__top_USB_RAM         = 0x400ba000 + 0x1000;
__top_RAM4            = 0x400ba000 + 0x1000;
```

---

## 7. Firmware Source Files

### 7.1 sensor_pipeline.h

Plain C API header — callable from both C and C++ translation units.

**Key constants:**

```c
#define SENSOR_PIPELINE_N_CHANNELS       5
#define SENSOR_PIPELINE_MAX_SAMPLES      96000    /* ~59 s at 1625 Hz */
#define SENSOR_PIPELINE_SCALOGRAM_FLOATS (5 * 64 * 64)   /* 20,480 floats = 80 KB */
#define SENSOR_PIPELINE_POOL_BYTES       (96000*4 + 2*2048*4)   /* ~391 KB */
#define SENSOR_PIPELINE_IMG_BYTES        (224 * 224 * 3)         /* 150,528 bytes */
```

**API:**

```c
// Phase 1: CWT — call once per channel (0..4) before init
SensorPipelineStatus sensor_pipeline_cwt_channel(
    uint8_t *pool, float *scalogram, float *signal, int n_samples, int ch_idx);

// Phase 2: TFLite init — reuses pool as tensor arena
SensorPipeline *sensor_pipeline_init(
    uint8_t *pool, size_t pool_bytes, float *scalogram, const uint8_t *model_data);

// Phase 3: get pointer to image input tensor (write INT8 pixels here)
int8_t *sensor_pipeline_image_input_ptr(SensorPipeline *pipeline);

// Phase 4: run inference (image already in tensor, scalogram passed explicitly)
SensorPipelineStatus sensor_pipeline_infer(
    SensorPipeline *pipeline, float *scalogram, float *wear_um_out);
```

### 7.2 sensor_pipeline.cpp

**Pool layout** (pool reused across both phases):

```
 Offset 0:           ┌─────────────── signal buffer (384 KB) ───────────────┐
 CWT phase:          │  float signal[96000]  — overwritten per channel      │
                     ├──────────────── CWT workspace (16 KB) ───────────────┤
                     │  float wksp[2 × 2048]                                │
 TFLite phase:       └────────────── tensor arena (391 KB total) ───────────┘
```

**Scalogram layout** (in SRAMX, separate from pool):

```
g_scalogram[5 × 64 × 64]  in NCHW order: [ch][h][w]
CWT writes: scalo_slice = scalogram + ch_idx * 64*64
Infer transposes to NHWC: scalo_in[h * 64*5 + w*5 + c] = scalogram[c*64*64 + h*64 + w]
```

**Key implementation notes:**

1. `s_resolver` is a file-scope static `tflite::MicroMutableOpResolver<19>` — 19 ops registered in `sensor_pipeline_init()` (not at construction time).

2. The interpreter is constructed via placement-new into a static buffer to avoid heap allocation:
   ```cpp
   alignas(tflite::MicroInterpreter)
   static uint8_t s_interpreter_buf[sizeof(tflite::MicroInterpreter)];
   // ...
   s_interpreter_storage = new (s_interpreter_buf) tflite::MicroInterpreter(
       model, s_resolver, arena, ARENA_BYTES);
   ```

3. The 19 ops required by the fusion model: `QUANTIZE, CONV_2D, RESHAPE, TRANSPOSE, DEQUANTIZE, RSQRT, MEAN, SQUARED_DIFFERENCE, ADD, SUB, MUL, RELU, CONCATENATION, MAX_POOL_2D, AVERAGE_POOL_2D, FULLY_CONNECTED, LOGISTIC, REDUCE_MAX, SUM`. If the model changes, inspect the flatbuffer operator_codes table and update the resolver.

4. Output dequantisation: `wear_um = (output_int8 - zero_point) * scale`

### 7.3 main.c

**Static memory declarations:**

```c
// Pool in main SRAM BSS (zero-initialised by startup code — safe)
static uint8_t g_pool[SENSOR_PIPELINE_POOL_BYTES];   // ~391 KB

// Scalogram in SRAMX — MUST use .noinit (not .bss) to avoid startup fault
// (startup BSS init runs before BOARD_InitBootClocks enables SRAMX)
__attribute__((section(".noinit.$SRAMX")))
static float g_scalogram[SENSOR_PIPELINE_SCALOGRAM_FLOATS];   // 80 KB
```

**Initialisation order in main():**

```c
BOARD_InitBootPins();
BOARD_InitBootClocks();      // ← enables SRAMX clock/power domain
BOARD_InitDebugConsole();
memset(g_scalogram, 0, sizeof(g_scalogram));  // ← manual SRAMX zero-init
PRINTF("READY\r\n");
```

> **Critical:** Use `.noinit.$SRAMX` (not `.bss.$SRAMX`) for the scalogram. The startup code runs BSS zero-initialisation before `main()`, before `BOARD_InitBootClocks()` has enabled SRAMX. Accessing SRAMX before it is enabled causes a HardFault (slow red LED blink) with no UART output. With `.noinit`, the startup code skips SRAMX, and we manually zero it after board init.

### 7.4 CWT: cwt_mcu.h / cwt_mcu.c

Located in `experiments/cwt_c/`. Key constants:

```c
#define CWT_MCU_N_KER     2048   // FFT size (controls workspace = 2 × 2048 × 4 = 16 KB)
#define CWT_MCU_CH_SIZE   (64 * 64)  // scalogram slice per channel
```

`cwt_mcu_process_channel(signal, n, ch_idx, scalo_slice, wksp)` applies:
1. High-pass filter (force channels only: fx/fy/fz)
2. Morlet CWT at 64 scales using kiss_fft (or CMSIS-DSP)
3. Log-magnitude, normalisation, resize to 64×64
4. Writes result into `scalo_slice` (64×64 floats)

Copy `cwt_mcu.h` and `cwt_mcu.c` + `cmsis_shim.h` into the MCUXpresso project `source/` directory. In `cmsis_shim.h`, use:

```c
#include "kiss_fft.h"    // bare filename — the eIQ SDK adds -I.../third_party/kissfft
```

Not `#include "kiss_fft/kiss_fft.h"` (the subdirectory form fails with the eIQ include path).

---

## 8. UART Protocol and Host Script

### MCU side (115200 baud, 8N1)

```
Boot:
  MCU → "READY\r\n"

For each channel n = 0..4:
  MCU → "SEND_CH{n}\r\n"
  Host → one float per line (ASCII, 8 decimal places), terminated by "END\r\n"
  MCU computes CWT for channel n

After all 5 channels:
  MCU → "[pipeline] Arena used: XXX KB / 391 KB\r\n"
  MCU → "[pipeline] Image input: scale=0.0078125  zero_point=0\r\n"
  MCU → "SEND_IMAGE\r\n"
  Host → 150,528 raw INT8 bytes (224×224×3, NHWC, no header, no framing)
  MCU → "Image received OK (150528 bytes)\r\n"
  MCU → "Running inference...\r\n"
  MCU → "Predicted tool wear: XX.X um\r\n"
```

### Host script

`experiments/phase6_deployment/send_fusion_uart.py`

**Image preprocessing pipeline:**
```
Raw image (JPG/PNG)
  → crop to tool flank region (crop coords from data/raw/sets.csv)
  → resize to 224×224 (BILINEAR)
  → convert to float32, normalise to [0,1]
  → apply ImageNet normalisation: (x - mean) / std
     mean = [0.485, 0.456, 0.406],  std = [0.229, 0.224, 0.225]
  → quantise: q = clip(round(x / 0.0078125), -128, 127)
  → result: 150,528 INT8 bytes in NHWC order
```

**Usage:**
```bash
python experiments/phase6_deployment/send_fusion_uart.py \
    --port /dev/cu.usbmodemXXXXXX \
    --csv  data/raw/Set4/sensordata/<file>.csv \
    --image data/raw/Set4/<flank_image>.jpg \
    --set-id 4 \
    [--baud 921600]      # optional: faster image transfer (~1.6 s vs ~13 s)
    [--no-mask]          # skip cutting-mask extraction
```

**Transfer time at 115200 baud:** ~13 s for the 150 KB image + ~30–90 s for sensor CSV (depends on signal length).  
**Transfer time at 921600 baud:** ~1.6 s for the image; sensor CSV unchanged (ASCII line-by-line).

---

## 9. Building and Flashing

### 9.1 Build

1. Close and reopen the project in MCUXpresso (forces `.cproject` reload after "Manage linker script" change)
2. **Project → Clean…** → select Debug → OK
3. **Project → Build Project**

Expected map output:
```
PROGRAM_FLASH0   used ~1,436 KB / 2,048 KB  (70%)  ✓
SRAM             used   ~405 KB /   416 KB  (97%)  ✓
SRAMX            used     80 KB /    96 KB  (83%)  ✓
```

If SRAM usage exceeds 416 KB, reduce `SENSOR_PIPELINE_MAX_SAMPLES` in `sensor_pipeline.h`. Every 4,096-sample reduction saves 16 KB of pool — but the pool must stay ≥ ~370 KB to hold the TFLite arena, so the minimum is approximately 93,000 samples.

### 9.2 Flashing with the GUI Flash Tool

**Do not use Run → Debug for initial flashing** — the debug launch configuration has a hardcoded 1 MB flash limit that prevents loading the 1.4 MB binary (`Load failed` error).

Instead:
1. In MCUXpresso: **Quickstart Panel → Flash your application** (or toolbar lightning bolt icon)
2. Target: MCXN947 (detected automatically via LinkServer + CMSIS-DAP)
3. File to program: `${workspace_loc}/frdmmcxn947_.../Debug/frdmmcxn947_...axf`
4. Action: **Program** (not "Program mass erase first" unless the chip seems stuck)
5. ✓ Reset target on completion
6. Click **Run…**

Expected console output:
```
Erased/Wrote sector 0-175 with XXXXXXX bytes in XXXXms
Finished writing Flash successfully.
Flash Write Done
Loaded 0x15XXXX bytes in XXXXms
Reset target (romstall)
```

The `Wire ACK Fault in DAP access` message at the end is harmless — it is the debugger failing to cleanly detach after the hardware reset, not a programming error.

### 9.3 Debug configuration flash limit

To use the debugger (Run → Debug) with the 1.4 MB binary, update the debug launch configuration:

1. **Run → Debug Configurations…**
2. Select `frdmmcxn947_... LinkServer Debug`
3. **Startup** or **Memory** tab → find the Flash region at 0x0 → change size from `0x100000` (1 MB) to `0x200000` (2 MB)
4. **Apply → Close**

The `.launch` file can also be edited directly:
```xml
<!-- Find the flash region entry and update the length -->
<memoryBlockExpression address="0x0" length="0x200000" .../>
```

### 9.4 Serial terminal

The FRDM board enumerates one virtual COM port (the VCOM from the on-board LPC55S69 debugger). Find it with:

```bash
ls /dev/cu.usbmodem*    # macOS
ls /dev/ttyACM*         # Linux
```

Connect at **115200 baud, 8N1**:

```bash
# macOS
screen /dev/cu.usbmodemXXXXXX 115200

# Python (cross-platform, always available since pyserial is installed)
python -m serial.tools.miniterm /dev/cu.usbmodemXXXXXX 115200
```

**Common issue — "Resource busy":** MCUXpresso IDE's internal debugger or terminal view holds the serial port open. Check with `lsof /dev/cu.usbmodemXXXXXX` and kill or disconnect the offending process (usually LinkServer or an orphaned `screen` session).

After connecting, press the **RST button** on the board to see the boot banner.

---

## 10. Known Issues and Workarounds

### Issue 1: MCUXpresso regenerates linker scripts on every build

**Symptom:** Flash/SRAM overflow linker errors reappear after every build even after manually editing `memory.ld`.

**Cause:** `.cproject` has `"Manage linker script" = true` for the Debug configuration. The IDE regenerates `Debug/*.ld` from internal device templates before each link step.

**Fix:** Set `value="false"` for **both** the C linker and C++ linker entries in `.cproject` (search for `com.crt.advproject.link.cpp.manage` and `com.crt.advproject.link.manage`). Close and reopen the project, then clean + rebuild.

---

### Issue 2: HardFault before any UART output (slow red LED blink)

**Symptom:** After flashing, the board blinks a red LED slowly, no UART output.

**Cause (most likely):** `g_scalogram` was declared with `__attribute__((section(".bss.$SRAMX")))`. The startup code zero-initialises all BSS sections (including SRAMX BSS) before calling `main()`, before `BOARD_InitBootClocks()` has enabled the SRAMX power/clock domain. The write to SRAMX causes a HardFault.

**Fix:** Use `.noinit.$SRAMX` instead of `.bss.$SRAMX`, and zero the scalogram manually after `BOARD_InitBootClocks()`:

```c
__attribute__((section(".noinit.$SRAMX")))
static float g_scalogram[SENSOR_PIPELINE_SCALOGRAM_FLOATS];

// In main(), after BOARD_InitBootClocks():
memset(g_scalogram, 0, sizeof(g_scalogram));
```

The linker script already has a `.noinit_RAM2` section that captures `.noinit.$SRAMX` and places it in SRAMX with `(NOLOAD)` — no startup zero-init.

---

### Issue 3: "Load failed" when using Run → Debug

**Symptom:** `Failed to execute MI command: -target-download / Load failed`

**Cause:** The GDB debug launch configuration has the flash region capped at 1 MB (the default). The 1.4 MB binary exceeds this.

**Fix:** Update the debug launch configuration's flash region size to 2 MB (see §9.3), or use the GUI Flash Tool (§9.2) for all programming tasks.

---

### Issue 4: Serial port "Resource busy"

**Symptom:** `screen`, `miniterm`, or the MCUXpresso terminal cannot open the serial port.

**Cause:** An orphaned `screen` session or MCUXpresso's LinkServer is holding the port file descriptor open.

**Fix:**
```bash
lsof /dev/cu.usbmodemXXXXXX   # find the PID
kill <PID>                      # kill the orphaned process
```

Or terminate all debug sessions in MCUXpresso (Debug perspective → Terminate all) and close the IDE terminal view.

---

### Issue 5: kiss_fft "file not found"

**Symptom:** Build error: `fatal error: 'kiss_fft/kiss_fft.h' file not found`

**Cause:** `cmsis_shim.h` uses `#include "kiss_fft/kiss_fft.h"` with a subdirectory prefix, but the eIQ SDK adds `-I.../third_party/kissfft` (which contains `kiss_fft.h` directly).

**Fix:** Change to `#include "kiss_fft.h"` (no subdirectory).

---

### Issue 6: BOARD_InitBootPeripherals undefined

**Symptom:** Compile error: implicit declaration of `BOARD_InitBootPeripherals`

**Cause:** Some NXP SDK example templates call this function, but not all board support packages define it. The cifar10 example project used as the base does not have it.

**Fix:** Remove the call. The sequence `BOARD_InitBootPins() + BOARD_InitBootClocks() + BOARD_InitDebugConsole()` is sufficient.

---

### Issue 7: PROGRAM_FLASH1 "overlapping regions" GDB warning

**Symptom:** GDB prints `warning: Overlapping regions in memory map: ignoring` during debug session startup.

**Cause:** In the expanded `memory.ld`, PROGRAM_FLASH1 (0x100000–0x1FFFFF) overlaps with the upper half of PROGRAM_FLASH0 (0x0–0x1FFFFF). This is intentional — FLASH1 is kept only for section-table compatibility (`.bss_RAM3`, `.data_RAM3` entries in the linker script that reference it).

**Impact:** None. The warning is cosmetic. No code or data actually targets PROGRAM_FLASH1 sections in a typical build.

---

## 11. Adapting for Sensor-Only and Image-Only Models

### Sensor-only deployment

Replace `fusion_model_data.h` with the sensor-only model header. No image input:

1. Remove the `SEND_IMAGE` step from `main.c` (no `sensor_pipeline_image_input_ptr`)
2. Update `sensor_pipeline_init()` to expect one input: `scalogram [1, 64, 64, 5]`
3. Remove image-related constants from `sensor_pipeline.h`
4. `g_scalogram` can live in main SRAM (no SRAMX needed) if pool is reduced:
   - Arena for sensor model: ~150 KB (much smaller than fusion)
   - `MAX_SAMPLES` can be reduced to ~40,000 (pool = 160 KB + 16 KB = 176 KB)
   - SRAM: 176 KB pool + 80 KB scalogram + 12 KB overhead ≈ 268 KB < 384 KB ✓
5. Linker script: default 1 MB flash / 384 KB SRAM is sufficient — no modifications needed.
6. Host script: remove image preprocessing, only send CSV channels.

### Image-only deployment

1. Remove all CWT / scalogram logic from `main.c` and `sensor_pipeline.h/cpp`
2. Single input: `image [1, 224, 224, 3]` INT8
3. Pool serves only as the TFLite tensor arena; no signal buffer needed:
   - Arena: ~200 KB for the compressed ResNet image encoder
   - Reduce `g_pool` accordingly (e.g., 256 KB)
4. Remove `g_scalogram` entirely
5. UART protocol simplifies to: `READY` → `SEND_IMAGE` → `Predicted tool wear: XX.X um`
6. Linker script: default values are sufficient for models ≤ 1 MB.

### Shared sensor_pipeline.h approach

If deploying multiple model variants from the same codebase, use a compile-time flag:

```c
#define PIPELINE_MODE_FUSION   0
#define PIPELINE_MODE_SENSOR   1
#define PIPELINE_MODE_IMAGE    2

#ifndef PIPELINE_MODE
#  define PIPELINE_MODE PIPELINE_MODE_FUSION
#endif
```

Then conditionally compile the appropriate init/infer paths.

---

## 12. End-to-End Validation

After successful flash and boot, validate the MCU prediction against the Python TFLite reference:

### Step 1 — MCU run

```bash
python experiments/phase6_deployment/send_fusion_uart.py \
    --port /dev/cu.usbmodemXXXXXX \
    --csv  data/raw/Set4/sensordata/<file>.csv \
    --image data/raw/Set4/<flank_image>.jpg \
    --set-id 4
```

Note the output: `Predicted tool wear: XX.X um`

### Step 2 — Python TFLite reference

```python
import tensorflow as tf, numpy as np
from PIL import Image

interp = tf.lite.Interpreter(
    "checkpoints/fusion_tflite/fusion_fp32_dedup_full_integer_quant.tflite")
interp.allocate_tensors()
inp = interp.get_input_details()
out = interp.get_output_details()

# Match host script preprocessing exactly
img_bytes = preprocess_image_int8(img_path, crop_coords)  # 150,528 INT8 bytes
scalogram_int8 = quantise_scalogram(cwt_output)            # 64×64×5 INT8 NHWC

interp.set_tensor(inp[0]["index"], img_bytes.reshape(1,224,224,3))
interp.set_tensor(inp[1]["index"], scalogram_int8.reshape(1,64,64,5))
interp.invoke()

raw = interp.get_tensor(out[0]["index"])
wear_um = (float(raw[0,0]) - out[0]["quantization"][1]) * out[0]["quantization"][0]
print(f"Reference: {wear_um:.1f} um")
```

### Acceptance criterion

MCU output should match the Python TFLite reference to within ±5% (or ±5 µm, whichever is larger). Larger deviation indicates a preprocessing mismatch (crop, normalisation, or quantisation constants differ between host and MCU).

**Checklist before declaring success:**

- [ ] Boot banner printed — MCU is running
- [ ] `[pipeline] Arena used: XXX KB / 391 KB` — TFLM initialised (expect 360–380 KB)
- [ ] `[pipeline] Image input: scale=0.0078125  zero_point=0` — correct quant params
- [ ] `Image received OK (150528 bytes)` — transfer complete
- [ ] `Predicted tool wear: XX.X um` — inference returned
- [ ] MCU result matches Python reference within ±5%
