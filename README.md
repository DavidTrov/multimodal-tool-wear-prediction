# Edge AI for Predictive Maintenance in CNC Metal Processing

**Bachelor Thesis — Maastricht University, 2026**  
Author: David Dimitrov;
Supervisors: Marcin Pietrasik & Charis Kouzinopoulos

A feasibility study on deploying deep learning tool-wear prediction models to a resource-constrained microcontroller (NXP FRDM-MCXN947, Cortex-M33 @ 150 MHz, 2 MB flash, 512 KB SRAM).

---

## Overview

CNC metal-cutting tools degrade over time. Replacing them too early wastes resources; too late causes defects. This project builds and compresses neural networks that predict flank wear (in µm) from:

- **Image modality** — optical microscope images of the tool flank
- **Sensor modality** — multi-axis force/vibration signals (CWT scalograms)
- **Fusion modality** — both inputs combined in a two-tower network

All three modalities are compressed to INT8 and converted to TFLite, targeting the NXP FRDM-MCXN947 MCU with its eIQ Neutron NPU.

---

## Hardware Target

| Feature | Value |
|---|---|
| SoC | NXP MCXN947 |
| CPU | Cortex-M33 @ 150 MHz |
| Flash | 2 MB |
| SRAM | 512 KB (416 KB usable with SRAMH merge) |
| ML runtime | TensorFlow Lite for Microcontrollers (TFLM) |
| Conversion tool | NXP `eiq-onnx2tflite` (QDQ-preserving) |

---

## Dataset

**MATWI** — tool wear images and sensor recordings from CNC milling, paired with flank-wear measurements (µm).

| Split | Sets | Samples |
|---|---|---|
| Train | 1, 2, 5, 7, 8, 10, 11 | 647 |
| Val | 3, 6, 12 | 300 |
| Test | 4, 9, 13 | 247 |

Splits are set-based (no data leakage between tools). Raw data lives in `data/raw/Set{N}/`. Labels are in `data/raw/labels.csv`.

---

## Repository Structure

```
├── src/                          # Shared config (split maps, ImageNet stats, metrics)
├── image/
│   ├── baseline/                 # Full ResNet18 image-only baseline
│   └── compression/
│       ├── pruning/              # Structured channel pruning
│       ├── distillation/         # Knowledge distillation (teacher → student)
│       └── quantization/         # QAT fine-tuning + INT8 export + evaluation
├── sensor/
│   ├── multiscale/               # MultiScaleSensorCNN (scalogram encoder, SE attention)
│   ├── cnn/                      # Single-scale CNN baseline
│   ├── tsfresh/                  # Classical feature-based baselines (XGBoost)
│   └── deployment/               # CWT C implementation for MCU (cwt_mcu.c)
├── fusion/
│   ├── two_tower/                # Two-tower fusion model + compression pipeline
│   │   └── compression/          # Pruning, distillation, QAT, static INT8, TFLite
│   ├── scalogram/                # Scalogram dataset/model helpers
│   ├── decision_level/           # Late-fusion baseline (separate model outputs)
│   └── deployment/               # MCU firmware project, UART host script, validation
├── scripts/                      # Data visualisation, feature extraction utilities
└── data/
    ├── raw/                      # MATWI sets (Set1–Set17), labels.csv, sets.csv
    └── processed/                # Precomputed scalograms, sensor features
```

---

## Models and Results

All test MAE values are on the held-out test split (Sets 4, 9, 13).

### Image-Only (ResNet18, compressed)

Compression pipeline: structured pruning → knowledge distillation → QAT → static INT8

| Model | Test MAE (FP32) | Test MAE (INT8) | Flash |
|---|---|---|---|
| ResNet18 uncompressed | 23.17 ± 19.12 µm | — | ~45 MB |
| Compressed 2M (QAT) | 19.07 µm | **21.97 ± 22.12 µm** | 1,934 KB |
| Compressed 1.5M (QAT) | 27.61 µm | **34.52 ± 34.15 µm** | 1,458 KB |
| Compressed 1M (QAT) | 28.30 µm | **29.46 ± 27.07 µm** | 970 KB |

Checkpoints and TFLite artifacts in `image/compression/checkpoints/`.

### Sensor-Only (MultiScaleSensorCNN with SE attention)

Input: 5-channel CWT scalogram [5 × 64 × 64]. GroupNorm + SE attention left as FP32 islands; zero Flex ops.

| Model | Test MAE (FP32) | Test MAE (INT8) | Flash |
|---|---|---|---|
| MultiScaleSensorCNN (SE) | 29.27 ± 25.83 µm | **28.86 ± 25.78 µm** | 238 KB |

Checkpoints and TFLite artifact in `sensor/multiscale/checkpoints/`.

### Fusion (Two-Tower, best deployable model)

Image encoder (compressed ResNet) + scalogram encoder (MultiScaleSensorCNN) joined by a small fusion head. GELU replaced with `approximate="tanh"` for TFLite-Micro compatibility.

| Model | Test MAE (FP32) | Test MAE (INT8) | Flash |
|---|---|---|---|
| Two-tower fusion | 15.55 µm | **20.33 ± 20.43 µm** | 1,230 KB |

This is the primary deployed model. TFLite artifact with full-integer INT8 I/O in `fusion/deployment/checkpoints/fusion_int8_qat_nxp_io.tflite`.

---

## Compression and Deployment Pipeline

All three modalities share the same INT8 conversion pipeline (Path B):

```
PyTorch FP32/QAT checkpoint
  → torch.onnx.export (opset 18)
  → ONNX Runtime quantize_static (QDQ, per-tensor, symmetric INT8, 200 calib samples)
  → make_static_and_dedup()        ← static batch dim + deduplicate shared Reduce axes
  → onnx2tflite --qdq-aware-conversion --keep-io-tensors-format
  → strip_io_qdq()                 ← boundary surgery → full-integer INT8 I/O
  → onnx2tflite (re-convert)       ← INT8-I/O TFLite ready for MCU
```

**Why Path B (NXP onnx2tflite) and not Path A (onnx2tf)?** Path A discards the calibrated QDQ scales and re-quantises from scratch, causing accuracy collapse (fusion: 68 µm vs 20 µm). Path B translates scales 1:1 — lossless in principle and confirmed empirically.

---

## Setup

```bash
pip install torch torchvision Pillow pandas scikit-learn xgboost joblib pyarrow
pip install onnx onnxruntime
pip install eiq-onnx2tflite          # NXP QDQ-preserving ONNX → TFLite converter
pip install tensorflow                # only needed for TFLite evaluation on host
```

All scripts are run from the **repo root**.

---

## Running the Key Scripts

### Image-Only

```bash
# Train full ResNet18 baseline
python image/baseline/train.py

# Prune + distill to 2M / 1.5M / 1M budgets
python image/compression/pruning/train.py
python image/compression/distillation/train.py

# QAT fine-tuning
python image/compression/quantization/qat.py

# Export QAT models to INT8 TFLite (all three budgets)
python image/compression/quantization/export_onnx_static.py

# Export non-QAT (distilled) models to INT8 TFLite for comparison
python image/compression/quantization/export_onnx_static_noqat.py

# Evaluate all FP32 and INT8 deploy variants (val + test MAE ± std)
python image/compression/quantization/evaluate_deploy_models.py
```

### Sensor-Only

```bash
# Train MultiScaleSensorCNN
python sensor/multiscale/train.py

# Export to INT8 TFLite
python sensor/multiscale/export_onnx_static_sensor.py
```

### Fusion

```bash
# Train two-tower fusion model
python fusion/two_tower/train.py

# Compress (prune + distill + QAT)
python fusion/two_tower/train_compressed.py
python fusion/two_tower/train_compressed_qat.py

# Export to INT8 TFLite
python fusion/two_tower/compression/static_quant/export_onnx.py
python fusion/two_tower/compression/static_quant/convert_tflite_nxp.py

# Evaluate TFLite (FP32 I/O)
python fusion/two_tower/compression/static_quant/eval_tflite_nxp.py
```

---

## MCU Deployment

See **[`fusion/deployment/DEPLOYMENT.md`](fusion/deployment/DEPLOYMENT.md)** for the full guide, covering:

- Linker script modifications (expand flash to 2 MB, merge SRAMH to 416 KB)
- MCUXpresso IDE setup (disable managed linker script, SRAMX `.noinit` attribute)
- TFLM op resolver setup (19 ops required by the fusion model)
- UART protocol for host ↔ MCU communication
- Host script: `fusion/deployment/send_fusion_uart.py`
- Known issues and workarounds (HardFault, "Load failed", serial port busy)

**Quick start — run inference on the MCU:**

```bash
# Preprocess and stream one sample over UART
python fusion/deployment/send_fusion_uart.py \
    --port /dev/cu.usbmodemXXXXX \
    --csv  data/raw/Set4/sensordata/<file>.csv \
    --image data/raw/Set4/<flank_image>.jpg \
    --set-id 4
```

---

## Deployed Model Summary

| Model | Test MAE (INT8) | Flash | Fits 2 MB |
|---|---|---|---|
| ResNet18 2M QAT | 21.97 µm | 1,934 KB | ✓ |
| ResNet18 1.5M QAT | 34.52 µm | 1,458 KB | ✓ |
| ResNet18 1M QAT | 29.46 µm | 970 KB | ✓ |
| MultiScaleSensorCNN | 28.86 µm | 238 KB | ✓ |
| **Two-tower fusion** | **20.33 µm** | **1,230 KB** | **✓** |

The two-tower fusion model is the primary deployment target — it achieves the best accuracy while fitting comfortably within the 2 MB flash constraint.

---

## License

See [LICENSE](LICENSE).
