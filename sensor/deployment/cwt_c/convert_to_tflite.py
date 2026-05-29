"""
convert_to_tflite.py — ONNX → INT8 TFLite conversion.

Converts sensor_cnn.onnx to a fully INT8-quantised TFLite model using
post-training quantisation (PTQ) calibrated against real scalograms.

Steps:
  1. ONNX → TF SavedModel  (via onnx-tf)
  2. SavedModel → INT8 TFLite  (via tf.lite.TFLiteConverter + representative dataset)

Usage:
    pip install onnx-tf tensorflow
    python experiments/cwt_c/make_calibration_dataset.py   # run once first
    python experiments/cwt_c/convert_to_tflite.py
"""

import numpy as np
import sys
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[2]
ONNX  = Path(__file__).parent / "sensor_cnn.onnx"
SAVED = Path(__file__).parent / "sensor_cnn_tf_savedmodel"
OUT   = Path(__file__).parent / "sensor_cnn_int8.tflite"
CALIB = Path(__file__).parent / "calibration_data"

# ── Checks ────────────────────────────────────────────────────────────────────
if not ONNX.exists():
    sys.exit(f"ONNX model not found: {ONNX}\n  Run: python export_sensor_cnn.py")

cal_files = sorted(CALIB.glob("*.npy"))
if not cal_files:
    sys.exit(f"No calibration data found in {CALIB}\n  Run: python make_calibration_dataset.py")

print(f"Input  : {ONNX}  ({ONNX.stat().st_size//1024} KB)")
print(f"Calib  : {len(cal_files)} samples in {CALIB}")

# ── Step 1: ONNX → TF SavedModel ─────────────────────────────────────────────
print("\n[1/2] Converting ONNX → TF SavedModel ...")
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load(str(ONNX))
tf_rep = prepare(onnx_model)
tf_rep.export_graph(str(SAVED))
print(f"      Saved → {SAVED}")

# ── Step 2: SavedModel → INT8 TFLite ─────────────────────────────────────────
print("\n[2/2] Quantising to INT8 TFLite ...")
import tensorflow as tf

def representative_dataset():
    for p in cal_files:
        arr = np.load(p).astype(np.float32)   # (1, 5, 64, 64)
        yield [arr]

converter = tf.lite.TFLiteConverter.from_saved_model(str(SAVED))
converter.optimizations             = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset    = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type      = tf.int8
converter.inference_output_type     = tf.int8

tflite_model = converter.convert()
OUT.write_bytes(tflite_model)

size_kb = len(tflite_model) // 1024
print(f"\nDone. INT8 TFLite model: {OUT}")
print(f"      Size: {size_kb} KB  (FP32 was {ONNX.stat().st_size//1024} KB, {ONNX.stat().st_size // len(tflite_model):.1f}× smaller)")

# ── Quick sanity check ────────────────────────────────────────────────────────
print("\nSanity check ...")
interp = tf.lite.Interpreter(model_content=tflite_model)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
print(f"  Input  : {inp['name']}  shape={inp['shape']}  dtype={inp['dtype'].__name__}")
print(f"  Output : {out['name']}  shape={out['shape']}  dtype={out['dtype'].__name__}")
print(f"  Quant  : scale={inp['quantization'][0]:.6f}  zero_point={inp['quantization'][1]}")

# Run one inference
dummy = np.zeros(inp['shape'], dtype=inp['dtype'])
interp.set_tensor(inp['index'], dummy)
interp.invoke()
result = interp.get_tensor(out['index'])
out_scale, out_zp = out['quantization']
wear_um = (float(result[0, 0]) - out_zp) * out_scale
print(f"  Test inference (zeros input): {wear_um:.1f} µm  ✓")
