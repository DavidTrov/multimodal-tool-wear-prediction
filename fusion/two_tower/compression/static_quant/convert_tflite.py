"""
ONNX -> TFLite Conversion for NXP FRDM-MCXN947 Deployment.

Converts the distilled fusion model to a fully-quantized INT8 TFLite model
via the following pipeline:

    PyTorch .pt  ->  FP32 ONNX
                 ->  FP32 ONNX (axes deduplicated for eiq-onnx2tflite compat)
                 ->  INT8 TFLite via onnx2tf flatbuffer_direct quantization

Note: CBAM has been removed from ``MultiScaleSensorCNN`` at the architecture
level (retrained without it).  No runtime replacement is needed.

Why axes are deduplicated
-------------------------
The PyTorch ONNX exporter (opset 18) emits ReduceMean/ReduceMax nodes that
share a single initializer tensor for their ``axes`` input.  NXP's
eiq-onnx2tflite (and onnx2tf before v2.5) misparse shared axes tensors as
INT32.  Duplicating them so each Reduce op has its own fixes the issue.

Note on data layout
-------------------
The TFLite model expects NHWC inputs:
  - image:     (1, 224, 224, 3)
  - scalogram: (1,  64,  64, 5)

Pre-requisite
-------------
    pip install onnx onnx2tf tensorflow torch

Usage
-----
    python experiments/compression/cwt/phase4_static_quant/convert_tflite.py
    python experiments/compression/cwt/phase4_static_quant/convert_tflite.py \\
        --calib-samples 100

Run from the thesis root.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from fusion.two_tower.dataset import MATWIFusionScalogramDataset

DATA_ROOT     = ROOT / "data" / "raw"
SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
CKPT_DIR      = ROOT / "checkpoints"
RESULTS_DIR   = Path(__file__).parent / "results"

DEFAULT_CALIB_SAMPLES = 50
NUM_WORKERS           = 0


# -- Step 1: Load model and export FP32 ONNX ----------------------------------

def export_fp32_onnx(pt_path: Path, onnx_path: Path):
    """Load distilled model and export to FP32 ONNX (no CBAM in architecture)."""
    model = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    model.eval()

    # Single-output wrapper for ONNX
    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, image, scalogram):
            out = self.m(image, scalogram)
            return out[0] if isinstance(out, tuple) else out

    wrapper = Wrapper(model)
    wrapper.eval()

    img  = torch.randn(1, 3, 224, 224)
    scal = torch.randn(1, 5, 64, 64)
    torch.onnx.export(
        wrapper, (img, scal),
        str(onnx_path),
        opset_version=18,
        input_names=["image", "scalogram"],
        output_names=["prediction"],
        dynamic_axes={
            "image": {0: "batch"},
            "scalogram": {0: "batch"},
            "prediction": {0: "batch"},
        },
    )
    size_kb = onnx_path.stat().st_size / 1024
    print(f"  Saved: {onnx_path.name} ({size_kb:.0f} KB)")
    return model  # Return for PyTorch evaluation


# -- Step 2: Deduplicate shared axes initializers ----------------------------

def deduplicate_axes(onnx_path: Path, output_path: Path):
    """Ensure each ReduceMean/ReduceMax has its own axes initializer."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))
    used_names = set()
    fixes = 0
    for node in model.graph.node:
        if node.op_type in ("ReduceMean", "ReduceMax") and len(node.input) >= 2:
            axes_name = node.input[1]
            if axes_name in used_names:
                for init in model.graph.initializer:
                    if init.name == axes_name:
                        arr = numpy_helper.to_array(init).astype(np.int64)
                        new_name = f"{axes_name}_{node.name}"
                        new_init = numpy_helper.from_array(arr, name=new_name)
                        model.graph.initializer.append(new_init)
                        node.input[1] = new_name
                        fixes += 1
                        break
            else:
                used_names.add(axes_name)

    onnx.save(model, str(output_path))
    print(f"  Deduplicated {fixes} shared axes tensors -> {output_path.name}")


# -- Step 3: onnx2tf with INT8 quantization ----------------------------------

def convert_to_int8_tflite(onnx_path: Path, output_dir: Path, calib_dir: Path):
    """Convert ONNX to INT8 TFLite via onnx2tf flatbuffer_direct quantizer."""
    import onnx2tf

    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(output_dir),
        non_verbose=True,
        output_integer_quantized_tflite=True,
        quant_type="per-channel",
        custom_input_op_name_np_data_path=[
            ["image",     str(calib_dir / "calib_images.npy")],
            ["scalogram", str(calib_dir / "calib_scalograms.npy")],
        ],
    )
    print("  onnx2tf conversion complete.")


# -- Step 4: Prepare calibration data ----------------------------------------

def prepare_calibration_data(n_samples: int, calib_dir: Path):
    """Save calibration data as numpy arrays for onnx2tf."""
    calib_dir.mkdir(parents=True, exist_ok=True)

    ds = MATWIFusionScalogramDataset(
        DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, "train"
    )
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=NUM_WORKERS)

    images, scalograms = [], []
    for i, (img, scal, _) in enumerate(loader):
        if i >= n_samples:
            break
        images.append(img.numpy())
        scalograms.append(scal.numpy())

    images     = np.concatenate(images, axis=0)       # (N, 3, 224, 224)
    scalograms = np.concatenate(scalograms, axis=0)   # (N, 5, 64, 64)

    np.save(str(calib_dir / "calib_images.npy"),     images)
    np.save(str(calib_dir / "calib_scalograms.npy"), scalograms)
    print(f"  Calibration data: {n_samples} samples saved to {calib_dir}")


# -- Evaluation ---------------------------------------------------------------

def evaluate_tflite(tflite_path: str, split: str) -> dict:
    """Evaluate a TFLite model on a dataset split. Returns MAE, std, n."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=4)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Map inputs by name
    input_map = {d["name"]: d["index"] for d in input_details}

    ds = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    preds, targets = [], []
    for img, scalo, tgt in loader:
        img_nhwc   = img.numpy().transpose(0, 2, 3, 1).astype(np.float32)
        scalo_nhwc = scalo.numpy().transpose(0, 2, 3, 1).astype(np.float32)

        interpreter.set_tensor(input_map["image"],     img_nhwc)
        interpreter.set_tensor(input_map["scalogram"], scalo_nhwc)
        interpreter.invoke()

        out = interpreter.get_tensor(output_details[0]["index"])
        preds.append(float(out[0, 0]))
        targets.append(float(tgt[0]))

    p = np.array(preds)
    t = np.array(targets)
    errs = np.abs(p - t)
    return {
        "mae":  round(float(errs.mean()), 2),
        "std":  round(float(errs.std()),  2),
        "n":    len(ds),
    }


def evaluate_pytorch(model, split: str) -> dict:
    """Evaluate a PyTorch model on a dataset split."""
    ds = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    preds, targets = [], []
    for img, scal, tgt in loader:
        with torch.no_grad():
            out = model(img, scal)
            pred = out[0] if isinstance(out, tuple) else out
        preds.append(pred.item())
        targets.append(tgt.item())

    p = np.array(preds)
    t = np.array(targets)
    errs = np.abs(p - t)
    return {
        "mae":  round(float(errs.mean()), 2),
        "std":  round(float(errs.std()),  2),
        "n":    len(ds),
    }


# -- Main --------------------------------------------------------------------

def run(args):
    pt_path    = CKPT_DIR / "fusion_distilled.pt"
    onnx_path  = CKPT_DIR / "fusion_fp32.onnx"
    dedup_path = CKPT_DIR / "fusion_fp32_dedup.onnx"
    tflite_dir = CKPT_DIR / "fusion_tflite"
    calib_dir  = CKPT_DIR / "calib_data"

    if not pt_path.exists():
        sys.exit(f"Checkpoint not found: {pt_path}")

    # -- Step 1: Export FP32 ONNX --------------------------------------------
    print("== Step 1: Export FP32 ONNX ==")
    model_fp32 = export_fp32_onnx(pt_path, onnx_path)

    # -- Step 2: Deduplicate axes --------------------------------------------
    print("\n== Step 2: Deduplicate shared axes ==")
    deduplicate_axes(onnx_path, dedup_path)

    # -- Step 3: Prepare calibration data ------------------------------------
    print(f"\n== Step 3: Prepare calibration data ({args.calib_samples} samples) ==")
    prepare_calibration_data(args.calib_samples, calib_dir)

    # -- Step 4: Convert to INT8 TFLite --------------------------------------
    print("\n== Step 4: onnx2tf -> INT8 TFLite ==")
    convert_to_int8_tflite(dedup_path, tflite_dir, calib_dir)

    # Identify output files
    int8_tflite = tflite_dir / f"{dedup_path.stem}_integer_quant.tflite"
    fp32_tflite = tflite_dir / f"{dedup_path.stem}_float32.tflite"

    fp32_kb = fp32_tflite.stat().st_size / 1024
    int8_kb = int8_tflite.stat().st_size / 1024
    print(f"\n  FP32 TFLite: {fp32_kb:.0f} KB ({fp32_kb/1024:.2f} MB)")
    print(f"  INT8 TFLite: {int8_kb:.0f} KB ({int8_kb/1024:.2f} MB)")

    # -- Step 5: Evaluate ----------------------------------------------------
    print("\n== Step 5: Evaluate ==")

    # PyTorch FP32
    pt_val  = evaluate_pytorch(model_fp32, "val")
    pt_test = evaluate_pytorch(model_fp32, "test")
    print(f"  PyTorch FP32         val MAE: {pt_val['mae']:.2f}  test MAE: {pt_test['mae']:.2f}")

    # TFLite FP32
    fp32_val  = evaluate_tflite(str(fp32_tflite), "val")
    fp32_test = evaluate_tflite(str(fp32_tflite), "test")
    print(f"  TFLite FP32          val MAE: {fp32_val['mae']:.2f}  test MAE: {fp32_test['mae']:.2f}")

    # TFLite INT8
    int8_val  = evaluate_tflite(str(int8_tflite), "val")
    int8_test = evaluate_tflite(str(int8_tflite), "test")
    print(f"  TFLite INT8          val MAE: {int8_val['mae']:.2f}  test MAE: {int8_test['mae']:.2f}")

    # -- Summary -------------------------------------------------------------
    print("\n== Summary ==")
    print(f"  {'Model':<30s} {'Val MAE':>8s} {'Test MAE':>8s} {'Size':>8s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'PyTorch FP32':<30s} {pt_val['mae']:8.2f} {pt_test['mae']:8.2f} {'—':>8s}")
    print(f"  {'TFLite FP32':<30s} {fp32_val['mae']:8.2f} {fp32_test['mae']:8.2f} {f'{fp32_kb:.0f} KB':>8s}")
    print(f"  {'TFLite INT8':<30s} {int8_val['mae']:8.2f} {int8_test['mae']:8.2f} {f'{int8_kb:.0f} KB':>8s}")
    print()
    fits = int8_kb <= 2048
    print(f"  Flash target: 2048 KB  |  INT8 model: {int8_kb:.0f} KB  |  "
          f"{'FITS' if fits else 'OVER'}")

    # -- Save results --------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "pipeline": "PyTorch -> ONNX -> onnx2tf flatbuffer_direct -> INT8 TFLite",
        "calib_samples": args.calib_samples,
        "fp32_tflite_kb": round(fp32_kb, 1),
        "int8_tflite_kb": round(int8_kb, 1),
        "fits_in_flash":  fits,
        "pytorch_fp32":   {"val_mae": pt_val["mae"],   "test_mae": pt_test["mae"]},
        "tflite_fp32":    {"val_mae": fp32_val["mae"],   "test_mae": fp32_test["mae"]},
        "tflite_int8":    {"val_mae": int8_val["mae"],   "test_mae": int8_test["mae"]},
        "accuracy_drop_int8_quant": {
            "val":  round(int8_val["mae"]  - fp32_val["mae"],  2),
            "test": round(int8_test["mae"] - fp32_test["mae"], 2),
        },
    }
    out_path = RESULTS_DIR / "tflite_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults -> {out_path}")
    print(f"Deploy  -> {int8_tflite}")
    print(f"\nNext: import {int8_tflite.name} into MCUXpresso IDE -> eIQ Toolkit\n"
          f"      or use xxd to convert to C array for bare-metal deployment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert fusion model to INT8 TFLite for NXP deployment"
    )
    parser.add_argument(
        "--calib-samples", type=int, default=DEFAULT_CALIB_SAMPLES,
        help="Training samples for INT8 calibration (default: 50)"
    )
    args = parser.parse_args()
    run(args)
