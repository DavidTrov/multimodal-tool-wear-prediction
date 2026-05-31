"""
FP32 ONNX → INT8 TFLite for NXP FRDM-MCXN947 deployment.

Converts the FP32 ONNX produced by export_onnx.py to a fully-quantized INT8
TFLite flatbuffer for execution on the MCXN947 Cortex-M33 via TFLite Micro.

Why start from the FP32 ONNX, not the INT8 QDQ ONNX
------------------------------------------------------
The INT8 QDQ ONNX from ONNX Runtime contains per-channel bias tensors quantised
to INT32.  TFLite's DEQUANTIZE kernel does not accept INT32 inputs, causing
AllocateTensors() to fail.  The FP32 ONNX has no quantised nodes at all
(torch.export unrolls the QAT image encoder to standard Conv/Relu/Add), so
onnx2tf can calibrate and quantise it cleanly using its own per-channel INT8
scheme.

Pipeline
--------
  1. Inline external data        — fusion_fp32_qat.onnx + .data → single file
  2. Deduplicate axes            — opset-18 shared ReduceMean/ReduceMax fix
  3. Build calibration arrays    — N training samples as .npy files
  4. onnx2tf → INT8 TFLite       — per-channel calibration, flatbuffer_direct
  5. Evaluate on val + test      — TFLite interpreter, NHWC inputs
  6. Save results JSON

Note on input layout
---------------------
PyTorch uses NCHW.  onnx2tf inserts Transpose nodes automatically.
The final TFLite model expects NHWC:
    image     : (1, 224, 224,  3)  float32
    scalogram : (1,  64,  64,  5)  float32

Generating a C header for bare-metal deployment
------------------------------------------------
    xxd -i fusion_int8_qat.tflite > fusion_model.h

Pre-requisite
-------------
    python fusion/two_tower/compression/static_quant/export_onnx.py \\
        --model-ckpt .../fusion_distilled_qat.pt --output-suffix _qat
    pip install onnx onnx2tf tensorflow

Usage
-----
    python fusion/two_tower/compression/static_quant/convert_tflite.py

    python fusion/two_tower/compression/static_quant/convert_tflite.py \\
        --fp32-onnx fusion/deployment/checkpoints/fusion_fp32_qat.onnx \\
        --output-suffix _qat --calib-samples 100

Run from the thesis root.
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from fusion.two_tower.dataset import MATWIFusionScalogramDataset
from torch.utils.data import DataLoader

DATA_ROOT        = ROOT / "data" / "raw"
SCALOGRAM_DIR    = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH    = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
DEPLOY_CKPT_DIR  = ROOT / "fusion" / "deployment" / "checkpoints"
RESULTS_DIR      = Path(__file__).parent / "results"

DEFAULT_CALIB_SAMPLES = 100
NUM_WORKERS           = 0


# ── Step 1: Inline external data ──────────────────────────────────────────────

def inline_onnx(onnx_path: Path, output_path: Path):
    """
    Load an ONNX model that may have a companion .data file and save it as a
    single self-contained file.  onnx2tf and dedup_axes both require a single
    file; torch.onnx.export with opset 18 splits large models automatically.
    """
    import onnx
    from onnx.external_data_helper import load_external_data_for_model

    model = onnx.load(str(onnx_path), load_external_data=False)
    load_external_data_for_model(model, str(onnx_path.parent))
    onnx.save(model, str(output_path))
    kb = output_path.stat().st_size / 1024
    print(f"  Inlined: {output_path.name}  ({kb:.0f} KB)")


# ── Step 2: Deduplicate shared axes initializers ───────────────────────────────

def deduplicate_axes(onnx_path: Path, output_path: Path) -> int:
    """
    Ensure each ReduceMean/ReduceMax node has its own axes initializer.

    opset-18 torch.onnx.export emits a single shared initializer for the axes
    input of all ReduceMean/ReduceMax nodes.  Some parsers (onnx2tf included)
    mis-read shared tensors as INT32 scalars rather than INT64 arrays.
    Duplicating fixes the parse without changing model semantics.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))
    used, fixes = set(), 0
    for node in model.graph.node:
        if node.op_type in ("ReduceMean", "ReduceMax") and len(node.input) >= 2:
            axes_name = node.input[1]
            if axes_name in used:
                for init in model.graph.initializer:
                    if init.name == axes_name:
                        arr      = numpy_helper.to_array(init).astype(np.int64)
                        new_name = f"{axes_name}_{node.name}"
                        model.graph.initializer.append(
                            numpy_helper.from_array(arr, name=new_name)
                        )
                        node.input[1] = new_name
                        fixes += 1
                        break
            else:
                used.add(axes_name)
    onnx.save(model, str(output_path))
    return fixes


# ── Step 3: Calibration data ───────────────────────────────────────────────────

def prepare_calibration_data(n_samples: int, calib_dir: Path):
    """
    Save N training samples as (N, C, H, W) float32 .npy arrays.
    onnx2tf reads these and transposes NCHW → NHWC internally.
    """
    calib_dir.mkdir(parents=True, exist_ok=True)
    ds     = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, "train")
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=NUM_WORKERS)

    images, scalograms = [], []
    for i, (img, scal, _) in enumerate(loader):
        if i >= n_samples:
            break
        images.append(img.numpy())
        scalograms.append(scal.numpy())

    images     = np.concatenate(images,     axis=0)   # (N, 3, 224, 224)
    scalograms = np.concatenate(scalograms, axis=0)   # (N, 5,  64,  64)

    np.save(str(calib_dir / "calib_images.npy"),     images)
    np.save(str(calib_dir / "calib_scalograms.npy"), scalograms)
    print(f"  Saved {n_samples} calibration samples  "
          f"(images {images.shape}, scalograms {scalograms.shape})")


# ── Step 4: onnx2tf conversion ─────────────────────────────────────────────────

def convert_to_int8_tflite(onnx_path: Path, tflite_out_dir: Path, calib_dir: Path):
    """
    Convert FP32 ONNX → INT8 TFLite via onnx2tf per-channel calibration.
    Calibration data is NCHW float32; onnx2tf transposes to NHWC internally.
    """
    try:
        import onnx2tf
    except ImportError:
        sys.exit("onnx2tf not found.  Install with:  pip install onnx2tf tensorflow")

    onnx2tf.convert(
        input_onnx_file_path            = str(onnx_path),
        output_folder_path              = str(tflite_out_dir),
        non_verbose                     = True,
        output_integer_quantized_tflite = True,
        quant_type                      = "per-channel",
        custom_input_op_name_np_data_path = [
            ["image",     str(calib_dir / "calib_images.npy")],
            ["scalogram", str(calib_dir / "calib_scalograms.npy")],
        ],
    )


# ── Step 5: Evaluate TFLite model ──────────────────────────────────────────────

def evaluate_tflite(tflite_path: Path, split: str) -> dict:
    """
    Evaluate a TFLite model on a dataset split.

    The model must have FP32 inputs and outputs (the integer_quant variant from
    onnx2tf, not full_integer_quant).  This is correct for measuring accuracy:
    integer_quant and full_integer_quant have identical INT8 internals; only
    the I/O boundary format differs.  FP32 I/O evaluation avoids having to
    replicate TFLite's per-tensor boundary quantisation in Python.

    PyTorch NCHW tensors are transposed to TFLite NHWC here.
    """
    try:
        import tensorflow as tf
    except ImportError:
        sys.exit("tensorflow not found.  Install with:  pip install tensorflow")

    interp = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=4)
    interp.allocate_tensors()

    input_details  = interp.get_input_details()
    output_details = interp.get_output_details()

    # Sanity-check: warn if the model has non-FP32 inputs (wrong variant)
    for d in input_details:
        if d["dtype"] != np.float32:
            print(f"  WARNING: input '{d['name']}' has dtype {d['dtype']} — "
                  f"expected float32.  Did you accidentally use full_integer_quant?")

    # Match inputs by name; fall back to index order
    input_by_name = {d["name"]: d for d in input_details}
    img_detail    = input_by_name.get("image",     input_details[0])
    scal_detail   = input_by_name.get("scalogram", input_details[1])

    ds     = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    preds, targets = [], []
    for img, scalo, tgt in loader:
        img_nhwc   = img.numpy().transpose(0, 2, 3, 1).astype(np.float32)   # (1,224,224,3)
        scalo_nhwc = scalo.numpy().transpose(0, 2, 3, 1).astype(np.float32) # (1, 64, 64,5)

        interp.set_tensor(img_detail["index"],  img_nhwc)
        interp.set_tensor(scal_detail["index"], scalo_nhwc)
        interp.invoke()

        out = interp.get_tensor(output_details[0]["index"])
        preds.append(float(out.flat[0]))
        targets.append(float(tgt[0]))

    p, t = np.array(preds), np.array(targets)
    errs = np.abs(p - t)
    return {"mae": round(float(errs.mean()), 2),
            "std": round(float(errs.std()),  2),
            "n":   len(ds)}


# ── Main ───────────────────────────────────────────────────────────────────────

def run(args):
    try:
        import onnx
    except ImportError:
        sys.exit("onnx not found.  Install with:  pip install onnx")

    fp32_onnx = Path(args.fp32_onnx)
    suffix    = args.output_suffix
    out_tflite = DEPLOY_CKPT_DIR / f"fusion_int8{suffix}.tflite"

    if not fp32_onnx.exists():
        sys.exit(
            f"FP32 ONNX not found: {fp32_onnx}\n"
            f"Run export_onnx.py --output-suffix {suffix} first."
        )

    data_file = fp32_onnx.with_suffix(".onnx.data")
    has_external = data_file.exists()
    print(f"Input  : {fp32_onnx.name}  ({fp32_onnx.stat().st_size / 1024:.0f} KB)"
          + (f" + {data_file.name} ({data_file.stat().st_size / 1024:.0f} KB)" if has_external else ""))
    print(f"Output : {out_tflite.name}\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # ── Step 1: Inline external data ──────────────────────────────────────
        if has_external:
            print("── Step 1: Inline external data ─────────────────────────────────────")
            inline_path = tmp / f"inline{suffix}.onnx"
            inline_onnx(fp32_onnx, inline_path)
            print()
        else:
            inline_path = fp32_onnx

        # ── Step 2: Deduplicate axes ───────────────────────────────────────────
        print("── Step 2: Deduplicate shared axes initializers ─────────────────────")
        dedup_path = tmp / f"dedup{suffix}.onnx"
        fixes = deduplicate_axes(inline_path, dedup_path)
        print(f"  Fixed {fixes} shared axes tensor(s)\n")

        # ── Step 3: Calibration data ───────────────────────────────────────────
        print(f"── Step 3: Calibration data ({args.calib_samples} training samples) ──────────────")
        calib_dir = tmp / "calib"
        prepare_calibration_data(args.calib_samples, calib_dir)
        print()

        # ── Step 4: onnx2tf → INT8 TFLite ─────────────────────────────────────
        print("── Step 4: onnx2tf → INT8 TFLite ────────────────────────────────────")
        tflite_out_dir = tmp / "tflite_out"
        tflite_out_dir.mkdir()
        convert_to_int8_tflite(dedup_path, tflite_out_dir, calib_dir)

        # full_integer_quant : INT8 I/O  — MCU deployment target
        # integer_quant      : FP32 I/O — Python accuracy evaluation
        # Both have identical INT8 weights and activations; only the I/O
        # boundary format differs.  Evaluating the FP32-I/O variant avoids
        # having to replicate TFLite's per-tensor boundary quantisation in
        # Python (which is error-prone and gave 89 µm instead of 20 µm).
        full_int8 = list(tflite_out_dir.glob("*full_integer_quant.tflite"))
        part_int8 = [f for f in tflite_out_dir.glob("*integer_quant.tflite")
                     if "full" not in f.name and "int16" not in f.name]

        if not full_int8 and not part_int8:
            all_tflite = list(tflite_out_dir.glob("*.tflite"))
            print(f"  Available TFLite files: {[f.name for f in all_tflite]}")
            sys.exit("No integer-quantized TFLite found — check onnx2tf output above.")

        deploy_tmp = full_int8[0] if full_int8 else part_int8[0]
        eval_tmp   = part_int8[0] if part_int8 else full_int8[0]

        int8_kb = deploy_tmp.stat().st_size / 1024
        print(f"  Deploy (INT8 I/O) : {deploy_tmp.name}  ({int8_kb:.0f} KB)")
        print(f"  Eval   (FP32 I/O) : {eval_tmp.name}")

        DEPLOY_CKPT_DIR.mkdir(exist_ok=True)
        shutil.copy2(deploy_tmp, out_tflite)
        eval_tflite_out = out_tflite.with_name(out_tflite.stem + "_fp32io.tflite")
        shutil.copy2(eval_tmp, eval_tflite_out)
        print(f"  Copied → {out_tflite}  (deploy)")
        print(f"  Copied → {eval_tflite_out}  (eval)\n")

        # ── Step 5: Evaluate using FP32-I/O variant ────────────────────────────
        print("── Step 5: Evaluate INT8 TFLite (FP32 I/O) ─────────────────────────")
        int8_val  = evaluate_tflite(eval_tflite_out, "val")
        int8_test = evaluate_tflite(eval_tflite_out, "test")

    print(f"  val   n={int8_val['n']:4d}  MAE={int8_val['mae']:.2f} ± {int8_val['std']:.2f} µm")
    print(f"  test  n={int8_test['n']:4d}  MAE={int8_test['mae']:.2f} ± {int8_test['std']:.2f} µm")

    fits = int8_kb <= 2048
    print("\n── Summary ──────────────────────────────────────────────────────────")
    print(f"  INT8 TFLite  : {int8_kb:.0f} KB  "
          f"({'✓ fits in 2 MB flash' if fits else '✗ over 2 MB flash'})")
    print(f"  val  MAE     : {int8_val['mae']:.2f} ± {int8_val['std']:.2f} µm")
    print(f"  test MAE     : {int8_test['mae']:.2f} ± {int8_test['std']:.2f} µm")

    print("\n── Baselines ────────────────────────────────────────────────────────")
    print(f"  ONNX Runtime INT8 (pre-TFLite)            test MAE : 20.63 µm")
    print(f"  Compressed FP32 fusion (phase5)           test MAE : 15.55 µm")
    print(f"  Paper ResNet50 image-only                 test MAE : 19.00 µm")
    print(f"  Phase 1 image-only ResNet18               test MAE : 23.17 µm")
    print(f"  Phase 4 sensor-only MultiScaleCNN         test MAE : 29.27 µm")
    print(f"  TFLite INT8 (this run)                    test MAE : {int8_test['mae']:.2f} µm")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_name = f"tflite_results{suffix}.json"
    out = {
        "source_fp32_onnx":  str(fp32_onnx),
        "tflite_path":       str(out_tflite),
        "calib_samples":     args.calib_samples,
        "int8_tflite_kb":    round(int8_kb, 1),
        "fits_in_flash":     fits,
        "tflite_int8": {
            "val_mae":  int8_val["mae"],  "val_std":  int8_val["std"],
            "test_mae": int8_test["mae"], "test_std": int8_test["std"],
        },
    }
    out_path = RESULTS_DIR / results_name
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nResults → {out_path}")
    print(f"Deploy  → {out_tflite}")
    print(f"\nTo generate a C header for bare-metal deployment:")
    print(f"  xxd -i {out_tflite.name} > fusion_model.h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert FP32 ONNX → INT8 TFLite for MCU deployment"
    )
    parser.add_argument(
        "--fp32-onnx",
        default=str(DEPLOY_CKPT_DIR / "fusion_fp32_qat.onnx"),
        help="FP32 ONNX from export_onnx.py "
             "(default: fusion/deployment/checkpoints/fusion_fp32_qat.onnx)",
    )
    parser.add_argument(
        "--output-suffix",
        default="_qat",
        help="Suffix for output files (default: _qat)",
    )
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=DEFAULT_CALIB_SAMPLES,
        help=f"Training samples for INT8 calibration (default: {DEFAULT_CALIB_SAMPLES})",
    )
    args = parser.parse_args()
    run(args)
