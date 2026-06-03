"""
Static INT8 Quantization + ONNX/TFLite Export for the NON-QAT (distilled-only)
Compressed Image-Only ResNet18.

This is the non-QAT counterpart of export_onnx_static.py. It runs the EXACT
same conversion pipeline (it imports the same helper functions) but starts from
the pruned + distilled FP32 checkpoints instead of the QAT-fine-tuned ones:

    resnet_distilled_{budget}.pt   (this script,  → resnet_{budget}_noqat_*)
    resnet_qat_int8_{budget}.pt    (export_onnx_static.py → resnet_{budget}_qat_*)

Running both lets you compare "what QAT bought you": the only variable between
the two output sets is the source weights (distilled FP32 vs QAT-tuned), since
the static-quantization pipeline (ONNX Runtime quantize_static → NXP
onnx2tflite → INT8-I/O boundary surgery) is byte-identical.

Unlike the QAT checkpoints, the distilled checkpoints are plain FP32 ResNet
objects with no dynamic-quantized fc, so the dequantize_fc step is a no-op here
(still called for parity / safety).

Pipeline (per budget 2m / 1p5m / 1m)
------------------------------------
  1. Load resnet_distilled_{budget}.pt (FP32, full-object pickle)
  2. Export FP32 ONNX (opset 18, single input: image [B,3,224,224])
  3. ONNX Runtime quantize_static → INT8 QDQ ONNX
  4. NXP onnx2tflite --qdq-aware-conversion → INT8 TFLite (FP32 I/O)
  5. INT8-I/O boundary surgery → full-integer INT8 I/O TFLite
  6. Evaluate all variants (FP32 PyTorch, INT8 ONNX, both TFLite)

Outputs (image/compression/checkpoints/)
-----------------------------------------
  resnet_{budget}_noqat_fp32.onnx          FP32 reference
  resnet_{budget}_noqat_int8.onnx          INT8 QDQ ONNX
  resnet_{budget}_noqat_int8_nxp.tflite    INT8 TFLite, FP32 I/O
  resnet_{budget}_noqat_int8_nxp_io.tflite INT8 TFLite, full-integer INT8 I/O
  resnet_{budget}_noqat_int8_nxp_io.io_quant.json   host I/O quant params
  results/static_quant_results_noqat.json  summary

Usage
-----
    # All three budgets
    python image/compression/quantization/export_onnx_static_noqat.py

    # Single budget
    python image/compression/quantization/export_onnx_static_noqat.py --budgets 2m

Run from the thesis root.

Pre-requisites
--------------
    pip install onnx onnxruntime eiq-onnx2tflite
"""

import argparse
import json
import subprocess
import warnings
from pathlib import Path

import numpy as np
import torch

# Reuse the exact pipeline helpers from the QAT export script so the two paths
# are guaranteed identical (only the source checkpoint + output names differ).
from export_onnx_static import (
    BATCH_SIZE,            # noqa: F401  (imported for parity / potential reuse)
    CKPT_DIR,
    DEFAULT_CALIB_SAMPLES,
    NUM_WORKERS,           # noqa: F401
    OPSET_VERSION,
    RESULTS_DIR,
    build_calibration_reader,
    dequantize_fc,
    evaluate_onnx,
    evaluate_tflite,
    evaluate_torch,
    make_static_and_dedup,
    strip_io_qdq,
)


def process_budget(budget, calib_samples):
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    ckpt_path  = CKPT_DIR / f"resnet_distilled_{budget}.pt"
    fp32_onnx  = CKPT_DIR / f"resnet_{budget}_noqat_fp32.onnx"
    int8_onnx  = CKPT_DIR / f"resnet_{budget}_noqat_int8.onnx"
    tflite_out = CKPT_DIR / f"resnet_{budget}_noqat_int8_nxp.tflite"
    tflite_io  = CKPT_DIR / f"resnet_{budget}_noqat_int8_nxp_io.tflite"

    if not ckpt_path.exists():
        print(f"  SKIP — checkpoint not found: {ckpt_path}")
        return None

    device = "cpu"
    torch.backends.quantized.engine = "qnnpack"

    # ── Load FP32 model ──────────────────────────────────────────────────────
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = dequantize_fc(model)          # no-op for distilled (pure FP32) models
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params  : {n_params:,}  ({n_params / 1024:.0f} KB INT8)")

    # ── FP32 baseline ────────────────────────────────────────────────────────
    print("  FP32 evaluation ...")
    fp32_val  = evaluate_torch(model, device, "val")
    fp32_test = evaluate_torch(model, device, "test")
    print(f"    val  MAE={fp32_val['mae']:.2f}  test MAE={fp32_test['mae']:.2f}")

    # ── Export FP32 ONNX ─────────────────────────────────────────────────────
    dummy = torch.randn(1, 3, 224, 224, device=device)
    CKPT_DIR.mkdir(exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model, dummy, str(fp32_onnx),
            input_names=["image"], output_names=["wear_depth_um"],
            dynamic_axes={"image": {0: "batch"}, "wear_depth_um": {0: "batch"}},
            opset_version=OPSET_VERSION, do_constant_folding=True,
        )
    onnx.checker.check_model(str(fp32_onnx))
    print(f"  FP32 ONNX : {fp32_onnx.name}  ({fp32_onnx.stat().st_size / 1024:.0f} KB)")

    # Sanity check
    sess_fp32 = ort.InferenceSession(str(fp32_onnx), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt_out = model(dummy).cpu().numpy()
    onnx_out = sess_fp32.run(["wear_depth_um"],
                              {"image": dummy.numpy().astype(np.float32)})[0]
    diff = float(np.abs(pt_out - onnx_out).max())
    print(f"  Sanity  : PT vs ONNX max diff = {diff:.6f}  {'✓' if diff < 0.01 else '⚠'}")

    # ── Static INT8 quantization ─────────────────────────────────────────────
    print(f"  Static INT8 quantization ({calib_samples} calibration samples) ...")
    calib_reader = build_calibration_reader(calib_samples)
    quantize_static(
        model_input=str(fp32_onnx), model_output=str(int8_onnx),
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ, per_channel=False,
        weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )
    int8_kb = int8_onnx.stat().st_size / 1024
    print(f"  INT8 ONNX : {int8_onnx.name}  ({int8_kb:.0f} KB)")

    # ── INT8 ONNX evaluation ─────────────────────────────────────────────────
    sess_int8 = ort.InferenceSession(str(int8_onnx), providers=["CPUExecutionProvider"])
    int8_val  = evaluate_onnx(sess_int8, "val")
    int8_test = evaluate_onnx(sess_int8, "test")
    print(f"    val  MAE={int8_val['mae']:.2f} (Δ={int8_val['mae']-fp32_val['mae']:+.2f})"
          f"  test MAE={int8_test['mae']:.2f} (Δ={int8_test['mae']-fp32_test['mae']:+.2f})")

    # ── NXP TFLite conversion (FP32 I/O) ─────────────────────────────────────
    print("  Converting to TFLite (NXP onnx2tflite) ...")
    tmp_static = CKPT_DIR / f"_nxp_static_noqat_{budget}.onnx"
    make_static_and_dedup(int8_onnx, tmp_static)
    cmd = ["onnx2tflite", "--qdq-aware-conversion", "--keep-io-tensors-format",
           "--skip-shape-inference", "-o", str(tflite_out), str(tmp_static)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not tflite_out.exists():
        print(f"  ⚠ onnx2tflite failed: {res.stderr[:200]}")
        tmp_static.unlink(missing_ok=True)
        return None
    tfl_kb = tflite_out.stat().st_size / 1024
    print(f"  TFLite FP32-I/O : {tflite_out.name}  ({tfl_kb:.0f} KB)")

    # ── INT8 I/O boundary surgery ────────────────────────────────────────────
    print("  INT8-I/O boundary surgery ...")
    tmp_io = CKPT_DIR / f"_nxp_int8_io_noqat_{budget}.onnx"
    io_params = strip_io_qdq(int8_onnx, tmp_io)
    make_static_and_dedup(tmp_io, tmp_static)
    cmd[-2] = str(tflite_io)  # reuse cmd, change output path
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not tflite_io.exists():
        print(f"  ⚠ onnx2tflite INT8-I/O failed: {res.stderr[:200]}")
    else:
        io_kb = tflite_io.stat().st_size / 1024
        print(f"  TFLite INT8-I/O : {tflite_io.name}  ({io_kb:.0f} KB)")
        for k, (s, z) in io_params.items():
            print(f"    {k:<14} scale={s:.8f}  zp={z}")

        # Save I/O quant params
        params_path = tflite_io.with_suffix(".io_quant.json")
        with open(params_path, "w") as f:
            json.dump({k: {"scale": s, "zero_point": z} for k, (s, z) in io_params.items()},
                      f, indent=2)

    # Cleanup scratch
    tmp_static.unlink(missing_ok=True)
    tmp_io.unlink(missing_ok=True)
    Path("sym_shape_infer_temp.onnx").unlink(missing_ok=True)

    # ── Evaluate TFLite ──────────────────────────────────────────────────────
    tfl_results = {}
    if tflite_out.exists():
        print("  Evaluating TFLite (FP32 I/O) ...")
        tfl_val  = evaluate_tflite(tflite_out, "val",  int8_io=False)
        tfl_test = evaluate_tflite(tflite_out, "test", int8_io=False)
        print(f"    val  MAE={tfl_val['mae']:.2f}  test MAE={tfl_test['mae']:.2f}")
        tfl_results["fp32_io"] = {"val_mae": tfl_val["mae"], "test_mae": tfl_test["mae"]}

    if tflite_io.exists():
        print("  Evaluating TFLite (INT8 I/O) ...")
        tfl_io_val  = evaluate_tflite(tflite_io, "val",  int8_io=True)
        tfl_io_test = evaluate_tflite(tflite_io, "test", int8_io=True)
        print(f"    val  MAE={tfl_io_val['mae']:.2f}  test MAE={tfl_io_test['mae']:.2f}")
        tfl_results["int8_io"] = {"val_mae": tfl_io_val["mae"], "test_mae": tfl_io_test["mae"]}

    # ── Summary ──────────────────────────────────────────────────────────────
    results = {
        "budget": budget,
        "source": "distilled_fp32_noqat",
        "n_params": n_params,
        "int8_kb": round(n_params / 1024, 1),
        "fp32": {"val_mae": fp32_val["mae"], "test_mae": fp32_test["mae"]},
        "int8_onnx": {"val_mae": int8_val["mae"], "test_mae": int8_test["mae"],
                      "size_kb": round(int8_kb, 0)},
        "tflite": tfl_results,
        "io_quant_params": {k: {"scale": s, "zero_point": z}
                            for k, (s, z) in io_params.items()} if io_params else {},
    }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Static INT8 ONNX + TFLite for NON-QAT (distilled) image-only ResNet18")
    parser.add_argument("--budgets", nargs="+", default=["2m", "1p5m", "1m"],
                        help="Budget sizes to process (default: all three)")
    parser.add_argument("--calib-samples", type=int, default=DEFAULT_CALIB_SAMPLES)
    args = parser.parse_args()

    all_results = {}
    for budget in args.budgets:
        print(f"\n{'═' * 70}")
        print(f"  BUDGET: {budget}  (NON-QAT / distilled source)")
        print(f"{'═' * 70}")
        r = process_budget(budget, args.calib_samples)
        if r:
            all_results[budget] = r

    # ── Final summary table ──────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  SUMMARY  (NON-QAT / distilled source)")
    print(f"{'═' * 70}")
    print(f"{'Budget':<8} {'Params':>8} {'INT8 KB':>8} {'FP32 test':>10} {'INT8 test':>10} "
          f"{'TFL test':>10} {'TFL-IO test':>12} {'Fits 2MB':>9}")
    print("─" * 85)
    for b, r in all_results.items():
        tfl_test = r["tflite"].get("fp32_io", {}).get("test_mae", "—")
        tfl_io_test = r["tflite"].get("int8_io", {}).get("test_mae", "—")
        fits = "✓" if r["int8_kb"] <= 2048 else "✗"
        print(f"{b:<8} {r['n_params']:>8,} {r['int8_kb']:>8.0f} "
              f"{r['fp32']['test_mae']:>10.2f} {r['int8_onnx']['test_mae']:>10.2f} "
              f"{str(tfl_test):>10} {str(tfl_io_test):>12} {fits:>9}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "static_quant_results_noqat.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
