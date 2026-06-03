"""
Static INT8 Quantization + ONNX Export for Compressed Image-Only ResNet18.

Converts the QAT-fine-tuned ResNet models (2M / 1.5M / 1M) to deployment-ready
INT8 ONNX with statically calibrated INT8 activations via ONNX Runtime, then
converts to INT8-I/O TFLite via NXP onnx2tflite.

The QAT checkpoints (resnet_qat_int8_{budget}.pt) are plain FP32 ResNet objects:
qat.py fine-tuned the distilled model with INT8-aware weight clamping, then ran
quantize_dynamic — which only quantizes Linear/LSTM/RNN, NOT Conv2d, so the
saved object retains FP32 Conv2d weights (the QAT-tuned ones) and exports to
ONNX cleanly. Static quantization here re-quantizes weights + activations to
INT8 from those QAT-tuned weights.

Why ONNX Runtime instead of PyTorch static quant
-------------------------------------------------
PyTorch's eager-mode static quantization (prepare/convert) cannot handle
ResNet's in-place residual additions (out += identity) without rewriting the
model to use FloatFunctional. Exporting to ONNX first and using ONNX Runtime
quantize_static bypasses this: the residual Add is just a regular ONNX node,
and QDQ pairs are inserted automatically around every quantizable op.

The previous image-only pipeline used dynamic quantization (weights INT8,
activations FP32). Static quantization also quantizes activations to INT8,
reducing peak RAM (e.g. conv1 output from 637 KB FP32 to 159 KB INT8).

Pipeline
--------
  For each budget (2m, 1p5m, 1m):
    1. Load resnet_qat_int8_{budget}.pt (QAT-tuned FP32, full-object pickle)
    2. Export FP32 ONNX (opset 18, single input: image [B,3,224,224])
    3. ONNX Runtime quantize_static → INT8 QDQ ONNX
    4. NXP onnx2tflite --qdq-aware-conversion → INT8 TFLite
    5. INT8-I/O boundary surgery → full-integer INT8 I/O TFLite
    6. Evaluate all variants (FP32 PyTorch, INT8 ONNX, TFLite)

Usage
-----
    # All three budgets
    python image/compression/quantization/export_onnx_static.py

    # Single budget
    python image/compression/quantization/export_onnx_static.py --budgets 2m

Run from the thesis root.

Pre-requisites
--------------
    pip install onnx onnxruntime eiq-onnx2tflite
"""

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from image.baseline.dataset import MATWIDataset
from src.metrics import mae

DATA_ROOT   = ROOT / "data" / "raw"
CKPT_DIR    = ROOT / "image" / "compression" / "checkpoints"
RESULTS_DIR = Path(__file__).parent / "results"

OPSET_VERSION         = 18
DEFAULT_CALIB_SAMPLES = 200
BATCH_SIZE            = 16
NUM_WORKERS           = 0


def dequantize_fc(model):
    """Replace a dynamic-quantized fc Linear with a plain FP32 nn.Linear.

    qat.py ran quantize_dynamic, which leaves Conv2d as QAT-tuned FP32 but
    converts the fc to a dynamic-quantized Linear (INT8 packed params). That
    packed object cannot be exported to ONNX, so we dequantize its weight back
    to FP32 and swap in a standard Linear. Conv2d weights are untouched.
    """
    fc = model.fc
    if not isinstance(fc, torch.ao.nn.quantized.dynamic.modules.linear.Linear):
        return model
    w = fc.weight().dequantize()
    b = fc.bias()
    lin = nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
    with torch.no_grad():
        lin.weight.copy_(w)
        if b is not None:
            lin.bias.copy_(b)
    model.fc = lin
    print("  Dequantized fc: dynamic-quantized Linear → FP32 nn.Linear "
          f"({w.shape[1]}→{w.shape[0]})")
    return model


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_torch(model, device, split):
    ds     = MATWIDataset(DATA_ROOT, split)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, tgts in loader:
            preds.append(model(imgs.to(device)).squeeze(1).cpu())
            targets.append(tgts)
    p, t = torch.cat(preds), torch.cat(targets)
    e = (p - t).abs()
    return {"mae": round(e.mean().item(), 2), "std": round(e.std().item(), 2), "n": len(ds)}


def evaluate_onnx(session, split):
    ds     = MATWIDataset(DATA_ROOT, split)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    preds, targets = [], []
    for imgs, tgts in loader:
        out = session.run(["wear_depth_um"],
                          {"image": imgs.numpy().astype(np.float32)})[0]
        preds.append(torch.from_numpy(out).squeeze(1))
        targets.append(tgts)
    p, t = torch.cat(preds), torch.cat(targets)
    e = (p - t).abs()
    return {"mae": round(e.mean().item(), 2), "std": round(e.std().item(), 2), "n": len(ds)}


def evaluate_tflite(model_path, split, int8_io=False):
    import tensorflow as tf
    ds     = MATWIDataset(DATA_ROOT, split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    preds, targets = [], []
    for imgs, tgts in loader:
        x = imgs.numpy().astype(np.float32)
        if int8_io:
            s, z = inp["quantization"]
            x = np.clip(np.round(x / s) + z, -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        raw = interp.get_tensor(out["index"]).reshape(-1)[0]
        if int8_io:
            s, z = out["quantization"]
            val = (float(raw) - z) * s
        else:
            val = float(raw)
        preds.append(val)
        targets.append(float(tgts.reshape(-1)[0]))
    p, t = np.array(preds), np.array(targets)
    e = np.abs(p - t)
    return {"mae": round(float(e.mean()), 2), "std": round(float(e.std()), 2), "n": len(ds)}


# ── Calibration data reader ──────────────────────────────────────────────────

def build_calibration_reader(n_samples):
    from onnxruntime.quantization import CalibrationDataReader

    class ImageCalibReader(CalibrationDataReader):
        def __init__(self):
            ds     = MATWIDataset(DATA_ROOT, "train")
            loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
            self.samples = []
            for i, (img, _) in enumerate(loader):
                if i >= n_samples:
                    break
                self.samples.append({"image": img.numpy().astype(np.float32)})
            self.idx = 0
            print(f"  Calibration dataset : {len(self.samples)} samples")

        def get_next(self):
            if self.idx >= len(self.samples):
                return None
            feed = self.samples[self.idx]
            self.idx += 1
            return feed

    return ImageCalibReader()


# ── ONNX preprocessing for NXP converter ─────────────────────────────────────

def make_static_and_dedup(src: Path, dst: Path):
    """Bake batch=1 and give every Reduce* node its own INT64 axes initializer."""
    import onnx
    from onnx import numpy_helper
    from onnx.tools import update_model_dims

    m = onnx.load(str(src))
    m = update_model_dims.update_inputs_outputs_dims(
        m,
        {"image": [1, 3, 224, 224]},
        {"wear_depth_um": [1, 1]},
    )
    m = onnx.shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)

    g = m.graph
    init = {i.name: i for i in g.initializer}
    reduce_ops = ("ReduceMean", "ReduceMax", "ReduceMin", "ReduceSum", "ReduceProd")
    dup = 0
    for n in g.node:
        if n.op_type in reduce_ops and len(n.input) >= 2 and n.input[1] in init:
            arr = numpy_helper.to_array(init[n.input[1]]).astype(np.int64)
            name = f"{n.input[1]}_dedup_{dup}"
            g.initializer.append(numpy_helper.from_array(arr.copy(), name))
            n.input[1] = name
            dup += 1
    print(f"  Patched: batch=1, deduplicated {dup} Reduce* axes")
    onnx.save(m, str(dst))


def strip_io_qdq(src: Path, dst: Path) -> dict:
    """Remove boundary Q/DQ to expose INT8 graph I/O."""
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(src))
    g = m.graph
    init = {i.name: i for i in g.initializer}
    INT8 = onnx.TensorProto.INT8

    def scale_zp(name):
        return (float(numpy_helper.to_array(init[f"{name}_scale"])),
                int(numpy_helper.to_array(init[f"{name}_zero_point"])))

    params = {}

    for vi in g.input:
        in_name = vi.name
        q = next((n for n in g.node
                  if n.op_type == "QuantizeLinear" and n.input[0] == in_name), None)
        if q is None:
            continue
        q_out = q.output[0]
        for n in g.node:
            n.input[:] = [in_name if t == q_out else t for t in n.input]
        g.node.remove(q)
        vi.type.tensor_type.elem_type = INT8
        params[in_name] = scale_zp(in_name)

    for vo in g.output:
        out_name = vo.name
        dq = next((n for n in g.node
                   if n.op_type == "DequantizeLinear" and n.output[0] == out_name), None)
        if dq is None:
            continue
        producer = next(n for n in g.node if n.output[0] == dq.input[0])
        producer.output[0] = out_name
        for n in g.node:
            n.input[:] = [out_name if t == dq.input[0] else t for t in n.input]
        g.node.remove(dq)
        vo.type.tensor_type.elem_type = INT8
        base = dq.input[1].rsplit("_scale", 1)[0] if dq.input[1].endswith("_scale") else out_name
        params[out_name] = scale_zp(base if f"{base}_scale" in init else out_name)

    onnx.save(m, str(dst))
    return params


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_budget(budget, calib_samples):
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_static, QuantFormat, QuantType

    ckpt_path  = CKPT_DIR / f"resnet_qat_int8_{budget}.pt"
    fp32_onnx  = CKPT_DIR / f"resnet_{budget}_fp32.onnx"
    int8_onnx  = CKPT_DIR / f"resnet_{budget}_int8.onnx"
    tflite_out = CKPT_DIR / f"resnet_{budget}_int8_nxp.tflite"
    tflite_io  = CKPT_DIR / f"resnet_{budget}_int8_nxp_io.tflite"

    if not ckpt_path.exists():
        print(f"  SKIP — checkpoint not found: {ckpt_path}")
        return None

    device = "cpu"
    torch.backends.quantized.engine = "qnnpack"

    # ── Load FP32 model ──────────────────────────────────────────────────────
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = dequantize_fc(model)
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
    tmp_static = CKPT_DIR / f"_nxp_static_{budget}.onnx"
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
    tmp_io = CKPT_DIR / f"_nxp_int8_io_{budget}.onnx"
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
        description="Static INT8 ONNX + TFLite for compressed image-only ResNet18")
    parser.add_argument("--budgets", nargs="+", default=["2m", "1p5m", "1m"],
                        help="Budget sizes to process (default: all three)")
    parser.add_argument("--calib-samples", type=int, default=DEFAULT_CALIB_SAMPLES)
    args = parser.parse_args()

    all_results = {}
    for budget in args.budgets:
        print(f"\n{'═' * 70}")
        print(f"  BUDGET: {budget}")
        print(f"{'═' * 70}")
        r = process_budget(budget, args.calib_samples)
        if r:
            all_results[budget] = r

    # ── Final summary table ──────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  SUMMARY")
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
    out_path = RESULTS_DIR / "static_quant_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
