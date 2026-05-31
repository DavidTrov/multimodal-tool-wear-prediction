"""
Convert the INT8 QDQ ONNX fusion model to a deployable INT8 TFLite — LOSSLESSLY.

Background
----------
onnx2tf (used by the older convert_tflite.py) runs its OWN fresh INT8 calibration
and discards the scales baked into the QAT/ORT-quantized ONNX. The result drifts
catastrophically: test MAE 20.63 µm (ONNX) -> 68.17 µm (onnx2tf TFLite).

NXP's `eiq-onnx2tflite` package provides an `onnx2tflite` CLI with a
`--qdq-aware-conversion` mode that translates the existing per-tensor QDQ scales /
zero-points 1:1 into TFLite quantization params. No re-calibration => lossless.

Result: fusion_int8_qat_nxp.tflite  test MAE 20.63 µm (matches ONNX exactly).

Two ONNX quirks must be patched before conversion:
  1. Symbolic 'batch' dim -> bake to 1 (static shapes required).
  2. opset-18 ReduceMean nodes share ONE INT64 axes initializer. The converter
     downcasts the shared tensor in place while processing the first node, so the
     remaining nodes then see INT32 and fail with INVALID_ONNX_OPERATOR. Fix:
     give each Reduce* node its own dedicated INT64 axes initializer.

Pre-requisite
-------------
    pip install eiq-onnx2tflite        # provides the `onnx2tflite` CLI
    (NXP package: https://github.com/NXP/eiq-onnx2tflite)

Usage
-----
    python fusion/two_tower/compression/static_quant/convert_tflite_nxp.py

Run from the thesis root.

Deployment note
---------------
The GELU activations export an `Erf` op that TFLite has no native kernel for, so
the produced model carries 3 `FlexErf` ops (TF Select). This runs fine under the
desktop TF runtime (used for validation here) but is NOT supported by TFLite-Micro
on the MCU. For bare-metal deployment, retrain with a tanh-approximation GELU so
the activation lowers to native TANH. Accuracy of the QDQ translation itself is
unaffected.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from onnx.tools import update_model_dims

ROOT = Path(__file__).resolve().parents[4]
DEPLOY_CKPT_DIR = ROOT / "fusion" / "deployment" / "checkpoints"


def make_static_and_dedup(src: Path, dst: Path) -> None:
    """Bake batch=1 and give every Reduce* node its own INT64 axes initializer."""
    m = onnx.load(str(src))
    m = update_model_dims.update_inputs_outputs_dims(
        m,
        {"image": [1, 3, 224, 224], "scalogram": [1, 5, 64, 64]},
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
    print(f"  Patched model: batch=1, deduplicated {dup} Reduce* axes -> INT64")
    onnx.save(m, str(dst))


def strip_io_qdq(src: Path, dst: Path) -> dict:
    """
    Expose INT8 graph I/O by removing the boundary Quantize/Dequantize nodes.

    A QDQ model from quantize_static wraps every graph input in a leading
    QuantizeLinear (FP32 -> INT8) and the graph output in a trailing
    DequantizeLinear (INT8 -> FP32), so the .onnx presents FP32 NCHW I/O.
    On the MCU that means the 224x224x3 image input is 602 KB of FP32 — over
    the 512 KB SRAM ceiling. Re-typing the I/O to INT8 drops it to 147 KB.

    Surgery (scales/zero-points are unchanged, so accuracy is identical):
      * each input `X`:  delete `QuantizeLinear(X)`, retype graph input `X`
        to INT8, and rewire the matching DequantizeLinear to read `X` directly.
      * output: delete the final `DequantizeLinear`, rename the producing
        QuantizeLinear's output to the graph-output name, retype it INT8.

    The QDQ-aware converter then folds the now-boundary DQ/Q into the I/O
    tensors' quantization params, yielding a full-integer INT8-I/O TFLite.

    Returns the per-tensor {scale, zero_point} the host needs to quantize the
    inputs and dequantize the output.
    """
    m = onnx.load(str(src))
    g = m.graph
    init = {i.name: i for i in g.initializer}
    INT8 = onnx.TensorProto.INT8

    def scale_zp(name):
        return (float(numpy_helper.to_array(init[f"{name}_scale"])),
                int(numpy_helper.to_array(init[f"{name}_zero_point"])))

    params = {}

    # ── Inputs: drop leading QuantizeLinear, retype graph input to INT8 ──────
    for vi in g.input:
        in_name = vi.name
        q = next((n for n in g.node
                  if n.op_type == "QuantizeLinear" and n.input[0] == in_name), None)
        if q is None:
            continue
        q_out = q.output[0]
        for n in g.node:                       # rewire consumers of Q output -> input
            n.input[:] = [in_name if t == q_out else t for t in n.input]
        g.node.remove(q)
        vi.type.tensor_type.elem_type = INT8
        params[in_name] = scale_zp(in_name)

    # ── Output: drop trailing DequantizeLinear, retype graph output to INT8 ──
    for vo in g.output:
        out_name = vo.name
        dq = next((n for n in g.node
                   if n.op_type == "DequantizeLinear" and n.output[0] == out_name), None)
        if dq is None:
            continue
        producer = next(n for n in g.node if n.output[0] == dq.input[0])
        producer.output[0] = out_name          # producing Q now writes the graph output
        for n in g.node:
            n.input[:] = [out_name if t == dq.input[0] else t for t in n.input]
        g.node.remove(dq)
        vo.type.tensor_type.elem_type = INT8
        # output scale/zp keyed off the QuantizeLinear's params
        base = dq.input[1].rsplit("_scale", 1)[0] if dq.input[1].endswith("_scale") else out_name
        params[out_name] = scale_zp(base if f"{base}_scale" in init else out_name)

    onnx.save(m, str(dst))
    print("  INT8 I/O surgery:")
    for k, (s, z) in params.items():
        print(f"    {k:<14} scale={s:.8f}  zero_point={z}")
    return params


def convert(static_onnx: Path, out_tflite: Path) -> None:
    cmd = [
        "onnx2tflite",
        "--qdq-aware-conversion",      # Path B: preserve QDQ scales 1:1 (lossless)
        "--keep-io-tensors-format",    # keep NCHW FP32 I/O (matches the ONNX harness)
        "--skip-shape-inference",      # shapes already static + inferred above
        "-o", str(out_tflite),
        str(static_onnx),
    ]
    print(f"  $ {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(res.stdout)
    sys.stderr.write(res.stderr)
    if res.returncode != 0 or not out_tflite.exists():
        sys.exit(f"onnx2tflite failed (exit {res.returncode}).")


def run(args):
    import json

    src = Path(args.int8_onnx)
    if not src.exists():
        sys.exit(f"INT8 ONNX not found: {src}")

    out = Path(args.output)
    tmp_dir = out.parent
    static = tmp_dir / "_nxp_static_dedup.onnx"

    print(f"Source INT8 ONNX : {src.name}  ({src.stat().st_size/1024:.0f} KB)")

    pre = src
    if args.int8_io:
        print("\n── INT8 I/O boundary surgery (strip Quantize/Dequantize) ────────────")
        stripped = tmp_dir / "_nxp_int8_io.onnx"
        params = strip_io_qdq(src, stripped)
        pre = stripped
        params_path = out.with_suffix(".io_quant.json")
        with open(params_path, "w") as f:
            json.dump({k: {"scale": s, "zero_point": z} for k, (s, z) in params.items()},
                      f, indent=2)
        print(f"    host quant params -> {params_path.name}")

    print("\n── Patching ONNX for the NXP converter ──────────────────────────────")
    make_static_and_dedup(pre, static)

    print("\n── onnx2tflite (QDQ-aware, lossless) ────────────────────────────────")
    convert(static, out)

    if not args.keep_intermediate:
        static.unlink(missing_ok=True)
        if args.int8_io:
            (tmp_dir / "_nxp_int8_io.onnx").unlink(missing_ok=True)
        # onnx2tflite drops this scratch file in CWD on shape-infer fallback
        Path("sym_shape_infer_temp.onnx").unlink(missing_ok=True)

    kb = out.stat().st_size / 1024
    print("\n── Done ─────────────────────────────────────────────────────────────")
    print(f"  Output : {out}  ({kb:.0f} KB, fits 2 MB flash: {kb <= 2048})")
    print(f"  Validate: python {Path(__file__).with_name('eval_tflite_nxp.py')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="NXP QDQ-aware INT8 ONNX -> TFLite")
    p.add_argument("--int8-onnx", default=str(DEPLOY_CKPT_DIR / "fusion_int8_qat.onnx"))
    p.add_argument("--output", default=str(DEPLOY_CKPT_DIR / "fusion_int8_qat_nxp.tflite"))
    p.add_argument("--int8-io", action="store_true",
                   help="Strip boundary Q/DQ so the TFLite has INT8 I/O (MCU-deployable)")
    p.add_argument("--keep-intermediate", action="store_true")
    run(p.parse_args())
