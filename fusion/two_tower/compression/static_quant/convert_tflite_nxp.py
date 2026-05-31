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
    src = Path(args.int8_onnx)
    if not src.exists():
        sys.exit(f"INT8 ONNX not found: {src}")

    out = Path(args.output)
    tmp_dir = out.parent
    static = tmp_dir / "_nxp_static_dedup.onnx"

    print(f"Source INT8 ONNX : {src.name}  ({src.stat().st_size/1024:.0f} KB)")
    print("\n── Patching ONNX for the NXP converter ──────────────────────────────")
    make_static_and_dedup(src, static)

    print("\n── onnx2tflite (QDQ-aware, lossless) ────────────────────────────────")
    convert(static, out)

    if not args.keep_intermediate:
        static.unlink(missing_ok=True)
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
    p.add_argument("--keep-intermediate", action="store_true")
    run(p.parse_args())
