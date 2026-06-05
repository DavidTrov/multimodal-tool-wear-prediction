"""
Static INT8 Quantization + ONNX/TFLite Export for the Multiscale Sensor CNN.

Sensor-modality counterpart of image/compression/quantization/export_onnx_static.py.
It runs the SAME Path-B pipeline (ONNX Runtime quantize_static → NXP onnx2tflite
--qdq-aware-conversion → INT8-I/O boundary surgery) so the sensor INT8 results are
methodologically consistent with the image and fusion deployables.

The quantize_static settings and the onnx2tflite flags here are identical to the
image script; only the I/O signature differs:
    input  : scalogram  [B, 5, 64, 64]
    output : wear_um     [B, 1]

Notes on op coverage
--------------------
The sensor net uses GroupNorm and an SE block (Sigmoid). ONNX Runtime's
quantize_static inserts QDQ pairs only around quantizable ops (Conv/MatMul/…);
GroupNorm and Sigmoid are left in FP32, so the resulting model is INT8 with FP32
normalization/attention islands (same situation as the fusion head). The script
reports any TF-Select ("Flex") ops in the final TFLite so deployability is
explicit. The CBAM→SE swap removed the spatial-attention Conv that corrupted
under NCHW→NHWC transposition; SE has the MobileNetV3 TFLite precedent.

Source checkpoint
-----------------
    sensor/multiscale/checkpoints/phase4_multiscale_sgdm_best.pt
    (state_dict for MultiScaleSensorCNN, attention="se")

Outputs (sensor/deployment/checkpoints/)
----------------------------------------
    sensor_multiscale_fp32.onnx
    sensor_multiscale_int8.onnx
    sensor_multiscale_int8_nxp.tflite          INT8, FP32 I/O
    sensor_multiscale_int8_nxp_io.tflite       INT8, full-integer INT8 I/O
    sensor_multiscale_int8_nxp_io.io_quant.json
    results/static_quant_results_sensor.json

Usage
-----
    python sensor/deployment/export_onnx_static_sensor.py

Run from the thesis root.
"""

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sensor.cnn.dataset import MATWISensorScalogramDataset
from sensor.multiscale.model import MultiScaleSensorCNN
from src.metrics import mae  # noqa: F401  (kept for parity; eval uses abs-error directly)

SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
# The trained source checkpoint stays with the other sensor models …
SRC_CKPT_DIR  = ROOT / "sensor" / "multiscale" / "checkpoints"
# … while every export artifact this script PRODUCES lives in the deployment
# package, next to this script.
CKPT_DIR      = ROOT / "sensor" / "deployment" / "checkpoints"
RESULTS_DIR   = Path(__file__).parent / "results"

SRC_CKPT = SRC_CKPT_DIR / "phase4_multiscale_sgdm_best.pt"

OPSET_VERSION         = 18
DEFAULT_CALIB_SAMPLES = 200
BATCH_SIZE            = 16
NUM_WORKERS           = 0

INPUT_NAME  = "scalogram"
OUTPUT_NAME = "wear_um"
INPUT_SHAPE = [1, 5, 64, 64]


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _ds(split):
    return MATWISensorScalogramDataset(SCALOGRAM_DIR, FEATURES_PATH, split=split)


def evaluate_torch(model, split):
    loader = DataLoader(_ds(split), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for scalos, tgts in loader:
            preds.append(model(scalos).squeeze(1).cpu())
            targets.append(tgts)
    p, t = torch.cat(preds), torch.cat(targets)
    e = (p - t).abs()
    return {"mae": round(e.mean().item(), 2), "std": round(e.std().item(), 2), "n": len(loader.dataset)}


def evaluate_onnx(session, split):
    loader = DataLoader(_ds(split), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    preds, targets = [], []
    for scalos, tgts in loader:
        out = session.run([OUTPUT_NAME], {INPUT_NAME: scalos.numpy().astype(np.float32)})[0]
        preds.append(torch.from_numpy(out).squeeze(1))
        targets.append(tgts)
    p, t = torch.cat(preds), torch.cat(targets)
    e = (p - t).abs()
    return {"mae": round(e.mean().item(), 2), "std": round(e.std().item(), 2), "n": len(loader.dataset)}


def evaluate_tflite(model_path, split, int8_io=False):
    import tensorflow as tf
    loader = DataLoader(_ds(split), batch_size=1, shuffle=False, num_workers=0)

    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    preds, targets = [], []
    for scalos, tgts in loader:
        x = scalos.numpy().astype(np.float32)
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
    return {"mae": round(float(e.mean()), 2), "std": round(float(e.std()), 2), "n": len(loader.dataset)}


# ── Calibration data reader ──────────────────────────────────────────────────

def build_calibration_reader(n_samples):
    from onnxruntime.quantization import CalibrationDataReader

    class ScalogramCalibReader(CalibrationDataReader):
        def __init__(self):
            loader = DataLoader(_ds("train"), batch_size=1, shuffle=True, num_workers=0)
            self.samples = []
            for i, (scalo, _) in enumerate(loader):
                if i >= n_samples:
                    break
                self.samples.append({INPUT_NAME: scalo.numpy().astype(np.float32)})
            self.idx = 0
            print(f"  Calibration dataset : {len(self.samples)} samples")

        def get_next(self):
            if self.idx >= len(self.samples):
                return None
            feed = self.samples[self.idx]
            self.idx += 1
            return feed

    return ScalogramCalibReader()


# ── ONNX preprocessing for NXP converter ─────────────────────────────────────

def make_static_and_dedup(src: Path, dst: Path):
    """Bake batch=1 and give every Reduce* node its own INT64 axes initializer.

    GroupNorm decomposition can emit ReduceMean nodes that share one INT64 axes
    initializer; the opset-18 path downcasts it in place, corrupting siblings.
    Giving each Reduce its own copy avoids that (same fix as the image script).
    """
    import onnx
    from onnx import numpy_helper
    from onnx.tools import update_model_dims

    m = onnx.load(str(src))
    m = update_model_dims.update_inputs_outputs_dims(
        m,
        {INPUT_NAME: INPUT_SHAPE},
        {OUTPUT_NAME: [1, 1]},
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
    """Remove boundary Q/DQ to expose INT8 graph I/O (generic; same as image script)."""
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


def count_flex_ops(tflite_path: Path) -> int:
    """Number of TF-Select ('Flex') ops embedded in a TFLite file (0 = TFLM-ready)."""
    return Path(tflite_path).read_bytes().count(b"Flex")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(calib_samples):
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    fp32_onnx  = CKPT_DIR / "sensor_multiscale_fp32.onnx"
    int8_onnx  = CKPT_DIR / "sensor_multiscale_int8.onnx"
    tflite_out = CKPT_DIR / "sensor_multiscale_int8_nxp.tflite"
    tflite_io  = CKPT_DIR / "sensor_multiscale_int8_nxp_io.tflite"

    if not SRC_CKPT.exists():
        sys.exit(f"Checkpoint not found: {SRC_CKPT}")

    device = "cpu"
    torch.backends.quantized.engine = "qnnpack"

    # ── Load FP32 model ──────────────────────────────────────────────────────
    model = MultiScaleSensorCNN(attention="se").to(device)
    model.load_state_dict(torch.load(SRC_CKPT, map_location=device, weights_only=True))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Source  : {SRC_CKPT.name}")
    print(f"  Params  : {n_params:,}  ({n_params / 1024:.0f} KB INT8)")

    # ── FP32 baseline ────────────────────────────────────────────────────────
    print("  FP32 evaluation ...")
    fp32_val  = evaluate_torch(model, "val")
    fp32_test = evaluate_torch(model, "test")
    print(f"    val  MAE={fp32_val['mae']:.2f} ± {fp32_val['std']:.2f}"
          f"  test MAE={fp32_test['mae']:.2f} ± {fp32_test['std']:.2f}")

    # ── Export FP32 ONNX ─────────────────────────────────────────────────────
    dummy = torch.zeros(*INPUT_SHAPE, device=device)
    CKPT_DIR.mkdir(exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # dynamo=False (legacy TorchScript exporter): the dynamo exporter
        # deduplicates constants into shared weight initializers, which makes
        # ONNX Runtime's QDQ quantizer raise "Quantization parameter shared mode
        # is not supported for weight yet". The legacy exporter keeps weights
        # per-node. (Same reason the old cwt_c sensor export forced dynamo=False.)
        torch.onnx.export(
            model, dummy, str(fp32_onnx),
            input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
            dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
            opset_version=OPSET_VERSION, do_constant_folding=True,
            dynamo=False,
        )
    onnx.checker.check_model(str(fp32_onnx))
    print(f"  FP32 ONNX : {fp32_onnx.name}  ({fp32_onnx.stat().st_size / 1024:.0f} KB)")

    # Sanity check
    sess_fp32 = ort.InferenceSession(str(fp32_onnx), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt_out = model(dummy).cpu().numpy()
    onnx_out = sess_fp32.run([OUTPUT_NAME], {INPUT_NAME: dummy.numpy().astype(np.float32)})[0]
    diff = float(np.abs(pt_out - onnx_out).max())
    print(f"  Sanity  : PT vs ONNX max diff = {diff:.6f}  {'✓' if diff < 0.01 else '⚠'}")

    # ── Static INT8 quantization (identical settings to image pipeline) ───────
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

    # Report which op types received QDQ (INT8) vs stayed FP32
    qm = onnx.load(str(int8_onnx))
    quant_targets = {n.input[0] for n in qm.graph.node if n.op_type == "QuantizeLinear"}
    op_counts = {}
    for n in qm.graph.node:
        if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
            continue
        op_counts.setdefault(n.op_type, 0)
        op_counts[n.op_type] += 1
    fp32_islands = sorted({op for op in op_counts if op in
                           ("GroupNormalization", "InstanceNormalization", "Sigmoid",
                            "Softmax", "Erf", "Pow", "ReduceMean", "ReduceMax")})
    print(f"  ONNX op types: {dict(sorted(op_counts.items()))}")
    if fp32_islands:
        print(f"  FP32 islands (not INT8-quantized): {fp32_islands}")

    # ── INT8 ONNX evaluation ─────────────────────────────────────────────────
    sess_int8 = ort.InferenceSession(str(int8_onnx), providers=["CPUExecutionProvider"])
    int8_val  = evaluate_onnx(sess_int8, "val")
    int8_test = evaluate_onnx(sess_int8, "test")
    print(f"    val  MAE={int8_val['mae']:.2f} ± {int8_val['std']:.2f} (Δ={int8_val['mae']-fp32_val['mae']:+.2f})"
          f"  test MAE={int8_test['mae']:.2f} ± {int8_test['std']:.2f} (Δ={int8_test['mae']-fp32_test['mae']:+.2f})")

    # ── NXP TFLite conversion (FP32 I/O) ─────────────────────────────────────
    print("  Converting to TFLite (NXP onnx2tflite) ...")
    tmp_static = CKPT_DIR / "_nxp_static_sensor.onnx"
    make_static_and_dedup(int8_onnx, tmp_static)
    cmd = ["onnx2tflite", "--qdq-aware-conversion", "--keep-io-tensors-format",
           "--skip-shape-inference", "-o", str(tflite_out), str(tmp_static)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not tflite_out.exists():
        print(f"  ⚠ onnx2tflite failed:\n{res.stderr[:800]}")
        tmp_static.unlink(missing_ok=True)
        return None
    tfl_kb = tflite_out.stat().st_size / 1024
    n_flex = count_flex_ops(tflite_out)
    print(f"  TFLite FP32-I/O : {tflite_out.name}  ({tfl_kb:.0f} KB)  Flex ops={n_flex}")

    # ── INT8 I/O boundary surgery ────────────────────────────────────────────
    print("  INT8-I/O boundary surgery ...")
    tmp_io = CKPT_DIR / "_nxp_int8_io_sensor.onnx"
    io_params = strip_io_qdq(int8_onnx, tmp_io)
    make_static_and_dedup(tmp_io, tmp_static)
    cmd[-2] = str(tflite_io)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not tflite_io.exists():
        print(f"  ⚠ onnx2tflite INT8-I/O failed:\n{res.stderr[:800]}")
        io_params = {}
    else:
        io_kb = tflite_io.stat().st_size / 1024
        io_flex = count_flex_ops(tflite_io)
        print(f"  TFLite INT8-I/O : {tflite_io.name}  ({io_kb:.0f} KB)  Flex ops={io_flex}")
        for k, (s, z) in io_params.items():
            print(f"    {k:<14} scale={s:.8f}  zp={z}")
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
        print(f"    val  MAE={tfl_val['mae']:.2f} ± {tfl_val['std']:.2f}"
              f"  test MAE={tfl_test['mae']:.2f} ± {tfl_test['std']:.2f}")
        tfl_results["fp32_io"] = {"val_mae": tfl_val["mae"], "val_std": tfl_val["std"],
                                  "test_mae": tfl_test["mae"], "test_std": tfl_test["std"],
                                  "flex_ops": count_flex_ops(tflite_out)}

    if tflite_io.exists():
        print("  Evaluating TFLite (INT8 I/O) ...")
        tfl_io_val  = evaluate_tflite(tflite_io, "val",  int8_io=True)
        tfl_io_test = evaluate_tflite(tflite_io, "test", int8_io=True)
        print(f"    val  MAE={tfl_io_val['mae']:.2f} ± {tfl_io_val['std']:.2f}"
              f"  test MAE={tfl_io_test['mae']:.2f} ± {tfl_io_test['std']:.2f}")
        tfl_results["int8_io"] = {"val_mae": tfl_io_val["mae"], "val_std": tfl_io_val["std"],
                                  "test_mae": tfl_io_test["mae"], "test_std": tfl_io_test["std"],
                                  "flex_ops": count_flex_ops(tflite_io)}

    # ── Summary ──────────────────────────────────────────────────────────────
    results = {
        "model": "multiscale_sensor_cnn",
        "source": SRC_CKPT.name,
        "n_params": n_params,
        "int8_kb": round(n_params / 1024, 1),
        "fp32": {"val_mae": fp32_val["mae"], "val_std": fp32_val["std"],
                 "test_mae": fp32_test["mae"], "test_std": fp32_test["std"]},
        "int8_onnx": {"val_mae": int8_val["mae"], "val_std": int8_val["std"],
                      "test_mae": int8_test["mae"], "test_std": int8_test["std"],
                      "size_kb": round(int8_kb, 0)},
        "tflite": tfl_results,
        "fp32_islands": fp32_islands,
        "io_quant_params": {k: {"scale": s, "zero_point": z}
                            for k, (s, z) in io_params.items()} if io_params else {},
    }

    # ── Final table ──────────────────────────────────────────────────────────
    def cell(mae, std):
        return f"{mae:.2f} ± {std:.2f}"

    def row(name, vcell, tcell, flex):
        print(f"{name:<22} {vcell:>16} {tcell:>16} {str(flex):>6}")

    print(f"\n{'═' * 78}")
    print("  SUMMARY — Multiscale Sensor CNN  (MAE ± std, µm)")
    print(f"{'═' * 78}")
    print(f"{'Variant':<22} {'Val MAE ± std':>16} {'Test MAE ± std':>16} {'Flex':>6}")
    print("─" * 78)
    row("FP32", cell(fp32_val["mae"], fp32_val["std"]),
        cell(fp32_test["mae"], fp32_test["std"]), "—")
    row("INT8 ONNX", cell(int8_val["mae"], int8_val["std"]),
        cell(int8_test["mae"], int8_test["std"]), "—")
    if "int8_io" in tfl_results:
        r = tfl_results["int8_io"]
        row("INT8 TFLite (INT8 I/O)", cell(r["val_mae"], r["val_std"]),
            cell(r["test_mae"], r["test_std"]), r["flex_ops"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "static_quant_results_sensor.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Static INT8 ONNX + TFLite for the multiscale sensor CNN")
    parser.add_argument("--calib-samples", type=int, default=DEFAULT_CALIB_SAMPLES)
    args = parser.parse_args()
    run(args.calib_samples)


if __name__ == "__main__":
    main()
