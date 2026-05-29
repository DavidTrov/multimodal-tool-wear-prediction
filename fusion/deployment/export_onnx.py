"""
export_onnx.py — Export MultiScaleSensorCNN to ONNX for NXP eIQ deployment.

Usage:
    python experiments/phase6_deployment/export_onnx.py
    python experiments/phase6_deployment/export_onnx.py \
        --checkpoint checkpoints/best/phase4_multiscale_sgdm_best_25.pt

Output:
    checkpoints/onnx/phase4_multiscale_sgdm_best_25.onnx   (self-contained, opset 18)

What this does
--------------
1. Loads the best MultiScaleSensorCNN checkpoint (inception-style entry, CBAM attention,
   ResBlock, GroupNorm throughout — 244K params, best test MAE ~28 µm).
2. Exports to ONNX opset 18 using the torch.export-based exporter (dynamo=True).
   GroupNorm → Reshape + InstanceNormalization + Mul + Add (standard ONNX ops).
   CBAM → ReduceMean + Sigmoid + Mul (channel attention) + ReduceMax + Conv + Sigmoid + Mul
   (spatial attention). All ops are standard and supported by ONNX2Quant.
3. Verifies the graph with onnx.checker.
4. Runs a numerical pass: compares PyTorch float32 vs ONNX Runtime outputs on a
   sample of test scalograms. Target: max abs diff < 1e-4 (float32 rounding only).
5. Reports model graph summary and MAE delta.

Memory note (INT8 on FRDM-MCXN947, 512 KB SRAM)
-------------------------------------------------
Model weights (244K × 1 byte = 239 KB) live as a const array in FLASH — zero SRAM cost.
Peak SRAM is the TFLite activation arena: ~384 KB at the inception-entry concat
(three parallel path outputs: 64+96+32 KB + concat output 192 KB, all INT8).
Total SRAM: ~445 KB < 512 KB ✓

Next step: INT8 quantization via NXP ONNX2Quant
    pip install eiq-onnx2tflite
    python experiments/phase6_deployment/build_calibration_data.py
    onnx2quant --input  checkpoints/onnx/phase4_multiscale_sgdm_best_25.onnx \
               --output checkpoints/onnx/phase4_multiscale_sgdm_best_25_int8.onnx \
               --calibration-data experiments/phase6_deployment/calibration_data.npz
"""

import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sensor.multiscale.model import MultiScaleSensorCNN

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CKPT  = ROOT / "checkpoints" / "best" / "phase4_multiscale_sgdm_best_25.pt"
OUT_DIR       = ROOT / "checkpoints" / "onnx"
SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"

OPSET    = 18
N_VERIFY = 50   # scalograms to load for numerical check


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path) -> MultiScaleSensorCNN:
    m = MultiScaleSensorCNN()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Checkpoint may be a raw state-dict or wrapped in a dict
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    m.load_state_dict(state)
    m.eval()
    return m


def export(model: MultiScaleSensorCNN, out_path: Path) -> onnx.ModelProto:
    """Export model to ONNX and return the loaded proto."""
    dummy = torch.zeros(1, 5, 64, 64)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        opset_version=OPSET,
        input_names=["scalogram"],
        output_names=["wear_um"],
        do_constant_folding=True,
        # dynamo=True is the default in PyTorch ≥ 2.5 and produces a cleaner
        # graph (34 nodes) via torch.export.  The legacy TorchScript exporter
        # (dynamo=False) produces 50 nodes with explicit Shape/Constant nodes.
    )

    # Ensure weights are embedded (no .onnx.data sidecar)
    proto = onnx.load(str(out_path), load_external_data=True)
    onnx.save(proto, str(out_path), save_as_external_data=False)

    sidecar = Path(str(out_path) + ".data")
    if sidecar.exists():
        sidecar.unlink()

    return onnx.load(str(out_path))


def print_graph_summary(proto: onnx.ModelProto) -> None:
    ops     = Counter(n.op_type for n in proto.graph.node)
    inits   = len(proto.graph.initializer)
    size_kb = sum(len(i.raw_data) for i in proto.graph.initializer) // 1024

    print("\n── Graph summary ───────────────────────────────────────────────")
    print(f"  Opset:        {proto.opset_import[0].version}")
    print(f"  Nodes:        {len(proto.graph.node)}")
    print(f"  Initializers: {inits}  ({size_kb} KB weights embedded)")
    print(f"  Op types:     {dict(ops)}")
    inp = proto.graph.input[0]
    out = proto.graph.output[0]
    in_shape  = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
    out_shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
    print(f"  Input:        '{inp.name}'  shape={in_shape}  "
          f"dtype={inp.type.tensor_type.elem_type}")
    print(f"  Output:       '{out.name}'  shape={out_shape}  "
          f"dtype={out.type.tensor_type.elem_type}")


def numerical_check(
    model: MultiScaleSensorCNN,
    proto: onnx.ModelProto,
    ckpt_path: Path,
) -> dict:
    """
    Compare PyTorch float32 vs ONNX Runtime outputs on real scalograms.

    Loads up to N_VERIFY .pt files from SCALOGRAM_DIR.
    Returns a dict with max_abs_diff, mean_abs_diff, and pt_mae / onnx_mae
    against labels if labels.csv is available.
    """
    # Collect scalogram paths
    pts = sorted(SCALOGRAM_DIR.glob("*.pt"))[:N_VERIFY]
    if not pts:
        print("  [warn] No scalogram .pt files found — skipping numerical check")
        return {}

    # Load labels for MAE comparison
    labels_df = None
    try:
        import pandas as pd
        labels_df = pd.read_csv(ROOT / "data" / "raw" / "labels.csv")
        labels_df = labels_df.reset_index()
    except Exception:
        pass

    sess      = ort.InferenceSession(proto.SerializeToString())
    inp_name  = sess.get_inputs()[0].name

    pt_preds, onnx_preds, targets = [], [], []
    abs_diffs = []

    model.eval()
    with torch.no_grad():
        for pt_path in pts:
            idx   = int(pt_path.stem)
            scalo = torch.load(pt_path, weights_only=True)  # (5, 64, 64) float32
            x     = scalo.unsqueeze(0)                       # (1, 5, 64, 64)

            pt_out   = model(x).item()
            onnx_out = sess.run(None, {inp_name: x.numpy()})[0].item()

            abs_diffs.append(abs(pt_out - onnx_out))
            pt_preds.append(pt_out)
            onnx_preds.append(onnx_out)

            if labels_df is not None and idx < len(labels_df):
                w = labels_df.loc[idx, "wear"] if "wear" in labels_df.columns else None
                if w is not None and not np.isnan(float(w)):
                    targets.append(float(w))

    result = {
        "n_checked":     len(abs_diffs),
        "max_abs_diff":  float(np.max(abs_diffs)),
        "mean_abs_diff": float(np.mean(abs_diffs)),
    }

    if targets and len(targets) == len(pt_preds):
        targets = np.array(targets)
        result["pt_mae"]   = float(np.mean(np.abs(np.array(pt_preds) - targets)))
        result["onnx_mae"] = float(np.mean(np.abs(np.array(onnx_preds) - targets)))

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export MultiScaleSensorCNN to ONNX for NXP eIQ deployment"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=str(DEFAULT_CKPT),
        help="Path to .pt checkpoint  (default: phase4_multiscale_sgdm_best_25.pt)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output .onnx path  (default: checkpoints/onnx/<checkpoint_stem>.onnx)",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    out_path = Path(args.output) if args.output else OUT_DIR / (ckpt_path.stem + ".onnx")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {ckpt_path}")
    model = load_model(ckpt_path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}  ({n_params * 4 / 1024:.1f} KB FP32)")

    # ── 2. Export ─────────────────────────────────────────────────────────────
    print(f"\nExporting to ONNX opset {OPSET}...")
    proto = export(model, out_path)

    # ── 3. Validate graph ─────────────────────────────────────────────────────
    print("Running onnx.checker...", end=" ")
    try:
        onnx.checker.check_model(proto)
        print("PASS")
    except onnx.checker.ValidationError as e:
        print(f"FAIL: {e}")
        sys.exit(1)

    print_graph_summary(proto)

    # ── 4. Numerical check ────────────────────────────────────────────────────
    print(f"\n── Numerical check ({N_VERIFY} scalograms) ──────────────────────────────────")
    stats = numerical_check(model, proto, ckpt_path)
    if stats:
        print(f"  Max  |PT − ONNX|: {stats['max_abs_diff']:.3e} µm")
        print(f"  Mean |PT − ONNX|: {stats['mean_abs_diff']:.3e} µm")
        if "pt_mae" in stats:
            print(f"  PT   MAE: {stats['pt_mae']:.2f} µm")
            print(f"  ONNX MAE: {stats['onnx_mae']:.2f} µm")
            print(f"  MAE delta: {abs(stats['onnx_mae'] - stats['pt_mae']):.2f} µm")

        tol = 1e-4
        status = "PASS ✓" if stats["max_abs_diff"] < tol else f"WARN (> {tol:.0e})"
        print(f"\n  Tolerance {tol:.0e}: {status}")

    # ── 5. Summary ────────────────────────────────────────────────────────────
    size_kb = out_path.stat().st_size / 1024
    print(f"\n── Output ──────────────────────────────────────────────────────")
    print(f"  {out_path}  ({size_kb:.1f} KB)")
    int8_path = out_path.with_name(ckpt_path.stem + "_int8.onnx")
    print(f"\nNext steps:")
    print(f"  # 1. Build calibration dataset (~100 representative scalograms)")
    print(f"  python experiments/phase6_deployment/build_calibration_data.py")
    print(f"")
    print(f"  # 2. INT8 post-training quantization (NXP ONNX2Quant)")
    print(f"  pip install eiq-onnx2tflite")
    print(f"  onnx2quant --input  {out_path} \\")
    print(f"             --output {int8_path} \\")
    print(f"             --calibration-data experiments/phase6_deployment/calibration_data.npz")
    print(f"")
    print(f"  # 3. ONNX → TFLite (eIQ ModelTool, local install)")
    print(f"  #    Input: {int8_path}")
    print(f"")
    print(f"  # 4. TFLite → Neutron NPU (Neutron Converter Tool)")
    print(f"  neutron-converter --input model_quant.tflite \\")
    print(f"                    --output model_npu.tflite \\")
    print(f"                    --target imxrt700 \\")
    print(f"                    --dump-header-file-output")


if __name__ == "__main__":
    main()
