"""
validate_tflite.py — Per-sample comparison of TFLite INT8 predictions vs
PyTorch FP32 on the MATWI test split.

Validation criterion: MAE delta (|MAE_tflite - MAE_pytorch|) < 5 µm
                      Max per-sample diff < 20 µm  (sanity guard)

Usage:
    python experiments/phase6_deployment/validate_tflite.py

Outputs:
    experiments/phase6_deployment/results/validate_tflite_results.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.multiscale_sensor_cnn import MultiScaleSensorCNN
from src.data.sensor_scalogram_dataset import MATWISensorScalogramDataset

try:
    from ai_edge_litert.interpreter import Interpreter as TFLiteInterpreter
except ImportError:
    sys.exit("ai_edge_litert not found. Run: pip install ai-edge-litert")

CKPT_PATH    = ROOT / "checkpoints" / "best" / "phase4_multiscale_sgdm_best_25.pt"
TFLITE_PATH  = ROOT / "checkpoints" / "onnx" / "phase4_multiscale_sgdm_best_25_int8.tflite"
SCALO_DIR    = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
RESULTS_DIR  = Path(__file__).parent / "results"

MAE_DELTA_THRESHOLD = 5.0   # µm — pass/fail criterion
MAX_DIFF_THRESHOLD  = 20.0  # µm — per-sample sanity guard


# ── PyTorch FP32 inference ────────────────────────────────────────────────────

def run_pytorch(split: str):
    print(f"Loading PyTorch FP32 model from {CKPT_PATH.name}...")
    model = MultiScaleSensorCNN(attention="cbam")  # checkpoint pre-dates SE replacement
    state = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()

    ds = MATWISensorScalogramDataset(
        SCALO_DIR, FEATURES_PATH, split=split
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    preds, targets = [], []
    with torch.no_grad():
        for scalogram, target in loader:
            pred = model(scalogram)
            preds.append(pred.item())
            targets.append(target.item())

    return np.array(preds, dtype=np.float32), np.array(targets, dtype=np.float32)


# ── TFLite INT8 inference ─────────────────────────────────────────────────────

def run_tflite(split: str):
    print(f"Loading TFLite INT8 model from {TFLITE_PATH.name}...")
    interp = TFLiteInterpreter(model_path=str(TFLITE_PATH))
    interp.allocate_tensors()

    inp_details  = interp.get_input_details()[0]
    out_details  = interp.get_output_details()[0]
    inp_idx      = inp_details["index"]
    out_idx      = out_details["index"]
    inp_dtype    = inp_details["dtype"]

    print(f"  Input:  {inp_details['shape']}  dtype={inp_dtype.__name__}  "
          f"scale={inp_details['quantization'][0]:.6f}  "
          f"zp={inp_details['quantization'][1]}")
    print(f"  Output: {out_details['shape']}  dtype={out_details['dtype'].__name__}  "
          f"scale={out_details['quantization'][0]:.6f}  "
          f"zp={out_details['quantization'][1]}")

    ds = MATWISensorScalogramDataset(
        SCALO_DIR, FEATURES_PATH, split=split
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    preds, targets = [], []
    for scalogram, target in loader:
        x = scalogram.numpy().astype(np.float32)  # (1, 5, 64, 64)

        if inp_dtype == np.float32:
            interp.set_tensor(inp_idx, x)
        elif inp_dtype == np.int8:
            scale, zp = inp_details["quantization"]
            q = np.clip(np.round(x / scale + zp), -128, 127).astype(np.int8)
            interp.set_tensor(inp_idx, q)
        else:
            raise RuntimeError(f"Unsupported input dtype: {inp_dtype}")

        interp.invoke()

        out = interp.get_tensor(out_idx)
        if out_details["dtype"] == np.float32:
            pred = float(out.flat[0])
        elif out_details["dtype"] == np.int8:
            scale, zp = out_details["quantization"]
            pred = (float(out.flat[0]) - zp) * scale
        else:
            raise RuntimeError(f"Unsupported output dtype: {out_details['dtype']}")

        preds.append(pred)
        targets.append(target.item())

    return np.array(preds, dtype=np.float32), np.array(targets, dtype=np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for split in ("val", "test"):
        print(f"\n{'='*60}")
        print(f"Split: {split}")
        print('='*60)

        pt_preds,  targets    = run_pytorch(split)
        tfl_preds, targets_v2 = run_tflite(split)

        # Sanity: both datasets iterated in the same order
        assert np.allclose(targets, targets_v2), "Target mismatch — datasets differ!"

        # Per-sample diff (tflite - pytorch)
        diff      = tfl_preds - pt_preds
        abs_diff  = np.abs(diff)

        pt_mae    = float(np.abs(pt_preds  - targets).mean())
        tfl_mae   = float(np.abs(tfl_preds - targets).mean())
        mae_delta = abs(tfl_mae - pt_mae)

        print(f"\nResults ({split}):")
        print(f"  PyTorch FP32 MAE:      {pt_mae:.2f} µm")
        print(f"  TFLite INT8 MAE:       {tfl_mae:.2f} µm")
        print(f"  MAE delta:             {mae_delta:.2f} µm  "
              f"({'PASS' if mae_delta < MAE_DELTA_THRESHOLD else 'FAIL'} < {MAE_DELTA_THRESHOLD} µm)")
        print(f"  Per-sample diff (PT→TFL):")
        print(f"    mean:   {diff.mean():+.3f} µm  (bias)")
        print(f"    std:    {diff.std():.3f} µm")
        print(f"    max abs:{abs_diff.max():.2f} µm  "
              f"({'OK' if abs_diff.max() < MAX_DIFF_THRESHOLD else 'WARN'} < {MAX_DIFF_THRESHOLD} µm)")
        print(f"    p95:    {np.percentile(abs_diff, 95):.2f} µm")
        print(f"  Correlation (PT vs TFL): {np.corrcoef(pt_preds, tfl_preds)[0,1]:.6f}")

        pass_mae   = mae_delta < MAE_DELTA_THRESHOLD
        pass_max   = abs_diff.max() < MAX_DIFF_THRESHOLD

        results[split] = {
            "n_samples":        int(len(targets)),
            "pytorch_fp32_mae": round(pt_mae, 4),
            "tflite_int8_mae":  round(tfl_mae, 4),
            "mae_delta":        round(mae_delta, 4),
            "mae_delta_pass":   pass_mae,
            "mae_delta_threshold": MAE_DELTA_THRESHOLD,
            "per_sample_diff": {
                "mean_bias":  round(float(diff.mean()), 4),
                "std":        round(float(diff.std()),  4),
                "max_abs":    round(float(abs_diff.max()), 4),
                "p95_abs":    round(float(np.percentile(abs_diff, 95)), 4),
                "max_abs_pass": pass_max,
            },
            "correlation":      round(float(np.corrcoef(pt_preds, tfl_preds)[0,1]), 6),
            "overall_pass":     pass_mae and pass_max,
        }

    print(f"\n{'='*60}")
    overall = all(r["overall_pass"] for r in results.values())
    print(f"OVERALL: {'✅ PASS' if overall else '❌ FAIL'}")
    print('='*60)

    out_path = RESULTS_DIR / "validate_tflite_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else x)
    print(f"\nSaved results to {out_path}")

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
