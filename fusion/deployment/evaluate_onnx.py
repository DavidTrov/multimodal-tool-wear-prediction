"""
Phase 5 - Deployment
Evaluate the ONNX exported model to ensure accuracy matches PyTorch.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("onnxruntime is required. Please run: pip install onnxruntime")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sensor.cnn.dataset import MATWISensorScalogramDataset

SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
RESULTS_DIR   = Path(__file__).parent / "results"

# Since we exported with a static shape containing dummy_input of batch size 1
# (we removed dynamic_axes), we strictly evaluate the ONNX model 1 sample at a time.
BATCH_SIZE    = 1 
NUM_WORKERS   = 0

def evaluate_onnx(split: str, session: ort.InferenceSession, input_name: str):
    ds = MATWISensorScalogramDataset(
        SCALOGRAM_DIR, FEATURES_PATH, split=split, augment=False,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_preds, all_targets = [], []
    
    for scalograms, targets in loader:
        # ONNX runtime expects numpy arrays instead of PyTorch tensors
        inputs_np = scalograms.numpy()
        
        # Run inference
        outputs = session.run(None, {input_name: inputs_np})[0]
        
        # Collect predictions and targets
        all_preds.append(outputs.squeeze(1).flatten())
        all_targets.append(targets.numpy())

    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    errors  = np.abs(preds - targets)

    return {
        "split":     split,
        "n_samples": len(ds),
        "mae":       round(float(errors.mean()), 2),
        "mae_std":   round(float(errors.std()), 2),
        "mae_min":   round(float(errors.min()), 2),
        "mae_max":   round(float(errors.max()), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/phase4_simple_best.onnx", help="Path to ONNX model")
    args = parser.parse_args()

    model_path = ROOT / args.model_path
    if not model_path.exists():
        sys.exit(f"ONNX model not found: {model_path}\nPlease run the export script first.")

    print(f"Loading ONNX model from {model_path}...")
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    print(f"Model expects input name: '{input_name}' with shape: {session.get_inputs()[0].shape}")

    results = {}
    for split in ("train", "val", "test"):
        r = evaluate_onnx(split, session, input_name)
        results[split] = r
        print(
            f"{split:5s}  n={r['n_samples']:4d}  "
            f"MAE={r['mae']:.2f} ± {r['mae_std']:.2f} µm  "
            f"(min={r['mae_min']:.2f}, max={r['mae_max']:.2f})"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "eval_onnx_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
