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

from fusion.two_tower.dataset import MATWIFusionScalogramDataset

DATA_ROOT     = ROOT / "data" / "raw"
SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
DEFAULT_MODEL = ROOT / "fusion" / "deployment" / "checkpoints" / "fusion_int8.onnx"
RESULTS_DIR   = Path(__file__).parent / "results"

BATCH_SIZE    = 1
NUM_WORKERS   = 0

def evaluate_onnx(split: str, session: ort.InferenceSession):
    ds = MATWIFusionScalogramDataset(
        DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, split=split,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_preds, all_targets = [], []

    for images, scalograms, targets in loader:
        outputs = session.run(
            None,
            {"image":     images.numpy().astype(np.float32),
             "scalogram": scalograms.numpy().astype(np.float32)},
        )[0]

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
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL),
                        help="Path to ONNX fusion model")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        sys.exit(f"ONNX model not found: {model_path}\nPlease run the export script first.")

    print(f"Loading ONNX model from {model_path}...")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    for inp in session.get_inputs():
        print(f"  Input: '{inp.name}' shape={inp.shape}")

    results = {}
    for split in ("val", "test"):
        r = evaluate_onnx(split, session)
        results[split] = r
        print(
            f"{split:5s}  n={r['n_samples']:4d}  "
            f"MAE={r['mae']:.2f} +/- {r['mae_std']:.2f} um  "
            f"(min={r['mae_min']:.2f}, max={r['mae_max']:.2f})"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "eval_onnx_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
