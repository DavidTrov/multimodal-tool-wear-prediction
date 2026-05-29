"""
Evaluate a saved ONNX model (FP32 or INT8) on val and test splits.

Usage
-----
    python experiments/compression/cwt/phase4_static_quant/eval_onnx.py
    python experiments/compression/cwt/phase4_static_quant/eval_onnx.py --model checkpoints/fusion_int8.onnx
    python experiments/compression/cwt/phase4_static_quant/eval_onnx.py --model checkpoints/fusion_fp32.onnx --split test

Run from the thesis root.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from fusion.two_tower.dataset import MATWIFusionScalogramDataset

DATA_ROOT     = ROOT / "data" / "raw"
SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
CKPT_DIR      = ROOT / "checkpoints"

BATCH_SIZE  = 16
NUM_WORKERS = 0


def evaluate(session, split: str) -> dict:
    ds     = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, split)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    preds, targets = [], []
    for imgs, scals, tgts in loader:
        out = session.run(
            None,
            {"image":     imgs.numpy().astype(np.float32),
             "scalogram": scals.numpy().astype(np.float32)},
        )[0]
        preds.append(torch.from_numpy(out).squeeze(1))
        targets.append(tgts)

    p, t = torch.cat(preds), torch.cat(targets)
    errs  = (p - t).abs()
    return {
        "mae": round(errs.mean().item(), 2),
        "std": round(errs.std().item(),  2),
        "n":   len(ds),
    }


def run(args):
    try:
        import onnxruntime as ort
    except ImportError:
        sys.exit("Missing dependency: pip install onnxruntime")

    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")

    size_kb = model_path.stat().st_size / 1024
    print(f"Model : {model_path.name}  ({size_kb:.0f} KB)")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    inputs = [i.name for i in session.get_inputs()]
    print(f"Inputs: {inputs}\n")

    splits = ["val", "test"] if args.split == "both" else [args.split]

    for split in splits:
        r = evaluate(session, split)
        print(f"  {split:<5}  n={r['n']}  MAE={r['mae']:.2f} ± {r['std']:.2f} µm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an ONNX fusion model")
    parser.add_argument(
        "--model", default=str(CKPT_DIR / "fusion_int8.onnx"),
        help="Path to ONNX model (default: checkpoints/fusion_int8.onnx)",
    )
    parser.add_argument(
        "--split", default="both", choices=["val", "test", "both"],
        help="Dataset split to evaluate (default: both)",
    )
    args = parser.parse_args()
    run(args)
