"""
Evaluate the NXP-converted INT8 TFLite fusion model on val and test splits.

The model produced by convert_tflite_nxp.py keeps NCHW FP32 I/O with INT8
internals, so it is fed exactly like the ONNX (no NHWC transpose, no boundary
quantization). It carries FlexErf (TF Select) ops from GELU, so it must run under
the full TensorFlow interpreter — NOT ai_edge_litert / TFLite-Micro.

Usage
-----
    python fusion/two_tower/compression/static_quant/eval_tflite_nxp.py

Run from the thesis root.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from fusion.two_tower.dataset import MATWIFusionScalogramDataset

DATA_ROOT       = ROOT / "data" / "raw"
SCALOGRAM_DIR   = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH   = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
DEPLOY_CKPT_DIR = ROOT / "fusion" / "deployment" / "checkpoints"


def evaluate(interp, by_name, out_idx, split: str) -> dict:
    ds = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    preds, tgts = [], []
    for img, scal, t in loader:
        interp.set_tensor(by_name["image"]["index"],     img.numpy().astype(np.float32))
        interp.set_tensor(by_name["scalogram"]["index"], scal.numpy().astype(np.float32))
        interp.invoke()
        o = interp.get_tensor(out_idx)
        preds.append(float(o.reshape(-1)[0]))
        tgts.append(float(t.reshape(-1)[0]))
    p, t = np.array(preds), np.array(tgts)
    e = np.abs(p - t)
    return {"n": len(ds), "mae": round(float(e.mean()), 2), "std": round(float(e.std()), 2)}


def run(args):
    import tensorflow as tf  # full runtime needed for FlexErf

    mp = Path(args.model)
    if not mp.exists():
        sys.exit(f"Model not found: {mp}")
    print(f"Model : {mp.name}  ({mp.stat().st_size/1024:.0f} KB)\n")

    interp = tf.lite.Interpreter(model_path=str(mp))
    interp.allocate_tensors()
    by_name = {d["name"]: d for d in interp.get_input_details()}
    out_idx = interp.get_output_details()[0]["index"]

    splits = ["val", "test"] if args.split == "both" else [args.split]
    for s in splits:
        r = evaluate(interp, by_name, out_idx, s)
        print(f"  {s:<5}  n={r['n']}  MAE={r['mae']:.2f} ± {r['std']:.2f} µm")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate NXP INT8 TFLite fusion model")
    p.add_argument("--model", default=str(DEPLOY_CKPT_DIR / "fusion_int8_qat_nxp.tflite"))
    p.add_argument("--split", default="both", choices=["val", "test", "both"])
    run(p.parse_args())
