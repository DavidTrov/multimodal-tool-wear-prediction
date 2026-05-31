"""
Evaluate the full-integer INT8-I/O fusion TFLite (MCU-deployable variant).

Unlike fusion_int8_qat_nxp.tflite (FP32 NCHW I/O), this model produced by
convert_tflite_nxp.py --int8-io exposes INT8 graph I/O: the host must quantize
the inputs and dequantize the scalar output. This is the exact arithmetic the
MCU firmware performs, so the MAE here is the true on-device accuracy.

  host quantize : q = clip(round(x / scale) + zero_point, -128, 127)   (int8)
  host dequant  : y = (q_out - zero_point) * scale

The model is native-op (TANH, no FlexErf), so it runs under ai_edge_litert /
TFLite-Micro. We still use tf.lite.Interpreter here for a single dependency.

Usage
-----
    python fusion/two_tower/compression/static_quant/eval_tflite_int8_io.py

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


def quantize(x: np.ndarray, scale: float, zp: int) -> np.ndarray:
    q = np.round(x / scale) + zp
    return np.clip(q, -128, 127).astype(np.int8)


def evaluate(interp, img_in, scal_in, out_det, split: str) -> dict:
    ds = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    o_scale, o_zp = out_det["quantization"]
    preds, tgts = [], []
    for img, scal, t in loader:
        interp.set_tensor(img_in["index"],
                          quantize(img.numpy(),  *img_in["quantization"]))
        interp.set_tensor(scal_in["index"],
                          quantize(scal.numpy(), *scal_in["quantization"]))
        interp.invoke()
        q_out = interp.get_tensor(out_det["index"]).reshape(-1)[0]
        preds.append((float(q_out) - o_zp) * o_scale)
        tgts.append(float(t.reshape(-1)[0]))
    p, t = np.array(preds), np.array(tgts)
    e = np.abs(p - t)
    return {"n": len(ds), "mae": round(float(e.mean()), 2), "std": round(float(e.std()), 2)}


def run(args):
    import tensorflow as tf

    mp = Path(args.model)
    if not mp.exists():
        sys.exit(f"Model not found: {mp}")
    print(f"Model : {mp.name}  ({mp.stat().st_size/1024:.0f} KB)\n")

    interp = tf.lite.Interpreter(model_path=str(mp))
    interp.allocate_tensors()

    # Map inputs by shape: image is the 224x224x3 tensor, scalogram the 5x64x64.
    ins = interp.get_input_details()
    img_in  = next(d for d in ins if int(np.prod(d["shape"])) == 3 * 224 * 224)
    scal_in = next(d for d in ins if int(np.prod(d["shape"])) == 5 * 64 * 64)
    out_det = interp.get_output_details()[0]

    print(f"  image     int8 scale={img_in['quantization'][0]:.8f} zp={img_in['quantization'][1]}")
    print(f"  scalogram int8 scale={scal_in['quantization'][0]:.8f} zp={scal_in['quantization'][1]}")
    print(f"  output    int8 scale={out_det['quantization'][0]:.8f} zp={out_det['quantization'][1]}\n")

    splits = ["val", "test"] if args.split == "both" else [args.split]
    for s in splits:
        r = evaluate(interp, img_in, scal_in, out_det, s)
        print(f"  {s:<5}  n={r['n']}  MAE={r['mae']:.2f} ± {r['std']:.2f} µm")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate full-integer INT8-I/O fusion TFLite")
    p.add_argument("--model", default=str(DEPLOY_CKPT_DIR / "fusion_int8_qat_nxp_io.tflite"))
    p.add_argument("--split", default="both", choices=["val", "test", "both"])
    run(p.parse_args())
