"""
Step 1 — Prove the GELU → tanh-GELU swap is accuracy-neutral (PyTorch, zero cost).

Why
---
The NXP INT8 TFLite (fusion_int8_qat_nxp.tflite) carries 3 FlexErf (TF-Select)
ops because nn.GELU() exports an `Erf` op with no native TFLite-Micro kernel.
Swapping to nn.GELU(approximate="tanh") lowers to a native TANH and makes the
model TFLite-Micro deployable. tanh-GELU differs from erf-GELU by ~3e-4, and
the three GELUs sit on tiny (128-/64-dim) post-LayerNorm vectors, so the swap
should be lossless. This script confirms that BEFORE touching the conversion
pipeline.

What it does
------------
  1. Load fusion_distilled_qat.pt (full object).
  2. Evaluate test/val MAE as-is (erf GELU baseline).
  3. Recursively replace every nn.GELU with nn.GELU(approximate="tanh").
  4. Re-evaluate. Expect MAE to match within FP noise (~20.63 µm test).

Usage
-----
    python fusion/two_tower/compression/static_quant/check_gelu_swap.py

Run from the thesis root.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from fusion.two_tower.dataset import MATWIFusionScalogramDataset

DATA_ROOT     = ROOT / "data" / "raw"
SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
CKPT = ROOT / "fusion" / "two_tower" / "compression" / "pruning" / "checkpoints" / "fusion_distilled_qat.pt"

BATCH_SIZE = 16


def count_gelu(module, approximate):
    return sum(
        1 for m in module.modules()
        if isinstance(m, nn.GELU) and m.approximate == approximate
    )


def swap_gelu_to_tanh(module):
    """Recursively replace nn.GELU('none') with nn.GELU('tanh') in place."""
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.GELU) and child.approximate == "none":
            setattr(module, name, nn.GELU(approximate="tanh"))
            n += 1
        else:
            n += swap_gelu_to_tanh(child)
    return n


def evaluate(model, device, split):
    ds = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, split)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for imgs, scals, t in loader:
            out = model(imgs.to(device), scals.to(device))[0].squeeze(1).cpu()
            preds.append(out)
            tgts.append(t)
    p, t = torch.cat(preds), torch.cat(tgts)
    e = (p - t).abs()
    return {"n": len(ds), "mae": round(e.mean().item(), 2), "std": round(e.std().item(), 2)}


def main():
    device = "cpu"
    torch.backends.quantized.engine = "qnnpack"

    if not CKPT.exists():
        sys.exit(f"Checkpoint not found: {CKPT}")
    print(f"Loading : {CKPT.name}\n")
    model = torch.load(CKPT, map_location=device, weights_only=False).to(device).eval()

    print(f"GELU(erf)  count before swap : {count_gelu(model, 'none')}")
    print(f"GELU(tanh) count before swap : {count_gelu(model, 'tanh')}\n")

    print("── erf-GELU baseline ───────────────────────────────────────────────")
    for s in ("val", "test"):
        r = evaluate(model, device, s)
        print(f"  {s:<5} n={r['n']}  MAE={r['mae']:.2f} ± {r['std']:.2f} µm")

    n = swap_gelu_to_tanh(model)
    print(f"\nSwapped {n} GELU module(s) -> approximate='tanh'")
    print(f"GELU(erf)  count after swap  : {count_gelu(model, 'none')}")
    print(f"GELU(tanh) count after swap  : {count_gelu(model, 'tanh')}\n")

    print("── tanh-GELU (deployable) ──────────────────────────────────────────")
    for s in ("val", "test"):
        r = evaluate(model, device, s)
        print(f"  {s:<5} n={r['n']}  MAE={r['mae']:.2f} ± {r['std']:.2f} µm")


if __name__ == "__main__":
    main()
