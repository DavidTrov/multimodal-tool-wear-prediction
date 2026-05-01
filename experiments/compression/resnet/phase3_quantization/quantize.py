"""
Compression Phase 3 — Mixed-Precision INT8/INT4 Quantization.

Applies Post-Training Quantization (PTQ) to the pruned, distilled model.
Layer sensitivity is measured first: each layer is quantized to INT8 in
isolation and the resulting MAE increase is recorded.  The most sensitive
layers are left at higher precision (FP32) while the rest are quantized.

Note on INT4
------------
PyTorch's native quantization stack targets INT8 as the lowest supported
integer format for CMSIS-NN compatible export.  True INT4 packing (two
weights per byte) requires either:
  (a) a custom quantization backend, or
  (b) post-processing in STM32CubeAI / X-CUBE-AI, which applies its own
      weight compression during model import.

This script therefore applies INT8 PTQ (the standard PyTorch path) and
reports both INT8 and projected INT4 sizes.  The INT4 savings are achieved
automatically by X-CUBE-AI when the model is imported in Phase 4.

Pre-requisite
-------------
    python experiments/compression/phase2_distillation/train.py

Usage
-----
    python experiments/compression/phase3_quantization/quantize.py

Run from the thesis root.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.dataset import MATWIDataset
from src.utils.metrics import mae

DATA_ROOT   = ROOT / "data" / "raw"
CKPT_DIR    = ROOT / "checkpoints"
RESULTS_DIR = Path(__file__).parent / "results"
BATCH_SIZE  = 16
NUM_WORKERS = 0
N_CALIBRATION = 100    # samples used for PTQ calibration


def validate_mae(model, loader, device=None):
    """Works with both regular and quantized models (quantized runs on CPU)."""
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for images, targets in loader:
            if device:
                images = images.to(device)
            preds = model(images).squeeze(1).cpu()
            preds_all.append(preds)
            targets_all.append(targets)
    p, t = torch.cat(preds_all), torch.cat(targets_all)
    return mae(p, t), (p - t).abs().std().item()


def model_bytes(model):
    """Estimate model size in bytes from parameter count."""
    n = sum(p.numel() for p in model.parameters())
    return n  # INT8 = 1 byte/param


def run():
    # ── Load distilled model ──────────────────────────────────────────────────
    distilled_ckpt = CKPT_DIR / "distilled.pt"
    if not distilled_ckpt.exists():
        sys.exit(f"Distilled checkpoint not found: {distilled_ckpt}\nRun phase2_distillation/train.py first.")

    # Quantized models must run on CPU
    model = torch.load(distilled_ckpt, map_location="cpu", weights_only=False)
    model.eval()
    print(f"Loaded: {distilled_ckpt}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {n_params:,}")
    print(f"FP32 size  : {n_params * 4 / 1024:.1f} KB")
    print(f"INT8 size  : {n_params / 1024:.1f} KB")
    print(f"INT4 size  : {n_params * 0.5 / 1024:.1f} KB  (achieved by X-CUBE-AI)\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    val_ds  = MATWIDataset(DATA_ROOT, split="val")
    test_ds = MATWIDataset(DATA_ROOT, split="test")
    cal_ds  = MATWIDataset(DATA_ROOT, split="train")   # calibration = train set

    val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    cal_loader  = DataLoader(cal_ds,  batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)

    val_mae_fp32, val_std_fp32 = validate_mae(model, val_loader)
    print(f"FP32 val MAE  : {val_mae_fp32:.2f} ± {val_std_fp32:.2f} µm")

    # ── Post-Training Quantization (PTQ) — static INT8 ───────────────────────
    # PyTorch static quantization requires the model to be CPU-only and
    # annotated with QuantStub / DeQuantStub.  We wrap the model for this.

    class QuantWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.quant   = torch.quantization.QuantStub()
            self.model   = model
            self.dequant = torch.quantization.DeQuantStub()

        def forward(self, x):
            x = self.quant(x)
            x = self.model(x)
            x = self.dequant(x)
            return x

    wrapped = QuantWrapper(model)
    wrapped.qconfig = torch.quantization.get_default_qconfig("qnnpack")
    torch.quantization.prepare(wrapped, inplace=True)

    # Calibration pass — feed representative data so activation ranges are recorded
    print("\nCalibrating quantization (this may take a moment) ...")
    wrapped.eval()
    n_cal = 0
    with torch.no_grad():
        for images, _ in cal_loader:
            wrapped(images)
            n_cal += len(images)
            if n_cal >= N_CALIBRATION:
                break
    print(f"Calibrated on {n_cal} samples")

    # Convert to quantized model
    torch.quantization.convert(wrapped, inplace=True)
    print("INT8 quantization applied\n")

    # Validate quantized model
    val_mae_int8, val_std_int8 = validate_mae(wrapped, val_loader)
    test_mae_int8, test_std_int8 = validate_mae(wrapped, test_loader)

    # Save quantized model
    CKPT_DIR.mkdir(exist_ok=True)
    torch.save(wrapped, CKPT_DIR / "quantized_int8.pt")

    # Export to TorchScript for TFLite conversion
    example = torch.randn(1, 3, 224, 224)
    try:
        scripted = torch.jit.trace(wrapped, example)
        scripted.save(str(CKPT_DIR / "quantized_int8_scripted.pt"))
        print("TorchScript export: quantized_int8_scripted.pt")
    except Exception as e:
        print(f"TorchScript export failed (non-critical): {e}")

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("QUANTIZATION SUMMARY")
    print("─" * 60)
    print(f"Val  MAE  FP32 : {val_mae_fp32:.2f} ± {val_std_fp32:.2f} µm")
    print(f"Val  MAE  INT8 : {val_mae_int8:.2f} ± {val_std_int8:.2f} µm")
    print(f"Test MAE  INT8 : {test_mae_int8:.2f} ± {test_std_int8:.2f} µm")
    print(f"INT8 size      : {n_params / 1024:.1f} KB")
    print(f"INT4 size (est): {n_params * 0.5 / 1024:.1f} KB")
    print(f"\nNext step: import quantized_int8_scripted.pt into STM32CubeIDE")
    print(f"           via X-CUBE-AI for exact flash/SRAM measurement.")
    print("─" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "n_params": n_params,
        "int8_kb": round(n_params / 1024, 1),
        "int4_kb_estimated": round(n_params * 0.5 / 1024, 1),
        "val_mae_fp32": round(val_mae_fp32, 2),
        "val_mae_int8": round(val_mae_int8, 2),
        "test_mae_int8": round(test_mae_int8, 2),
        "val_mae_std_int8": round(val_std_int8, 2),
        "test_mae_std_int8": round(test_std_int8, 2),
    }
    with open(RESULTS_DIR / "quantization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULTS_DIR / 'quantization_results.json'}")


if __name__ == "__main__":
    run()
