"""
Static INT8 Quantization + ONNX Export for the Distilled Fusion Model.

Converts fusion_distilled.pt (or fusion_distilled_qat.pt) to a deployment-ready
INT8 ONNX model with calibrated INT8 activations, resolving the RAM constraint
that blocked dynamic-quant deployment on the NXP FRDM-MCXN947.

Why static quantization rather than dynamic
-------------------------------------------
Dynamic quantization keeps activations FP32. At 224×224 input, the first
convolutional feature map alone (13×112×112 FP32) is 637 KB — exceeding the
512 KB SRAM ceiling. Static quantization calibrates per-tensor INT8 scales
for activations, shrinking that same map to 159 KB:

  Peak RAM (dynamic, FP32 acts) : ~1,289 KB   ✗ 2.5× over budget
  Peak RAM (static,  INT8 acts) :   ~370 KB   ✓ fits with 142 KB headroom

Why ONNX Runtime quantization (not PyTorch static quant)
---------------------------------------------------------
PyTorch's quantized ops (torch.ops.quantized.*) do not export to standard
ONNX. ONNX Runtime post-training static quantization writes QDQ
(QuantizeLinear / DequantizeLinear) nodes — the format accepted by:
  - NXP eIQ Model Tool / Neutron NPU backend
  - ONNX Runtime on-device inference
  - onnx2tf → TFLite if needed
  - TensorRT INT8 (future)

Layers that cannot be INT8 (GroupNorm, GELU, Sigmoid) are left FP32
automatically by the calibration pass — no manual exclusion list needed.

Note on the QAT variant (--output-suffix _qat)
----------------------------------------------
fusion_distilled_qat.pt embeds a QAT INT8 image encoder. torch.onnx.export
will represent those layers as QLinear/QDQ nodes in the ONNX graph. ONNX
Runtime's quantize_static pass will then calibrate the remaining FP32 layers
(sensor CNN + fusion head), producing a fully INT8 ONNX model.

Pipeline
--------
  1. Load fusion_distilled[_qat].pt  →  verify FP32/INT8 accuracy on CPU
  2. Wrap model (drop auxiliary output)  →  export ONNX (opset 18)
  3. Verify ONNX vs PyTorch output diff  (sanity check)
  4. Calibrate on N training samples     →  compute per-tensor INT8 scales
  5. Write INT8 QDQ ONNX
  6. Evaluate INT8 accuracy on val + test via ONNX Runtime
  7. Print RAM analysis + baseline comparisons
  8. Save results JSON + both ONNX files

Outputs
-------
  fusion/deployment/checkpoints/fusion_fp32[suffix].onnx   FP32 reference
  fusion/deployment/checkpoints/fusion_int8[suffix].onnx   INT8 QDQ (deployable)
  fusion/two_tower/compression/static_quant/results/static_quant_results[suffix].json

Pre-requisite
-------------
    python fusion/two_tower/compression/pruning/distill.py [--output-suffix _qat]
    pip install onnx onnxruntime

Usage
-----
    # Non-QAT pipeline (default)
    python fusion/two_tower/compression/static_quant/export_onnx.py

    # QAT pipeline
    python fusion/two_tower/compression/static_quant/export_onnx.py \\
        --model-ckpt fusion/two_tower/compression/pruning/checkpoints/fusion_distilled_qat.pt \\
        --output-suffix _qat

Run from the thesis root.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from fusion.two_tower.dataset import MATWIFusionScalogramDataset
from src.metrics import mae

DATA_ROOT        = ROOT / "data" / "raw"
SCALOGRAM_DIR    = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH    = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
PRUNING_CKPT_DIR = Path(__file__).parents[1] / "pruning" / "checkpoints"
DEPLOY_CKPT_DIR  = ROOT / "fusion" / "deployment" / "checkpoints"   # ONNX outputs
RESULTS_DIR      = Path(__file__).parent / "results"

OPSET_VERSION         = 18    # torch.onnx >= 2.0 targets opset 18 by default;
                              # NXP eIQ Model Tool accepts opset 7–18
DEFAULT_CALIB_SAMPLES = 200   # ~30% of training set; sufficient for PTQ
BATCH_SIZE            = 16
NUM_WORKERS           = 0


# ── GELU → tanh-GELU swap (TFLite-Micro deployability) ─────────────────────────

def swap_gelu_to_tanh(module: nn.Module) -> int:
    """
    Recursively replace nn.GELU(approximate='none') with approximate='tanh'.

    The default ('none') GELU exports an `Erf` op that TFLite has no native
    kernel for, forcing 3 FlexErf (TF-Select) ops that TFLite-Micro cannot run.
    The tanh approximation lowers to a native TANH. The two activations are
    numerically equivalent to ~3e-4 on the tiny post-LayerNorm projection
    vectors here, so accuracy is unchanged (verified by check_gelu_swap.py:
    test MAE 20.48 µm identical before/after).
    """
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.GELU) and child.approximate == "none":
            setattr(module, name, nn.GELU(approximate="tanh"))
            n += 1
        else:
            n += swap_gelu_to_tanh(child)
    return n


# ── ONNX export wrapper ────────────────────────────────────────────────────────

class FusionExportWrapper(nn.Module):
    """
    Thin wrapper that drops the auxiliary regression head output so ONNX
    sees a single (batch, 1) output tensor named 'wear_depth_um'.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, scalogram: torch.Tensor) -> torch.Tensor:
        final, _ = self.model(image, scalogram)
        return final   # (B, 1)


# ── Calibration data reader ───────────────────────────────────────────────────

def build_calibration_reader(dataset, n_samples: int):
    """
    Build an ONNX Runtime CalibrationDataReader from the training split.
    Feeds normalized float32 tensors — same preprocessing as training.
    """
    from onnxruntime.quantization import CalibrationDataReader

    class FusionCalibReader(CalibrationDataReader):
        def __init__(self):
            loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
            self.samples = []
            for i, (img, scalo, _) in enumerate(loader):
                if i >= n_samples:
                    break
                self.samples.append({
                    "image":     img.numpy().astype(np.float32),
                    "scalogram": scalo.numpy().astype(np.float32),
                })
            self.idx = 0
            print(f"  Calibration dataset : {len(self.samples)} samples")

        def get_next(self):
            if self.idx >= len(self.samples):
                return None
            feed = self.samples[self.idx]
            self.idx += 1
            return feed

    return FusionCalibReader()


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def evaluate_torch(model, device, split):
    """Evaluate a PyTorch model; returns dict with mae, std, n."""
    ds = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, split)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, scals, tgts in loader:
            out = model(imgs.to(device), scals.to(device))[0].squeeze(1).cpu()
            preds.append(out)
            targets.append(tgts)
    p, t = torch.cat(preds), torch.cat(targets)
    errs = (p - t).abs()
    return {"mae": round(errs.mean().item(), 2),
            "std": round(errs.std().item(),  2),
            "n":   len(ds)}


def evaluate_onnx(session, split):
    """Evaluate an ONNX Runtime session on a dataset split."""
    ds = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, split)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    preds, targets = [], []
    for imgs, scals, tgts in loader:
        out = session.run(
            ["wear_depth_um"],
            {"image":     imgs.numpy().astype(np.float32),
             "scalogram": scals.numpy().astype(np.float32)},
        )[0]                          # ndarray (B, 1)
        preds.append(torch.from_numpy(out).squeeze(1))
        targets.append(tgts)
    p, t = torch.cat(preds), torch.cat(targets)
    errs = (p - t).abs()
    return {"mae": round(errs.mean().item(), 2),
            "std": round(errs.std().item(),  2),
            "n":   len(ds)}


# ── RAM analysis ──────────────────────────────────────────────────────────────

def peak_ram_kb(act_bytes_per_elem: int) -> dict:
    """
    Estimate worst-case peak RAM during inference at the first conv layer.
    Worst case: input image + conv1 output both live in SRAM simultaneously.

    Compressed ResNet first layer: 13 output channels, stride-2 → 112×112.
    Stack/heap estimate: 64 KB bare metal.
    """
    img_kb   = 224 * 224 * 3 * act_bytes_per_elem / 1024
    conv1_kb = 13  * 112 * 112 * act_bytes_per_elem / 1024
    peak_kb  = img_kb + conv1_kb + 64      # +64 KB stack/heap
    return {
        "img_kb":    round(img_kb,   1),
        "conv1_kb":  round(conv1_kb, 1),
        "peak_kb":   round(peak_kb,  1),
        "budget_kb": 512,
        "fits":      peak_kb <= 512,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run(args):
    try:
        import onnx
        import onnxruntime as ort
        from onnxruntime.quantization import (
            quantize_static, QuantFormat, QuantType,
        )
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\n  pip install onnx onnxruntime")

    # ONNX export runs on CPU; PyTorch QAT INT8 ops require QuantizedCPU.
    # torch.onnx.export always traces on CPU for quantized models anyway.
    device = "cpu"
    torch.backends.quantized.engine = "qnnpack"
    print(f"Device : {device}\n")

    model_ckpt = Path(args.model_ckpt)
    if not model_ckpt.exists():
        sys.exit(f"Checkpoint not found: {model_ckpt}\n"
                 "Run fusion/two_tower/compression/pruning/distill.py first.")

    suffix       = args.output_suffix
    fp32_onnx    = DEPLOY_CKPT_DIR / f"fusion_fp32{suffix}.onnx"
    int8_onnx    = DEPLOY_CKPT_DIR / f"fusion_int8{suffix}.onnx"
    results_name = f"static_quant_results{suffix}.json"

    # ── Load FP32 distilled model ─────────────────────────────────────────────
    print(f"Loading : {model_ckpt.name}")
    model = torch.load(model_ckpt, map_location=device, weights_only=False)
    model = model.to(device).eval()

    # Swap erf-GELU -> tanh-GELU so the exported graph carries native TANH
    # instead of Erf (which becomes a non-TFLM FlexErf op). Accuracy-neutral.
    n_swapped = swap_gelu_to_tanh(model)
    print(f"GELU    : swapped {n_swapped} erf-GELU -> tanh-GELU (native TFLite TANH)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params  : {n_params:,}  ({n_params / 1024:.0f} KB INT8 weights)\n")

    # ── FP32 baseline ─────────────────────────────────────────────────────────
    print("── FP32 evaluation (PyTorch baseline) ──────────────────────────────")
    fp32_val  = evaluate_torch(model, device, "val")
    fp32_test = evaluate_torch(model, device, "test")
    print(f"  val   n={fp32_val['n']}  MAE={fp32_val['mae']:.2f} ± {fp32_val['std']:.2f} µm")
    print(f"  test  n={fp32_test['n']}  MAE={fp32_test['mae']:.2f} ± {fp32_test['std']:.2f} µm")

    # ── ONNX export (FP32) ────────────────────────────────────────────────────
    print(f"\n── Exporting FP32 ONNX (opset {OPSET_VERSION}) ─────────────────────────")
    wrapper     = FusionExportWrapper(model).to(device).eval()
    dummy_img   = torch.randn(1, 3, 224, 224, device=device)
    dummy_scalo = torch.randn(1, 5, 64,  64,  device=device)

    DEPLOY_CKPT_DIR.mkdir(exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            wrapper,
            (dummy_img, dummy_scalo),
            str(fp32_onnx),
            input_names    = ["image", "scalogram"],
            output_names   = ["wear_depth_um"],
            dynamic_axes   = {
                "image":         {0: "batch"},
                "scalogram":     {0: "batch"},
                "wear_depth_um": {0: "batch"},
            },
            opset_version       = OPSET_VERSION,
            do_constant_folding = True,
        )
    onnx.checker.check_model(str(fp32_onnx))
    fp32_onnx_kb = fp32_onnx.stat().st_size / 1024
    print(f"  Saved  : {fp32_onnx.name}  ({fp32_onnx_kb:.0f} KB)")

    # Sanity: ONNX output must match PyTorch within floating-point noise
    sess_fp32 = ort.InferenceSession(str(fp32_onnx), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt_out = wrapper(dummy_img, dummy_scalo).cpu().numpy()
    onnx_out = sess_fp32.run(
        ["wear_depth_um"],
        {"image":     dummy_img.cpu().numpy().astype(np.float32),
         "scalogram": dummy_scalo.cpu().numpy().astype(np.float32)},
    )[0]
    max_diff = float(np.abs(pt_out - onnx_out).max())
    status   = "✓" if max_diff < 0.01 else "⚠ large diff — check export"
    print(f"  Sanity : PyTorch vs ONNX max diff = {max_diff:.6f} µm  {status}")

    # ── Static INT8 calibration ───────────────────────────────────────────────
    print(f"\n── Static INT8 quantization (calibrating on {args.calib_samples} samples) ──")
    train_ds     = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, "train")
    calib_reader = build_calibration_reader(train_ds, args.calib_samples)

    # Per-tensor symmetric INT8 — most compatible with MCU backends.
    # Non-quantizable layers (GroupNorm, GELU, Sigmoid) are left FP32
    # automatically by ONNX Runtime's op-support check.
    quantize_static(
        model_input             = str(fp32_onnx),
        model_output            = str(int8_onnx),
        calibration_data_reader = calib_reader,
        quant_format            = QuantFormat.QDQ,   # standard QDQ nodes
        per_channel             = False,             # per-tensor: MCU-safe
        weight_type             = QuantType.QInt8,
        activation_type         = QuantType.QInt8,
        extra_options           = {
            "ActivationSymmetric": True,   # zero_point = 0 where possible
            "WeightSymmetric":     True,
        },
    )
    int8_onnx_kb = int8_onnx.stat().st_size / 1024
    print(f"  Saved  : {int8_onnx.name}  ({int8_onnx_kb:.0f} KB)")

    # ── INT8 evaluation ───────────────────────────────────────────────────────
    print("\n── INT8 ONNX evaluation (ONNX Runtime) ─────────────────────────────")
    sess_int8 = ort.InferenceSession(str(int8_onnx), providers=["CPUExecutionProvider"])
    int8_val  = evaluate_onnx(sess_int8, "val")
    int8_test = evaluate_onnx(sess_int8, "test")
    print(
        f"  val   n={int8_val['n']}  MAE={int8_val['mae']:.2f} ± {int8_val['std']:.2f} µm"
        f"  (FP32: {fp32_val['mae']:.2f}  Δ={int8_val['mae'] - fp32_val['mae']:+.2f})"
    )
    print(
        f"  test  n={int8_test['n']}  MAE={int8_test['mae']:.2f} ± {int8_test['std']:.2f} µm"
        f"  (FP32: {fp32_test['mae']:.2f}  Δ={int8_test['mae'] - fp32_test['mae']:+.2f})"
    )

    # ── RAM analysis ──────────────────────────────────────────────────────────
    ram_dyn  = peak_ram_kb(act_bytes_per_elem=4)   # dynamic quant (phase3)
    ram_stat = peak_ram_kb(act_bytes_per_elem=1)   # static quant  (this)

    print("\n── Peak RAM at first conv (224×224 input, NXP 512 KB budget) ───────")
    print(f"  Dynamic INT8 (FP32 activations) : img {ram_dyn['img_kb']:.0f} KB"
          f" + conv1 {ram_dyn['conv1_kb']:.0f} KB = {ram_dyn['peak_kb']:.0f} KB"
          f"  {'✓' if ram_dyn['fits'] else '✗ OVER budget'}")
    print(f"  Static  INT8 (INT8 activations) : img {ram_stat['img_kb']:.0f} KB"
          f" + conv1 {ram_stat['conv1_kb']:.0f} KB = {ram_stat['peak_kb']:.0f} KB"
          f"  {'✓' if ram_stat['fits'] else '✗ OVER budget'}")

    # ── Baseline comparison ───────────────────────────────────────────────────
    print("\n── Baseline comparison ──────────────────────────────────────────────")
    print(f"  Paper ResNet50 (image-only)                      test MAE : 19.00 µm")
    print(f"  Compressed QAT fusion INT8 ONNX (QAT pipeline)  test MAE : 20.63 µm  ✓ deployable")
    print(f"  Non-QAT pipeline INT8 ONNX (fusion_int8.onnx)   test MAE : 34.80 µm  ✓ deployable (older/worse run)")
    print(f"  Compressed FP32 fusion (phase5, FP32)            test MAE : 15.55 µm")
    print(f"  Phase 1 image-only (ResNet18)                    test MAE : 23.17 µm")
    print(f"  Phase 4 sensor-only (MultiScaleCNN SE)           test MAE : 29.27 µm")
    print(f"  This run static INT8 (INT8 acts, .onnx)          test MAE : {int8_test['mae']:.2f} µm"
          f"  {'✓ deployable' if ram_stat['fits'] else '✗ check RAM'}")

    # ── Save results ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "model_ckpt":          str(model_ckpt),
        "fp32_onnx":           str(fp32_onnx),
        "int8_onnx":           str(int8_onnx),
        "opset_version":       OPSET_VERSION,
        "calib_samples":       args.calib_samples,
        "n_params":            n_params,
        "fp32_onnx_kb":        round(fp32_onnx_kb, 1),
        "int8_onnx_kb":        round(int8_onnx_kb, 1),
        "fp32": {"val_mae": fp32_val["mae"],  "val_std": fp32_val["std"],
                 "test_mae": fp32_test["mae"], "test_std": fp32_test["std"]},
        "int8": {"val_mae": int8_val["mae"],  "val_std": int8_val["std"],
                 "test_mae": int8_test["mae"], "test_std": int8_test["std"]},
        "accuracy_drop": {
            "val":  round(int8_val["mae"]  - fp32_val["mae"],  2),
            "test": round(int8_test["mae"] - fp32_test["mae"], 2),
        },
        "ram": {
            "dynamic_quant_fp32_acts": ram_dyn,
            "static_quant_int8_acts":  ram_stat,
        },
    }
    out_path = RESULTS_DIR / results_name
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved → {out_path}")
    print(f"ONNX  → {fp32_onnx}")
    print(f"ONNX  → {int8_onnx}")
    print(
        "\nNext step: load fusion_int8.onnx into the NXP eIQ Model Tool\n"
        "           (MCUXpresso IDE → eIQ Toolkit → Import ONNX model)\n"
        "           to generate the C array for on-device deployment."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ckpt",    default=str(PRUNING_CKPT_DIR / "fusion_distilled.pt"),
                        help="Full-object distilled fusion model checkpoint "
                             "(use fusion_distilled_qat.pt for the QAT pipeline)")
    parser.add_argument("--output-suffix", default="",
                        help="Appended to output filenames, e.g. '_v2'")
    parser.add_argument("--calib-samples", type=int, default=DEFAULT_CALIB_SAMPLES,
                        help="Training samples used for INT8 calibration (default 200)")
    args = parser.parse_args()
    run(args)
