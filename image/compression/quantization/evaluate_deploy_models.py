"""
Deployment-model evaluation: val + test MAE ± std for the image-only ResNet18
budget models, FP32 reference vs ready-to-deploy full-static INT8 TFLite.

For each budget (2m / 1p5m / 1m) and each source (QAT vs non-QAT / distilled)
this reports the two variants that matter for the thesis:

  • FP32 PyTorch model
      QAT     : resnet_qat_int8_{budget}.pt   (dequantize_fc → pure FP32)
      non-QAT : resnet_distilled_{budget}.pt
  • INT8 full-static TFLite, INT8 I/O — the artifact that ships to the MCU
      QAT     : resnet_{budget}_qat_int8_nxp_io.tflite
      non-QAT : resnet_{budget}_noqat_int8_nxp_io.tflite

"std" is the standard deviation of the per-sample absolute error |pred - true|
(same convention as qat.py / the export scripts), so each cell reads as
"MAE ± std" in micrometres.

Pre-requisite: run export_onnx_static.py and export_onnx_static_noqat.py first,
so the INT8-I/O TFLite files exist.

Usage
-----
    python image/compression/quantization/evaluate_deploy_models.py
    python image/compression/quantization/evaluate_deploy_models.py --budgets 2m

Run from the thesis root.
"""

import argparse
import json

import torch

from export_onnx_static import (
    CKPT_DIR,
    RESULTS_DIR,
    dequantize_fc,
    evaluate_tflite,
    evaluate_torch,
)


def eval_fp32(ckpt_name):
    """Load a full-object FP32/QAT checkpoint and evaluate val + test."""
    ckpt = CKPT_DIR / ckpt_name
    if not ckpt.exists():
        return None
    torch.backends.quantized.engine = "qnnpack"
    model = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = dequantize_fc(model)          # no-op for distilled; FP32-izes QAT fc
    model = model.eval()
    return {
        "val":  evaluate_torch(model, "cpu", "val"),
        "test": evaluate_torch(model, "cpu", "test"),
    }


def eval_tflite_io(tflite_name):
    """Evaluate a full-integer INT8-I/O TFLite model on val + test."""
    path = CKPT_DIR / tflite_name
    if not path.exists():
        return None
    return {
        "val":  evaluate_tflite(path, "val",  int8_io=True),
        "test": evaluate_tflite(path, "test", int8_io=True),
    }


def process_budget(budget):
    sources = {
        "qat": {
            "fp32_ckpt":  f"resnet_qat_int8_{budget}.pt",
            "tflite_io":  f"resnet_{budget}_qat_int8_nxp_io.tflite",
        },
        "noqat": {
            "fp32_ckpt":  f"resnet_distilled_{budget}.pt",
            "tflite_io":  f"resnet_{budget}_noqat_int8_nxp_io.tflite",
        },
    }

    out = {}
    for src, paths in sources.items():
        label = "QAT" if src == "qat" else "non-QAT"
        print(f"\n  ── {label} ──────────────────────────────────────────")

        fp32 = eval_fp32(paths["fp32_ckpt"])
        if fp32 is None:
            print(f"    FP32        : SKIP (missing {paths['fp32_ckpt']})")
        else:
            print(f"    FP32        : "
                  f"val {fp32['val']['mae']:.2f} ± {fp32['val']['std']:.2f}   "
                  f"test {fp32['test']['mae']:.2f} ± {fp32['test']['std']:.2f}  µm")

        tfl = eval_tflite_io(paths["tflite_io"])
        if tfl is None:
            print(f"    INT8 TFLite : SKIP (missing {paths['tflite_io']})")
        else:
            print(f"    INT8 TFLite : "
                  f"val {tfl['val']['mae']:.2f} ± {tfl['val']['std']:.2f}   "
                  f"test {tfl['test']['mae']:.2f} ± {tfl['test']['std']:.2f}  µm")

        out[src] = {"fp32": fp32, "int8_tflite_io": tfl}
    return out


def fmt(entry, variant, split):
    """'mae ± std' or '—' if the variant is missing."""
    v = entry.get(variant)
    if v is None:
        return "—"
    return f"{v[split]['mae']:.2f} ± {v[split]['std']:.2f}"


def main():
    parser = argparse.ArgumentParser(
        description="val/test MAE ± std for FP32 and INT8-I/O TFLite image models")
    parser.add_argument("--budgets", nargs="+", default=["2m", "1p5m", "1m"])
    args = parser.parse_args()

    all_results = {}
    for budget in args.budgets:
        print(f"\n{'═' * 70}")
        print(f"  BUDGET: {budget}")
        print(f"{'═' * 70}")
        all_results[budget] = process_budget(budget)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'═' * 88}")
    print("  SUMMARY — MAE ± std (µm)")
    print(f"{'═' * 88}")
    print(f"{'Budget':<7} {'Source':<8} {'Variant':<12} "
          f"{'Val MAE ± std':>16} {'Test MAE ± std':>16}")
    print("─" * 88)
    for budget, srcs in all_results.items():
        for src in ("noqat", "qat"):
            entry = srcs.get(src, {})
            label = "QAT" if src == "qat" else "non-QAT"
            for variant, vname in (("fp32", "FP32"), ("int8_tflite_io", "INT8 TFLite")):
                print(f"{budget:<7} {label:<8} {vname:<12} "
                      f"{fmt(entry, variant, 'val'):>16} {fmt(entry, variant, 'test'):>16}")
        print("─" * 88)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "deploy_eval_mae_std.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
