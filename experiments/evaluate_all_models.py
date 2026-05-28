"""
Unified evaluation of all thesis models on the corrected 247-sample test set.

Produces MAE ± std for val and test splits, plus per-set MAE for test.
Saves structured JSON results to experiments/results/all_models_eval.json.

Models evaluated (all SE-based, no CBAM):
  1. Image-only ResNet18 (unpruned, FP32)
  2. Pruned ResNet (2M / 1.5M / 1M) — FP32 distilled
  3. Pruned ResNet (2M / 1.5M / 1M) — QAT INT8
  4. Multiscale Sensor CNN (SE, FP32)
  5. Fusion pipeline:
     a. Phase 5 compressed fusion (2.29M, SE, FP32 PyTorch)
     b. Pruned+distilled fusion (1.13M, SE, FP32 PyTorch)
     c. ONNX INT8 (1.13M, from export_onnx.py)
     d. TFLite FP32 / INT8 (1.13M, from convert_tflite.py)

Usage:
    python experiments/evaluate_all_models.py

Run from the thesis root.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import MATWIDataset, SPLIT_MAP
from src.data.sensor_scalogram_dataset import MATWISensorScalogramDataset
from src.data.fusion_scalogram_dataset import MATWIFusionScalogramDataset
from src.models.image_model import build_resnet18_regressor
from src.models.multiscale_sensor_cnn import MultiScaleSensorCNN
from src.models.multiscale_fusion_model import MultiScaleFusionModel

DATA_ROOT     = ROOT / "data" / "raw"
SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
CKPT_DIR      = ROOT / "checkpoints"
RESULTS_DIR   = ROOT / "experiments" / "results"

BATCH_SIZE  = 16
NUM_WORKERS = 0

TEST_SETS = sorted(SPLIT_MAP["test"])   # [4, 9, 13]


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _per_set_breakdown(errors, sets):
    """Compute per-set MAE for test split."""
    per_set = {}
    for s in TEST_SETS:
        mask = sets == s
        if mask.sum() > 0:
            per_set[f"set_{s}"] = {
                "n":   int(mask.sum()),
                "mae": round(float(errors[mask].mean()), 2),
                "std": round(float(errors[mask].std()),  2),
            }
    return per_set


def _make_result(errors, n_samples, per_set=None):
    result = {
        "n_samples": n_samples,
        "mae":       round(float(errors.mean()), 2),
        "mae_std":   round(float(errors.std()),  2),
        "mae_min":   round(float(errors.min()),  2),
        "mae_max":   round(float(errors.max()),  2),
    }
    if per_set is not None:
        result["per_set"] = per_set
    return result


def eval_image_model(model, split, device):
    """Evaluate an image-only model."""
    ds     = MATWIDataset(DATA_ROOT, split=split)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            preds = model(images.to(device)).squeeze(1).cpu()
            all_preds.append(preds)
            all_targets.append(targets)

    preds   = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    errors  = np.abs(preds - targets)

    per_set = None
    if split == "test":
        labels = pd.read_csv(DATA_ROOT / "labels.csv")
        test_labels = labels[
            labels["Set"].isin(TEST_SETS) &
            labels["wear"].notna() &
            labels["ImageFile"].notna()
        ].reset_index(drop=True)
        per_set = _per_set_breakdown(errors, test_labels["Set"].values)

    return _make_result(errors, len(ds), per_set)


def _get_test_set_values(dataset_type="fusion"):
    """Get Set values for each test sample, aligned with dataset ordering."""
    labels  = pd.read_csv(DATA_ROOT / "labels.csv")
    labels["labels_idx"] = labels.index
    filt = labels["Set"].isin(TEST_SETS) & labels["wear"].notna()
    if dataset_type == "fusion":
        filt = filt & labels["ImageFile"].notna()
    labels_test = labels[filt]
    feats = pd.read_parquet(FEATURES_PATH)
    feats_test = feats[feats["Set"].isin(TEST_SETS)][["labels_idx"]].copy()
    merged = labels_test.merge(feats_test, on="labels_idx", how="inner")
    merged = merged.sort_values("labels_idx").reset_index(drop=True)
    return merged["Set"].values


def eval_sensor_model(model, split, device):
    """Evaluate a sensor-only model."""
    ds     = MATWISensorScalogramDataset(SCALOGRAM_DIR, FEATURES_PATH, split=split)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for scalograms, targets in loader:
            preds = model(scalograms.to(device)).squeeze(1).cpu()
            all_preds.append(preds)
            all_targets.append(targets)

    preds   = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    errors  = np.abs(preds - targets)

    per_set = None
    if split == "test":
        sets = _get_test_set_values("sensor")
        per_set = _per_set_breakdown(errors, sets)

    return _make_result(errors, len(ds), per_set)


def eval_fusion_model(model, split, device):
    """Evaluate a fusion model."""
    ds     = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, split)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=NUM_WORKERS)

    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for images, scalograms, targets in loader:
            preds = model(images.to(device), scalograms.to(device))[0].squeeze(1).cpu()
            all_preds.append(preds)
            all_targets.append(targets)

    preds   = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    errors  = np.abs(preds - targets)

    per_set = None
    if split == "test":
        sets = _get_test_set_values("fusion")
        per_set = _per_set_breakdown(errors, sets)

    return _make_result(errors, len(ds), per_set)


def eval_onnx_fusion(session, split):
    """Evaluate an ONNX Runtime fusion session."""
    ds     = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, split)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    preds_all, targets_all = [], []
    for imgs, scals, tgts in loader:
        out = session.run(
            None,
            {"image":     imgs.numpy().astype(np.float32),
             "scalogram": scals.numpy().astype(np.float32)},
        )[0]
        preds_all.append(torch.from_numpy(out).squeeze(1))
        targets_all.append(tgts)

    p = torch.cat(preds_all).numpy()
    t = torch.cat(targets_all).numpy()
    errors = np.abs(p - t)

    per_set = None
    if split == "test":
        sets = _get_test_set_values("fusion")
        per_set = _per_set_breakdown(errors, sets)

    return _make_result(errors, len(ds), per_set)


def eval_tflite_fusion(tflite_path, split):
    """Evaluate a TFLite fusion model (float I/O, integer_quant variant)."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_map = {d["name"]: d["index"] for d in input_details}

    ds = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    preds, targets = [], []
    for img, scalo, tgt in loader:
        img_nhwc   = img.numpy().transpose(0, 2, 3, 1).astype(np.float32)
        scalo_nhwc = scalo.numpy().transpose(0, 2, 3, 1).astype(np.float32)
        interpreter.set_tensor(input_map["image"],     img_nhwc)
        interpreter.set_tensor(input_map["scalogram"], scalo_nhwc)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]["index"])
        preds.append(float(out[0, 0]))
        targets.append(float(tgt[0]))

    p = np.array(preds)
    t = np.array(targets)
    errors = np.abs(p - t)

    per_set = None
    if split == "test":
        sets = _get_test_set_values("fusion")
        per_set = _per_set_breakdown(errors, sets)

    return _make_result(errors, len(ds), per_set)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def print_result(label, split, r):
    line = f"  {split:5s}  n={r['n_samples']:4d}  MAE={r['mae']:.2f} +/- {r['mae_std']:.2f} um"
    if split == "test" and "per_set" in r:
        per = r["per_set"]
        parts = [f"Set {k.split('_')[1]}: {v['mae']:.2f}+/-{v['std']:.2f}" for k, v in per.items()]
        line += f"  [{', '.join(parts)}]"
    print(line)


def section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate thesis models")
    parser.add_argument(
        "--only", nargs="+", metavar="KEY",
        help=(
            "Run only the specified model keys. Available keys:\n"
            "  image_resnet18_fp32, resnet_2m_fp32, resnet_1p5m_fp32, resnet_1m_fp32,\n"
            "  resnet_2m_qat, resnet_1p5m_qat, resnet_1m_qat,\n"
            "  sensor_se_fp32, fusion_11m_cbam_fp32, fusion_2m_se_fp32,\n"
            "  fusion_1m_se_fp32, fusion_1m_onnx_int8, fusion_tflite_fp32, fusion_tflite_int8"
        ),
    )
    args = parser.parse_args()
    only = set(args.only) if args.only else None

    def skip(key):
        return only is not None and key not in only

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}\n")

    torch.backends.quantized.engine = "qnnpack"

    all_results = {}

    # ── 1. Image-only baseline (Phase 1) ─────────────────────────────────────
    if not skip("image_resnet18_fp32"):
        section("1. Image-only ResNet18 (unpruned, 11.2M, FP32)")
        model = build_resnet18_regressor().to(device)
        model.load_state_dict(torch.load(CKPT_DIR / "phase1_best.pt", map_location=device))
        model.eval()
        n = count_params(model)
        print(f"   Parameters: {n:,}")
        entry = {"params": n, "int8_kb": round(n / 1024, 1)}
        for split in ("val", "test"):
            r = eval_image_model(model, split, device)
            entry[split] = r
            print_result("image_resnet18_fp32", split, r)
        all_results["image_resnet18_fp32"] = entry

    # ── 2. Pruned ResNet (FP32 distilled) ────────────────────────────────────
    resnet_variants = [
        ("resnet_2m_fp32",   "resnet_distilled_budget.pt", "Pruned ResNet 2M (budget, FP32)"),
        ("resnet_1p5m_fp32", "resnet_distilled_1p5m.pt",   "Pruned ResNet 1.5M (FP32)"),
        ("resnet_1m_fp32",   "resnet_distilled_1m.pt",     "Pruned ResNet 1M (FP32)"),
    ]
    for key, ckpt_name, label in resnet_variants:
        if skip(key):
            continue
        section(f"2. {label}")
        model = torch.load(CKPT_DIR / ckpt_name, map_location=device, weights_only=False)
        model.eval()
        n = count_params(model)
        print(f"   Parameters: {n:,}")
        entry = {"params": n, "int8_kb": round(n / 1024, 1)}
        for split in ("val", "test"):
            r = eval_image_model(model, split, device)
            entry[split] = r
            print_result(key, split, r)
        all_results[key] = entry

    # ── 3. Pruned ResNet (QAT INT8) ─────────────────────────────────────────
    qat_variants = [
        ("resnet_2m_qat",   "resnet_qat_int8_budget.pt", "Pruned ResNet 2M (budget, QAT INT8)"),
        ("resnet_1p5m_qat", "resnet_qat_int8_1p5m.pt",   "Pruned ResNet 1.5M (QAT INT8)"),
        ("resnet_1m_qat",   "resnet_qat_int8_1m.pt",     "Pruned ResNet 1M (QAT INT8)"),
    ]
    for key, ckpt_name, label in qat_variants:
        if skip(key):
            continue
        section(f"3. {label}")
        model = torch.load(CKPT_DIR / ckpt_name, map_location="cpu", weights_only=False)
        model.eval()
        n = count_params(model)
        print(f"   Parameters: {n:,}")
        entry = {"params": n, "int8_kb": round(n / 1024, 1)}
        for split in ("val", "test"):
            r = eval_image_model(model, split, "cpu")
            entry[split] = r
            print_result(key, split, r)
        all_results[key] = entry

    # ── 4. Sensor CNN (SE, FP32) ────────────────────────────────────────────
    if not skip("sensor_se_fp32"):
        section("4. Multiscale Sensor CNN (SE, 244K, FP32)")
        model = MultiScaleSensorCNN(attention="se").to(device)
        model.load_state_dict(torch.load(
            CKPT_DIR / "phase4_multiscale_sgdm_best.pt", map_location=device
        ))
        model.eval()
        n = count_params(model)
        print(f"   Parameters: {n:,}")
        entry = {"params": n, "int8_kb": round(n / 1024, 1)}
        for split in ("val", "test"):
            r = eval_sensor_model(model, split, device)
            entry[split] = r
            print_result("sensor_se_fp32", split, r)
        all_results["sensor_se_fp32"] = entry

    # ── 5. Fusion — 11.5M CBAM FP32 (Phase 5 full) ────────────────────────
    full_fusion_ckpt = CKPT_DIR / "phase5_multiscale_fusion_best.pt"
    if full_fusion_ckpt.exists() and not skip("fusion_11m_cbam_fp32"):
        section("5. Full fusion (11.5M, CBAM, FP32 PyTorch)")
        model = MultiScaleFusionModel()
        model.sensor_cnn = MultiScaleSensorCNN(attention="cbam")
        model = model.to(device)
        model.load_state_dict(torch.load(full_fusion_ckpt, map_location=device, weights_only=True))
        model.eval()
        n = count_params(model)
        n_img = sum(p.numel() for p in model.image_encoder.parameters())
        n_sen = sum(p.numel() for p in model.sensor_cnn.parameters())
        print(f"   Total params   : {n:,}")
        print(f"   Image encoder  : {n_img:,}")
        print(f"   Sensor encoder : {n_sen:,}")
        print(f"   Fusion head    : {n - n_img - n_sen:,}")
        entry = {"params": n, "params_image": n_img, "params_sensor": n_sen,
                 "attention": "cbam"}
        for split in ("val", "test"):
            r = eval_fusion_model(model, split, device)
            entry[split] = r
            print_result("fusion_11m_cbam_fp32", split, r)
        all_results["fusion_11m_cbam_fp32"] = entry

    # ── 5a. Fusion — 2.29M SE FP32 (Phase 5 compressed) ────────────────────
    compressed_ckpt = CKPT_DIR / "phase5_compressed_fusion_best.pt"
    resnet_ckpt     = CKPT_DIR / "resnet_distilled_budget.pt"
    if compressed_ckpt.exists() and resnet_ckpt.exists():
        section("5a. Compressed fusion (2.29M, SE, FP32 PyTorch)")
        model = MultiScaleFusionModel(image_feat_dim=309)
        model.load_compressed_image_encoder(str(resnet_ckpt), device=device)
        model.load_state_dict(torch.load(compressed_ckpt, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
        n = count_params(model)
        n_img = sum(p.numel() for p in model.image_encoder.parameters())
        n_sen = sum(p.numel() for p in model.sensor_cnn.parameters())
        print(f"   Total params   : {n:,}")
        print(f"   Image encoder  : {n_img:,}")
        print(f"   Sensor encoder : {n_sen:,}")
        print(f"   Fusion head    : {n - n_img - n_sen:,}")
        entry = {"params": n, "int8_kb": round(n / 1024, 1),
                 "params_image": n_img, "params_sensor": n_sen}
        for split in ("val", "test"):
            r = eval_fusion_model(model, split, device)
            entry[split] = r
            print_result("fusion_2m_se_fp32", split, r)
        all_results["fusion_2m_se_fp32"] = entry

    # ── 5b. Fusion — 1.13M SE FP32 (pruned+distilled) ──────────────────────
    distilled_ckpt = CKPT_DIR / "fusion_distilled.pt"
    if distilled_ckpt.exists():
        section("5b. Pruned+distilled fusion (1.13M, SE, FP32 PyTorch)")
        model = torch.load(distilled_ckpt, map_location=device, weights_only=False)
        model = model.to(device).eval()
        n = count_params(model)
        print(f"   Parameters: {n:,}")
        entry = {"params": n, "int8_kb": round(n / 1024, 1)}
        for split in ("val", "test"):
            r = eval_fusion_model(model, split, device)
            entry[split] = r
            print_result("fusion_1m_se_fp32", split, r)
        all_results["fusion_1m_se_fp32"] = entry

    # ── 5c. Fusion — 1.13M ONNX INT8 ───────────────────────────────────────
    int8_onnx = CKPT_DIR / "fusion_int8.onnx"
    if int8_onnx.exists():
        try:
            import onnxruntime as ort
            section("5c. Fusion ONNX INT8 (1.13M, static quant)")
            size_kb = int8_onnx.stat().st_size / 1024
            print(f"   File size: {size_kb:.1f} KB")
            session = ort.InferenceSession(str(int8_onnx), providers=["CPUExecutionProvider"])
            entry = {"file_size_kb": round(size_kb, 1)}
            for split in ("val", "test"):
                r = eval_onnx_fusion(session, split)
                entry[split] = r
                print_result("fusion_1m_onnx_int8", split, r)
            all_results["fusion_1m_onnx_int8"] = entry
        except ImportError:
            print("   onnxruntime not installed — skipping")

    # ── 5d. Fusion — 1.13M TFLite (FP32 and INT8) ──────────────────────────
    tflite_dir = CKPT_DIR / "fusion_tflite"
    tflite_models = [
        ("fusion_tflite_fp32", tflite_dir / "fusion_fp32_dedup_float32.tflite", "TFLite FP32"),
        ("fusion_tflite_int8", tflite_dir / "fusion_fp32_dedup_integer_quant.tflite", "TFLite INT8"),
    ]
    for key, path, label in tflite_models:
        if not path.exists():
            continue
        try:
            section(f"5d. Fusion {label} (1.13M, onnx2tf)")
            size_kb = path.stat().st_size / 1024
            print(f"   File size: {size_kb:.1f} KB")
            entry = {"file_size_kb": round(size_kb, 1)}
            for split in ("val", "test"):
                r = eval_tflite_fusion(str(path), split)
                entry[split] = r
                print_result(key, split, r)
            all_results[key] = entry
        except ImportError:
            print("   tensorflow not installed — skipping TFLite eval")

    # ── 6. Model file sizes ─────────────────────────────────────────────────
    section("6. Model file sizes")
    size_entries = {}
    for fname, label in [
        ("phase5_compressed_fusion_best.pt", "Fusion 2.29M FP32 .pt"),
        ("fusion_distilled.pt",              "Fusion 1.13M FP32 .pt (pruned)"),
        ("fusion_fp32.onnx",                 "Fusion 1.13M ONNX FP32"),
        ("fusion_int8.onnx",                 "Fusion 1.13M ONNX INT8"),
    ]:
        p = CKPT_DIR / fname
        if p.exists():
            kb = p.stat().st_size / 1024
            print(f"   {label:50s} {kb:8.1f} KB")
            size_entries[fname] = round(kb, 1)
    if tflite_dir.exists():
        for f in sorted(tflite_dir.glob("*.tflite")):
            kb = f.stat().st_size / 1024
            print(f"   {f.name:50s} {kb:8.1f} KB")
            size_entries[f"fusion_tflite/{f.name}"] = round(kb, 1)
    all_results["file_sizes_kb"] = size_entries

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "all_models_eval.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: bool(x) if isinstance(x, np.bool_) else x)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
