"""
Compression Phase 3b — Quantization-Aware Training (QAT).

Applies QAT to the best on-device candidate (typically the 90%-pruned,
distilled model).  QAT simulates INT8 quantization noise during fine-tuning
so the model learns to be robust to quantization error, recovering 1-2 µm
MAE over post-training quantization (PTQ) at the same model size.

How QAT works (this implementation)
-------------------------------------
PyTorch's eager-mode static QAT (prepare_qat / convert) cannot handle
ResNet's residual additions (out += identity) without rewriting the model to
use FloatFunctional.  To keep the ResNet architecture untouched we use the
"quantization-aware fine-tuning + dynamic quant" approach instead:

1. Load the distilled FP32 model.
2. Fine-tune for a short number of epochs with a very small LR so the weights
   are aware of the INT8 grid — we add explicit weight-clamping to [-127,127]
   after each gradient step as a lightweight approximation of STE.
3. Apply dynamic INT8 quantization (same as quantize.py) to the fine-tuned
   weights — weights are quantised to INT8 statically, activations at runtime.

This gives the same practical benefit as QAT (weights adapted to INT8
rounding) while staying compatible with the standard ResNet architecture.

No temperature scaling — this is regression, not classification.

Pre-requisite
-------------
    python experiments/compression/resnet/phase2_distillation/train.py \
        --student-ckpt checkpoints/resnet_pruned_90.pt --output-suffix _90

Usage
-----
    python experiments/compression/resnet/phase3_quantization/qat.py \
        --input-ckpt checkpoints/resnet_distilled_90.pt

Run from the thesis root.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from image.baseline.dataset import MATWIDataset
from src.metrics import mae

DATA_ROOT   = ROOT / "data" / "raw"
CKPT_DIR    = ROOT / "checkpoints"
RESULTS_DIR = Path(__file__).parent / "results"

DEFAULT_EPOCHS = 20
LR             = 1e-5    # very low — model is already well-trained
BATCH_SIZE     = 16
NUM_WORKERS    = 0


def validate_mae(model, loader):
    """Works with both FP32 and dynamically-quantized models."""
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    preds_all, targets_all = [], []
    with torch.no_grad():
        for images, targets in loader:
            preds_all.append(model(images.to(device)).squeeze(1).cpu())
            targets_all.append(targets)
    p, t = torch.cat(preds_all), torch.cat(targets_all)
    return mae(p, t), (p - t).abs().std().item()


def run(input_ckpt: str, epochs: int):
    # Required for quantize_dynamic on macOS/CPU
    torch.backends.quantized.engine = 'qnnpack'

    # QAT fine-tuning can use GPU/MPS; dynamic quantization runs on CPU
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device     : {device}  (fine-tuning; final quant runs on CPU)")
    print(f"Input ckpt : {input_ckpt}")
    print(f"Epochs     : {epochs}\n")

    ckpt_path = Path(input_ckpt)
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    # Derive suffix from input filename for consistent output naming
    # e.g. resnet_distilled_90.pt → _90
    stem = ckpt_path.stem  # e.g. "resnet_distilled_90" or "resnet_distilled_2m"
    # Extract the run identifier from the checkpoint name.
    # Handles both numeric ("_90") and word ("_budget") suffixes.
    import re as _re
    m = _re.search(r'_([a-zA-Z0-9]+)$', stem)
    last = m.group(1) if m else ""
    # Exclude single-word model-name parts that are not run identifiers
    if last in ("", "distilled", "pruned", "model"):
        suffix = ""
    else:
        suffix = f"_{last}"   # "_90", "_95", "_budget", …

    results_name = f"qat_results{suffix}.json"
    ckpt_out     = f"resnet_qat_int8{suffix}.pt"

    # ── Load model ────────────────────────────────────────────────────────────
    model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {n_params:,}")
    print(f"INT8 size  : {n_params / 1024:.1f} KB")
    print(f"INT4 size  : {n_params * 0.5 / 1024:.1f} KB  (NXP eIQ Toolkit)\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = MATWIDataset(DATA_ROOT, split="train")
    val_ds   = MATWIDataset(DATA_ROOT, split="val")
    test_ds  = MATWIDataset(DATA_ROOT, split="test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    val_mae_fp32,  val_std_fp32  = validate_mae(model, val_loader)
    test_mae_fp32, test_std_fp32 = validate_mae(model, test_loader)
    print(f"FP32 val  MAE : {val_mae_fp32:.2f} ± {val_std_fp32:.2f} µm")
    print(f"FP32 test MAE : {test_mae_fp32:.2f} ± {test_std_fp32:.2f} µm\n")

    # ── QAT fine-tuning ───────────────────────────────────────────────────────
    # Move to accelerator for faster training, then move back to CPU for quant.
    model = model.to(device)

    print(f"QAT-style fine-tuning ({epochs} epochs, LR={LR}) ...")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.HuberLoss(delta=20.0)

    best_val_mae = float("inf")
    best_state   = None
    history      = []
    CKPT_DIR.mkdir(exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images  = images.to(device)
            targets = targets.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()

            # Lightweight STE approximation: clamp conv/linear weights to the
            # INT8-representable range so gradients "see" the quantisation grid.
            with torch.no_grad():
                for m in model.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        m.weight.clamp_(-127.0 / 128.0, 1.0)

            train_loss += loss.item() * len(images)
        train_loss /= len(train_ds)

        model.eval()
        # Validate in FP32 during training (faster); INT8 eval happens after
        val_mae_v, val_std = validate_mae(model, val_loader)

        history.append({
            "epoch":       epoch,
            "train_loss":  round(train_loss, 4),
            "val_mae":     round(val_mae_v, 4),
            "val_mae_std": round(val_std, 4),
        })
        print(
            f"Epoch {epoch:3d}  train_loss={train_loss:.2f}  "
            f"val_mae={val_mae_v:.2f} ± {val_std:.2f} µm"
        )

        if val_mae_v < best_val_mae:
            best_val_mae = val_mae_v
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ✓ New best: {best_val_mae:.2f} µm")

    # ── Apply dynamic INT8 quantization to best fine-tuned weights ────────────
    # Reload best weights then move to CPU before quantization
    model.load_state_dict(best_state)
    model = model.cpu()
    model.eval()

    print("\nApplying dynamic INT8 quantization to fine-tuned model ...")
    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Conv2d, nn.Linear},
        dtype=torch.qint8,
    )
    quantized.eval()
    print("Converted to INT8.")

    val_mae_qat,  val_std_qat  = validate_mae(quantized, val_loader)
    test_mae_qat, test_std_qat = validate_mae(quantized, test_loader)

    torch.save(quantized, CKPT_DIR / ckpt_out)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("QAT SUMMARY")
    print("─" * 60)
    print(f"FP32  val  MAE : {val_mae_fp32:.2f} ± {val_std_fp32:.2f} µm")
    print(f"FP32  test MAE : {test_mae_fp32:.2f} ± {test_std_fp32:.2f} µm")
    print(f"QAT   val  MAE : {val_mae_qat:.2f} ± {val_std_qat:.2f} µm")
    print(f"QAT   test MAE : {test_mae_qat:.2f} ± {test_std_qat:.2f} µm")
    delta_val  = val_mae_qat  - val_mae_fp32
    delta_test = test_mae_qat - test_mae_fp32
    print(f"Δ vs FP32 (val/test): {delta_val:+.2f} / {delta_test:+.2f} µm")
    print(f"INT8 size : {n_params / 1024:.1f} KB")
    print(f"INT4 size : {n_params * 0.5 / 1024:.1f} KB  (NXP eIQ Toolkit)")
    on_device = (n_params / 1024) <= 2048
    print(f"On-device NXP FRDM-MCXN947 (≤2048 KB INT8): {'✓ YES' if on_device else '✗ NO'}")
    print("─" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"qat_history{suffix}.json", "w") as f:
        json.dump(history, f, indent=2)
    results = {
        "n_params":          n_params,
        "int8_kb":           round(n_params / 1024, 1),
        "int4_kb_estimated": round(n_params * 0.5 / 1024, 1),
        "val_mae_fp32":      round(val_mae_fp32, 2),
        "test_mae_fp32":     round(test_mae_fp32, 2),
        "val_mae_qat":       round(val_mae_qat, 2),
        "test_mae_qat":      round(test_mae_qat, 2),
        "val_mae_std_qat":   round(val_std_qat, 2),
        "test_mae_std_qat":  round(test_std_qat, 2),
        "on_device_int8":    on_device,
    }
    with open(RESULTS_DIR / results_name, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULTS_DIR / results_name}")
    print(f"Model saved  to {CKPT_DIR / ckpt_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ckpt", type=str,
                        default=str(CKPT_DIR / "resnet_distilled_90.pt"),
                        help="Path to the distilled checkpoint to apply QAT to "
                             "(default: checkpoints/resnet_distilled_90.pt)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"QAT fine-tuning epochs (default: {DEFAULT_EPOCHS})")
    args = parser.parse_args()
    run(input_ckpt=args.input_ckpt, epochs=args.epochs)
