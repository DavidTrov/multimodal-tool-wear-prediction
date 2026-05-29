"""
Joint Fusion Pruning — structured channel pruning of the full fusion pipeline.

Prunes the compressed image encoder, sensor CNN, and projection towers jointly
using the fusion MAE as the loss signal. This is superior to sequential
single-modality pruning because:

  - Channels are removed based on their value to the FUSION prediction, not to
    standalone per-modality prediction.
  - Image encoder channels that are redundant given sensor features (and vice
    versa) are identified and removed — impossible with sequential pruning.
  - Projection tower dimensions (img_proj: 309→128, sen_proj: 96→128) are
    co-optimised with both encoder outputs.

Architecture handled
--------------------
  image_encoder : compressed ResNet (pruned, non-standard channel widths)
  sensor_cnn    : MultiScaleSensorCNN (GroupNorm, inception entry, CBAM, ResBlock)
  image_norm / sensor_norm : LayerNorm — updated automatically by dependency graph
  img_proj / sen_proj       : Linear towers — fully prunable
  head                      : Linear(256→64)→GELU→Linear(64→1) — protected at output
  aux_head                  : Linear(96→1) — adapts with sensor CNN output channels

Protected layers
----------------
  model.head[-1]  — Linear(64→1): scalar output must stay at 1
  round_to=8      — keeps GroupNorm(G=8) valid throughout sensor CNN

Pipeline
--------
  1. Load trained compressed fusion (phase5_compressed_fusion_best.pt)
     Unfreeze ALL parameters — joint fine-tune needs full gradient flow.
  2. Sensitivity analysis: probe each Conv2d layer at target sparsity,
     measure fusion MAE increase, classify as insensitive/sensitive/very_sensitive.
  3. Per-layer sparsity assignment (with residual-path penalty for both
     ResNet basicblock.conv2 and sensor ResBlock second conv).
  4. Iterative channel pruning via torch-pruning DependencyGraph.
     Intermediate fine-tunes of 10 epochs between steps.
  5. Final long fine-tune (100 epochs, early-stop patience=15).
  6. Save full model object → checkpoints/fusion_pruned{suffix}.pt
     (state dict is not usable — both encoder architectures change)

Usage
-----
    python experiments/compression/cwt/fusion_pruning/train.py
    python experiments/compression/cwt/fusion_pruning/train.py --sparsity 0.5
    python experiments/compression/cwt/fusion_pruning/train.py --skip-sensitivity
    python experiments/compression/cwt/fusion_pruning/train.py --sparsity 0.7 \\
        --output-suffix _70

Run from the thesis root.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.data.fusion_scalogram_dataset import MATWIFusionScalogramDataset
from src.models.multiscale_fusion_model import MultiScaleFusionModel
from src.utils.metrics import mae

DATA_ROOT     = ROOT / "data" / "raw"
SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
CKPT_DIR      = ROOT / "checkpoints"
RESULTS_DIR   = Path(__file__).parent / "results"

COMPRESSED_IMG_CKPT = CKPT_DIR / "resnet_distilled_budget.pt"
PHASE4_SENSOR_CKPT  = CKPT_DIR / "phase4_multiscale_sgdm_best.pt"
FUSION_CKPT         = CKPT_DIR / "phase5_compressed_fusion_best.pt"

IMAGE_FEAT_DIM = 309
AUX_LAMBDA     = 0.2

DEFAULT_SPARSITY        = 0.50   # conservative start — image enc already at 85%
DEFAULT_FINETUNE_EPOCHS = 40
LR                      = 1e-4
WEIGHT_DECAY            = 1e-3
BATCH_SIZE              = 16
NUM_WORKERS             = 0
EARLY_STOP_PATIENCE     = 8

ITERATIVE_THRESHOLD          = 0.50
DEFAULT_ITERATIVE_STEPS      = 3
INTERMEDIATE_FINETUNE_EPOCHS = 5

DEFAULT_SENSITIVE_THRESHOLD      = 0.20
DEFAULT_VERY_SENSITIVE_THRESHOLD = 1.00
DEFAULT_RESIDUAL_PENALTY         = 0.5


# ── Utilities ──────────────────────────────────────────────────────────────────

def model_size_report(model: nn.Module) -> dict:
    n = sum(p.numel() for p in model.parameters())
    return {
        "n_params": n,
        "fp32_kb":  round(n * 4   / 1024, 1),
        "int8_kb":  round(n       / 1024, 1),
        "int4_kb":  round(n * 0.5 / 1024, 1),
    }


def print_size(label: str, stats: dict):
    print(
        f"{label:45s}  params={stats['n_params']:>9,}  "
        f"FP32={stats['fp32_kb']:>7.1f} KB  "
        f"INT8={stats['int8_kb']:>6.1f} KB"
    )


def validate(model, loader, device):
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for images, scalograms, targets in loader:
            preds = model(images.to(device), scalograms.to(device))[0].squeeze(1).cpu()
            preds_all.append(preds)
            targets_all.append(targets)
    p, t = torch.cat(preds_all), torch.cat(targets_all)
    return mae(p, t), (p - t).abs().std().item()


def is_residual_layer(name: str) -> bool:
    """Identify conv layers whose output feeds into a skip-connection addition."""
    # ResNet BasicBlock: second conv and downsample projection
    if ".conv2" in name or ".downsample." in name:
        return True
    # Sensor CNN _ResBlock: second conv is at block.3 (Conv→GN→ReLU→Conv→GN)
    if "block.3" in name:
        return True
    return False


def load_fusion_model(device: str) -> MultiScaleFusionModel:
    """Load the trained compressed fusion model with correct architectures."""
    model = MultiScaleFusionModel(image_feat_dim=IMAGE_FEAT_DIM)
    model.load_compressed_image_encoder(COMPRESSED_IMG_CKPT, device=device)
    model.load_phase4_weights(PHASE4_SENSOR_CKPT, device=device)
    model.load_state_dict(
        torch.load(FUSION_CKPT, map_location=device, weights_only=True)
    )
    # Unfreeze everything — joint fine-tune needs full gradient flow
    for p in model.parameters():
        p.requires_grad = True
    return model.to(device)


# ── Sensitivity analysis ───────────────────────────────────────────────────────

def sensitivity_analysis(model, val_loader, device, probe_sparsity: float):
    """
    Soft-prune each Conv2d layer at probe_sparsity and measure fusion MAE increase.
    Covers both image encoder and sensor CNN conv layers.
    """
    baseline, _ = validate(model, val_loader, device)
    results = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue

        original = module.weight.data.clone()
        norms    = module.weight.data.abs().sum(dim=[1, 2, 3])
        n_prune  = int(probe_sparsity * len(norms))
        if n_prune == 0:
            continue

        module.weight.data[norms.argsort()[:n_prune]] = 0.0
        layer_mae, _ = validate(model, val_loader, device)
        delta    = layer_mae - baseline
        relative = delta / (baseline + 1e-8)

        results.append({
            "layer":             name,
            "out_channels":      module.out_channels,
            "mae_increase":      round(delta, 2),
            "relative_increase": round(relative, 4),
            "is_residual":       is_residual_layer(name),
        })
        module.weight.data = original

        modality = "img" if name.startswith("image_encoder") else "sen"
        marker   = " (residual)" if is_residual_layer(name) else ""
        print(
            f"  [{modality}] {name:55s}  C={module.out_channels:4d}  "
            f"ΔMAE={delta:+6.2f} µm  ({relative:+.1%}){marker}"
        )

    return results, baseline


def classify_and_assign(results, target_sparsity, sensitive_t, very_sensitive_t,
                        residual_penalty):
    mapping, classes = {}, []
    for r in results:
        rel = r["relative_increase"]
        if rel > very_sensitive_t:
            cls, ratio = "very_sensitive", 0.0
        elif rel > sensitive_t:
            cls, ratio = "sensitive",      target_sparsity * 0.5
        else:
            cls, ratio = "insensitive",    target_sparsity

        if r["is_residual"] and ratio > 0:
            ratio *= residual_penalty

        mapping[r["layer"]] = ratio
        classes.append({**r, "sensitivity_class": cls, "assigned_sparsity": round(ratio, 4)})
    return mapping, classes


# ── Fine-tuning ────────────────────────────────────────────────────────────────

def finetune(model, train_loader, val_loader, device, epochs,
             save_best=True, label="finetune", ckpt_name="fusion_pruned.pt",
             history_name="finetune_history.json", initial_val_mae=None):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    criterion    = nn.HuberLoss(delta=20.0)
    train_ds_len = len(train_loader.dataset)
    patience_ctr = 0
    history      = []

    best_val_mae = initial_val_mae if initial_val_mae is not None else float("inf")
    if initial_val_mae is not None and save_best:
        CKPT_DIR.mkdir(exist_ok=True)
        torch.save(model, CKPT_DIR / ckpt_name)
        print(f"  Pre-fine-tune checkpoint saved ({best_val_mae:.2f} µm)")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, scalograms, targets in train_loader:
            images     = images.to(device)
            scalograms = scalograms.to(device)
            targets    = targets.to(device).unsqueeze(1)

            optimizer.zero_grad()
            p_final, p_aux = model(images, scalograms)
            loss = criterion(p_final, targets) + AUX_LAMBDA * criterion(p_aux, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(images)

        train_loss /= train_ds_len
        val_mae_v, val_std = validate(model, val_loader, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_mae": round(val_mae_v, 4), "val_mae_std": round(val_std, 4), "lr": lr,
        })
        print(
            f"  [{label}] Epoch {epoch:3d}  "
            f"train_loss={train_loss:.2f}  "
            f"val_mae={val_mae_v:.2f} ± {val_std:.2f} µm  lr={lr:.2e}"
        )

        if val_mae_v < best_val_mae:
            best_val_mae = val_mae_v
            patience_ctr = 0
            if save_best:
                torch.save(model, CKPT_DIR / ckpt_name)
                print(f"    ✓ New best: {best_val_mae:.2f} µm  (saved {ckpt_name})")
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    if save_best:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / history_name, "w") as f:
            json.dump(history, f, indent=2)

    return best_val_mae


# ── Main ───────────────────────────────────────────────────────────────────────

def run(args):
    try:
        import torch_pruning as tp
    except ImportError:
        sys.exit("torch-pruning not found.  Install with:  pip install torch-pruning")

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps"  if torch.backends.mps.is_available()
        else "cpu"
    )
    sparsity          = args.sparsity
    finetune_epochs   = args.finetune_epochs
    skip_sensitivity  = args.skip_sensitivity
    residual_penalty  = args.residual_penalty
    sensitive_t       = args.sensitive_threshold
    very_sensitive_t  = args.very_sensitive_threshold
    iterative_steps   = args.iterative_steps if args.iterative_steps > 0 else (
        DEFAULT_ITERATIVE_STEPS if sparsity >= ITERATIVE_THRESHOLD else 1
    )
    suffix = args.output_suffix

    ckpt_name        = f"fusion_pruned{suffix}.pt"
    results_name     = f"pruning_results{suffix}.json"
    history_name     = f"finetune_history{suffix}.json"
    sensitivity_name = f"sensitivity_results{suffix}.json"

    print(f"Device                : {device}")
    print(f"Target sparsity       : {sparsity:.0%}")
    print(f"Iterative prune steps : {iterative_steps}")
    print(f"Intermediate epochs   : {INTERMEDIATE_FINETUNE_EPOCHS} per step")
    print(f"Final fine-tune epochs: {finetune_epochs}  (patience={EARLY_STOP_PATIENCE})")
    print(f"Pruning scope         : image encoder + sensor CNN + projection towers")
    print()

    for p in (COMPRESSED_IMG_CKPT, PHASE4_SENSOR_CKPT, FUSION_CKPT):
        if not p.exists():
            sys.exit(f"Required checkpoint not found: {p}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, "train")
    val_ds   = MATWIFusionScalogramDataset(DATA_ROOT, SCALOGRAM_DIR, FEATURES_PATH, "val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print(f"Train : {len(train_ds)} samples    Val : {len(val_ds)} samples\n")

    # ── Load model ─────────────────────────────────────────────────────────────
    model = load_fusion_model(device)
    baseline_stats = model_size_report(model)
    print_size("Baseline (phase5_compressed_fusion)", baseline_stats)

    val_mae_before, _ = validate(model, val_loader, device)
    print(f"Val MAE before pruning : {val_mae_before:.2f} µm\n")

    # Two-input example for torch-pruning dependency graph tracing
    example_inputs = (
        torch.randn(1, 3, 224, 224).to(device),
        torch.randn(1, 5,  64,  64).to(device),
    )

    # ── Sensitivity analysis ──────────────────────────────────────────────────
    ch_sparsity_dict    = {}
    sensitivity_results = []

    if not skip_sensitivity:
        print("=" * 75)
        print(f"SENSITIVITY ANALYSIS  (probe at {sparsity:.0%}, both encoders)")
        print("=" * 75)
        sensitivity_results, _ = sensitivity_analysis(model, val_loader, device, sparsity)

        rels        = [r["relative_increase"] for r in sensitivity_results]
        rels_sorted = sorted(rels)
        n           = len(rels)
        print(
            f"\nRelative-MAE-increase distribution ({n} conv layers):\n"
            f"  min    : {min(rels):+.1%}\n"
            f"  q1     : {rels_sorted[n//4]:+.1%}\n"
            f"  median : {rels_sorted[n//2]:+.1%}\n"
            f"  q3     : {rels_sorted[3*n//4]:+.1%}\n"
            f"  max    : {max(rels):+.1%}\n"
            f"  mean   : {statistics.mean(rels):+.1%}\n"
            f"\nThresholds: sensitive>{sensitive_t:+.0%}  "
            f"very_sensitive>{very_sensitive_t:+.0%}"
        )

        layer_sparsity_map, sensitivity_results = classify_and_assign(
            sensitivity_results, sparsity, sensitive_t, very_sensitive_t, residual_penalty
        )

        name_to_module = {n: m for n, m in model.named_modules()}
        for layer_name, ratio in layer_sparsity_map.items():
            mod = name_to_module.get(layer_name)
            if mod is not None:
                ch_sparsity_dict[mod] = ratio

        print("\nPer-layer assignments:")
        n_img_layers = sum(1 for r in sensitivity_results if r["layer"].startswith("image_encoder"))
        n_sen_layers = len(sensitivity_results) - n_img_layers
        print(f"  Image encoder : {n_img_layers} conv layers")
        print(f"  Sensor CNN    : {n_sen_layers} conv layers")
        for r in sensitivity_results:
            tag = " (residual)" if r["is_residual"] else ""
            mod = "img" if r["layer"].startswith("image_encoder") else "sen"
            print(
                f"  [{mod}] {r['layer']:55s}  "
                f"[{r['sensitivity_class']:14s}]{tag:11s}  → {r['assigned_sparsity']:.0%}"
            )

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / sensitivity_name, "w") as f:
            json.dump({"target_sparsity": sparsity, "baseline_val_mae": round(val_mae_before, 2),
                       "layers": sensitivity_results}, f, indent=2)
    else:
        print("Sensitivity analysis skipped — uniform sparsity for all layers.")

    # ── Pruning ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print(f"PRUNING  ({iterative_steps} step{'s' if iterative_steps > 1 else ''})")
    print("=" * 75)

    # Only protect the scalar output layer.
    # Everything else (image encoder, sensor CNN, projection towers, aux head) is prunable.
    ignored = [model.head[-1]]

    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=tp.importance.MagnitudeImportance(p=1),
        iterative_steps=iterative_steps,
        ch_sparsity=sparsity,
        ch_sparsity_dict=ch_sparsity_dict if ch_sparsity_dict else None,
        ignored_layers=ignored,
        round_to=8,   # GroupNorm(G=8) compatibility throughout sensor CNN
    )

    CKPT_DIR.mkdir(exist_ok=True)

    for step in range(iterative_steps):
        pruner.step()
        post_mae, _ = validate(model, val_loader, device)
        stats = model_size_report(model)
        n_img = sum(p.numel() for p in model.image_encoder.parameters())
        n_sen = sum(p.numel() for p in model.sensor_cnn.parameters())
        print(f"\nStep {step + 1}/{iterative_steps} — cumulative ≈ {(step+1)/iterative_steps*sparsity:.0%}")
        print_size(f"  After step {step+1}", stats)
        print(f"  Image encoder : {n_img:,}  Sensor CNN : {n_sen:,}")
        print(f"  Val MAE post-prune : {post_mae:.2f} µm")

        if step < iterative_steps - 1:
            print(f"  Intermediate fine-tune ({INTERMEDIATE_FINETUNE_EPOCHS} epochs)")
            finetune(model, train_loader, val_loader, device,
                     epochs=INTERMEDIATE_FINETUNE_EPOCHS, save_best=False,
                     label=f"step{step+1}")

    post_prune_mae, _ = validate(model, val_loader, device)
    pruned_stats      = model_size_report(model)
    n_img_pruned      = sum(p.numel() for p in model.image_encoder.parameters())
    n_sen_pruned      = sum(p.numel() for p in model.sensor_cnn.parameters())

    print()
    print_size("After all pruning", pruned_stats)
    print(f"  Image encoder : {n_img_pruned:,}  ({n_img_pruned/1024:.0f} KB INT8)")
    print(f"  Sensor CNN    : {n_sen_pruned:,}  ({n_sen_pruned/1024:.0f} KB INT8)")
    print(f"Val MAE before final fine-tune : {post_prune_mae:.2f} µm")

    # ── Final long fine-tune ──────────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print(f"FINAL FINE-TUNE  ({finetune_epochs} epochs max, patience={EARLY_STOP_PATIENCE})")
    print("=" * 75)
    best_val_mae = finetune(
        model, train_loader, val_loader, device,
        epochs=finetune_epochs, save_best=True,
        label="final", ckpt_name=ckpt_name, history_name=history_name,
        initial_val_mae=post_prune_mae,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    best_model   = torch.load(CKPT_DIR / ckpt_name, map_location="cpu", weights_only=False)
    final_stats  = model_size_report(best_model)
    n_img_final  = sum(p.numel() for p in best_model.image_encoder.parameters())
    n_sen_final  = sum(p.numel() for p in best_model.sensor_cnn.parameters())

    print("\n" + "─" * 75)
    print("JOINT FUSION PRUNING SUMMARY")
    print("─" * 75)
    print_size("Baseline", baseline_stats)
    print_size(f"Pruned  ", final_stats)
    actual_reduction = 1 - final_stats["n_params"] / baseline_stats["n_params"]
    print(f"\nActual param reduction : {actual_reduction:.1%}")
    print(f"Image encoder          : {n_img_final:,} params  ({n_img_final/1024:.0f} KB INT8)")
    print(f"Sensor CNN             : {n_sen_final:,} params  ({n_sen_final/1024:.0f} KB INT8)")
    print(f"INT8 total             : {final_stats['int8_kb']:.0f} KB  "
          f"(target: 2048 KB, gap: {final_stats['int8_kb'] - 2048:+.0f} KB)")
    print(f"\nVal MAE baseline       : {val_mae_before:.2f} µm")
    print(f"Val MAE pruned         : {best_val_mae:.2f} µm")
    print(f"Checkpoint             : {CKPT_DIR / ckpt_name}")
    print(f"  (saved as full model object — load with torch.load(..., weights_only=False))")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / results_name, "w") as f:
        json.dump({
            "sparsity":              sparsity,
            "n_params_before":       baseline_stats["n_params"],
            "n_params_after":        final_stats["n_params"],
            "actual_reduction":      round(actual_reduction, 4),
            "n_params_image_encoder": n_img_final,
            "n_params_sensor_cnn":   n_sen_final,
            "int8_kb_before":        baseline_stats["int8_kb"],
            "int8_kb_after":         final_stats["int8_kb"],
            "fits_in_flash":         final_stats["int8_kb"] <= 2048,
            "val_mae_baseline":      round(val_mae_before, 2),
            "val_mae_pruned":        round(best_val_mae, 2),
            "checkpoint":            str(CKPT_DIR / ckpt_name),
        }, f, indent=2)
    print(f"Results                : {RESULTS_DIR / results_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparsity",              type=float, default=DEFAULT_SPARSITY,
                        help="Target channel sparsity (default: 0.50 — image enc already at 85%%)")
    parser.add_argument("--finetune-epochs",       type=int,   default=DEFAULT_FINETUNE_EPOCHS)
    parser.add_argument("--iterative-steps",       type=int,   default=0,
                        help="0 = auto (3 if sparsity>=0.50, else 1)")
    parser.add_argument("--sensitive-threshold",   type=float, default=DEFAULT_SENSITIVE_THRESHOLD)
    parser.add_argument("--very-sensitive-threshold", type=float,
                        default=DEFAULT_VERY_SENSITIVE_THRESHOLD)
    parser.add_argument("--residual-penalty",      type=float, default=DEFAULT_RESIDUAL_PENALTY)
    parser.add_argument("--skip-sensitivity",      action="store_true")
    parser.add_argument("--output-suffix",         default="",
                        help="e.g. '_50' → fusion_pruned_50.pt")
    args = parser.parse_args()
    run(args)
