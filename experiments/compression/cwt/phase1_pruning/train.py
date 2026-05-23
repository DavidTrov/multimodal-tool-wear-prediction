"""
CWT Sensor CNN — Compression Phase 1: Structured Channel Pruning.

Implements Li et al. "Pruning Filters for Efficient ConvNets" (ICLR 2017)
adapted for MultiScaleSensorCNN (GroupNorm, inception entry, CBAM, ResBlock).

Pipeline
--------
  1. Load phase4_multiscale_sgdm_best.pt
  2. Sensitivity analysis at target sparsity — probe each conv layer, report ΔMAE
  3. Per-layer sparsity assignment (insensitive / sensitive / very sensitive)
  4. Apply channel pruning via torch-pruning DependencyGraph
     - round_to=8 keeps GroupNorm(G=8) valid throughout
     - model.head[1] (Linear(96→1)) is ignored → extract_features() stays 96-d
  5. Fine-tune with HuberLoss(δ=20) + Adam
  6. Save full model object → checkpoints/sensor_pruned{suffix}.pt

The 96-d output of extract_features() is preserved so the existing fusion model
(MultiScaleFusionModel with SENSOR_FEAT_DIM=96) works without changes.

Note on CBAM: torch-pruning traces the channel-attention Linear layers via the
dependency graph. round_to=8 prevents GroupNorm group-count violations. If the
CBAM trace fails, add the attention linears to ignored_layers.

Usage
-----
    python experiments/compression/cwt/phase1_pruning/train.py
    python experiments/compression/cwt/phase1_pruning/train.py --sparsity 0.5
    python experiments/compression/cwt/phase1_pruning/train.py --sparsity 0.7 --output-suffix _70
    python experiments/compression/cwt/phase1_pruning/train.py --skip-sensitivity

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

from src.data.sensor_scalogram_dataset import MATWISensorScalogramDataset
from src.models.multiscale_sensor_cnn import MultiScaleSensorCNN
from src.utils.metrics import mae

DATA_ROOT     = ROOT / "data" / "raw"
SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
CKPT_DIR      = ROOT / "checkpoints"
RESULTS_DIR   = Path(__file__).parent / "results"

DEFAULT_SPARSITY        = 0.70
DEFAULT_FINETUNE_EPOCHS = 30
LR                      = 1e-4
WEIGHT_DECAY            = 1e-3
BATCH_SIZE              = 16
NUM_WORKERS             = 0
EARLY_STOP_PATIENCE     = 10

ITERATIVE_THRESHOLD          = 0.70
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
        f"{label:40s}  params={stats['n_params']:>8,}  "
        f"FP32={stats['fp32_kb']:>6.1f} KB  "
        f"INT8={stats['int8_kb']:>5.1f} KB  "
        f"INT4={stats['int4_kb']:>5.1f} KB"
    )


def validate(model, loader, device):
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for scalograms, targets in loader:
            preds = model(scalograms.to(device)).squeeze(1).cpu()
            preds_all.append(preds)
            targets_all.append(targets)
    p, t = torch.cat(preds_all), torch.cat(targets_all)
    return mae(p, t), (p - t).abs().std().item()


def is_residual_layer(name: str) -> bool:
    """Layers whose output touches a skip connection in _ResBlock."""
    return "block.2" in name or "block.3" in name  # second conv + GN in ResBlock


# ── Sensitivity analysis ───────────────────────────────────────────────────────

def sensitivity_analysis(model, val_loader, device, probe_sparsity: float):
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
        marker = " (residual)" if is_residual_layer(name) else ""
        print(
            f"  {name:45s}  C={module.out_channels:3d}  "
            f"ΔMAE={delta:+6.2f} µm  ({relative:+.1%}){marker}"
        )

    return results, baseline


def classify_and_assign(results, target_sparsity, sensitive_t, very_sensitive_t, residual_penalty):
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
             save_best=True, label="finetune", ckpt_name="sensor_pruned.pt",
             history_name="finetune_history.json", initial_val_mae=None):
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
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
        for scalograms, targets in train_loader:
            scalograms = scalograms.to(device)
            targets    = targets.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(scalograms), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(scalograms)
        train_loss /= train_ds_len

        val_mae_v, val_std = validate(model, val_loader, device)
        scheduler.step(val_mae_v)
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

    ckpt_name        = f"sensor_pruned{suffix}.pt"
    results_name     = f"pruning_results{suffix}.json"
    history_name     = f"finetune_history{suffix}.json"
    sensitivity_name = f"sensitivity_results{suffix}.json"

    print(f"Device                : {device}")
    print(f"Target sparsity       : {sparsity:.0%}")
    print(f"Iterative prune steps : {iterative_steps}")
    print(f"Final fine-tune epochs: {finetune_epochs}")
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    phase4_ckpt = CKPT_DIR / "phase4_multiscale_sgdm_best.pt"
    if not phase4_ckpt.exists():
        sys.exit(f"Phase-4 checkpoint not found: {phase4_ckpt}")

    model = MultiScaleSensorCNN().to(device)
    model.load_state_dict(torch.load(phase4_ckpt, map_location=device, weights_only=True))
    print_size("Baseline (phase4_multiscale_sgdm_best)", model_size_report(model))

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = MATWISensorScalogramDataset(SCALOGRAM_DIR, FEATURES_PATH, "train")
    val_ds   = MATWISensorScalogramDataset(SCALOGRAM_DIR, FEATURES_PATH, "val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    val_mae_before, _ = validate(model, val_loader, device)
    print(f"Val MAE before pruning : {val_mae_before:.2f} µm\n")

    example_input = torch.randn(1, 5, 64, 64).to(device)

    # ── Sensitivity analysis ──────────────────────────────────────────────────
    ch_sparsity_dict     = {}
    sensitivity_results  = []

    if not skip_sensitivity:
        print("=" * 70)
        print(f"SENSITIVITY ANALYSIS  (probe at {sparsity:.0%})")
        print("=" * 70)
        sensitivity_results, _ = sensitivity_analysis(model, val_loader, device, sparsity)

        rels = [r["relative_increase"] for r in sensitivity_results]
        rels_sorted = sorted(rels)
        n = len(rels)
        print(
            f"\nRelative-MAE-increase distribution ({n} conv layers):\n"
            f"  min    : {min(rels):+.1%}\n"
            f"  median : {rels_sorted[n//2]:+.1%}\n"
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

        print("\nPer-layer sparsity assignments:")
        for r in sensitivity_results:
            tag = " (residual)" if r["is_residual"] else ""
            print(
                f"  {r['layer']:45s}  [{r['sensitivity_class']:14s}]"
                f"{tag:11s}  → {r['assigned_sparsity']:.0%}"
            )

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / sensitivity_name, "w") as f:
            json.dump({"target_sparsity": sparsity, "baseline_val_mae": round(val_mae_before, 2),
                       "layers": sensitivity_results}, f, indent=2)
    else:
        print("Sensitivity analysis skipped — uniform sparsity for all layers.")

    # ── Pruning ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"PRUNING  ({iterative_steps} step{'s' if iterative_steps > 1 else ''})")
    print("=" * 70)

    # Protect Linear(96→1) head — keeps extract_features() output at 96-d
    # so the fusion model (MultiScaleFusionModel) needs no changes.
    ignored = [model.head[1]]

    pruner = tp.pruner.MetaPruner(
        model,
        example_input,
        importance=tp.importance.MagnitudeImportance(p=1),
        iterative_steps=iterative_steps,
        ch_sparsity=sparsity,
        ch_sparsity_dict=ch_sparsity_dict if ch_sparsity_dict else None,
        ignored_layers=ignored,
        round_to=8,          # keeps GroupNorm(G=8) valid
    )

    CKPT_DIR.mkdir(exist_ok=True)

    for step in range(iterative_steps):
        pruner.step()
        post_mae, _ = validate(model, val_loader, device)
        stats = model_size_report(model)
        print(f"\nStep {step + 1}/{iterative_steps} — cumulative ≈ {(step+1)/iterative_steps*sparsity:.0%}")
        print_size(f"  After step {step+1}", stats)
        print(f"  Val MAE post-prune : {post_mae:.2f} µm")

        if step < iterative_steps - 1:
            print(f"  Intermediate fine-tune ({INTERMEDIATE_FINETUNE_EPOCHS} epochs)")
            finetune(model, train_loader, val_loader, device,
                     epochs=INTERMEDIATE_FINETUNE_EPOCHS, save_best=False,
                     label=f"step{step+1}")

    post_prune_mae, _ = validate(model, val_loader, device)
    print()
    print_size("After all pruning", model_size_report(model))
    print(f"Val MAE before final fine-tune : {post_prune_mae:.2f} µm")

    # ── Final fine-tune ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"FINAL FINE-TUNE  ({finetune_epochs} epochs max, early-stop patience={EARLY_STOP_PATIENCE})")
    print("=" * 70)
    best_val_mae = finetune(
        model, train_loader, val_loader, device,
        epochs=finetune_epochs, save_best=True,
        label="final", ckpt_name=ckpt_name, history_name=history_name,
        initial_val_mae=post_prune_mae,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    final_stats = model_size_report(torch.load(CKPT_DIR / ckpt_name, map_location="cpu", weights_only=False))
    print("\n" + "─" * 70)
    print("PRUNING SUMMARY")
    print("─" * 70)
    print_size("Baseline",       {"n_params": 244463, "fp32_kb": 976.9, "int8_kb": 238.7, "int4_kb": 119.4})
    print_size(f"Pruned {sparsity:.0%}", final_stats)
    print(f"\nVal MAE baseline : {val_mae_before:.2f} µm")
    print(f"Val MAE pruned   : {best_val_mae:.2f} µm")
    print(f"Checkpoint       : {CKPT_DIR / ckpt_name}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / results_name, "w") as f:
        json.dump({
            "sparsity":            sparsity,
            "n_params_before":     244463,
            "n_params_after":      final_stats["n_params"],
            "actual_reduction":    round(1 - final_stats["n_params"] / 244463, 4),
            "int8_kb_before":      238.7,
            "int8_kb_after":       final_stats["int8_kb"],
            "val_mae_baseline":    round(val_mae_before, 2),
            "val_mae_pruned":      round(best_val_mae, 2),
            "checkpoint":          str(CKPT_DIR / ckpt_name),
        }, f, indent=2)
    print(f"Results          : {RESULTS_DIR / results_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparsity",              type=float, default=DEFAULT_SPARSITY)
    parser.add_argument("--finetune-epochs",       type=int,   default=DEFAULT_FINETUNE_EPOCHS)
    parser.add_argument("--iterative-steps",       type=int,   default=0,
                        help="0 = auto (3 if sparsity>=0.70, else 1)")
    parser.add_argument("--sensitive-threshold",   type=float, default=DEFAULT_SENSITIVE_THRESHOLD)
    parser.add_argument("--very-sensitive-threshold", type=float, default=DEFAULT_VERY_SENSITIVE_THRESHOLD)
    parser.add_argument("--residual-penalty",      type=float, default=DEFAULT_RESIDUAL_PENALTY)
    parser.add_argument("--skip-sensitivity",      action="store_true")
    parser.add_argument("--output-suffix",         default="",
                        help="e.g. '_50' → sensor_pruned_50.pt")
    args = parser.parse_args()
    run(args)
