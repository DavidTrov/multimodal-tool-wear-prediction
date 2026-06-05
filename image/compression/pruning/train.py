"""
Compression Phase 1 — Structured Channel Pruning with Sensitivity Analysis.

Implements Li et al. "Pruning Filters for Efficient ConvNets" (ICLR 2017),
with architecture-aware adaptations for ResNet (Li et al. §3.3 / §4.2) and
greedy-style iterative pruning at high sparsity targets (Li et al. §3.4).

Pipeline
--------
  1. Load phase1_best.pt
  2. Sensitivity analysis at the *target* sparsity (Fix 1):
     - For every conv layer, soft-prune at the user's actual target ratio
     - Record relative MAE increase
     - Report the full distribution so thresholds can be tuned (Fix 4)
  3. Per-layer sparsity assignment based on sensitivity class
  4. Apply residual-path penalty: layers whose outputs touch a residual
     skip connection (`.conv2` / `.downsample.`) are pruned at a reduced
     ratio, matching Li et al.'s ResNet-specific recommendation (Fix 3)
  5. Hard-prune all layers at once via torch-pruning's dependency graph.
     For target sparsity ≥ ITERATIVE_THRESHOLD, pruning is split across
     multiple steps with intermediate fine-tunes — Li et al.'s greedy
     strategy, which is more accurate when many filters are pruned (Fix 2)
  6. Final long fine-tune
  7. Save full pruned model to checkpoints/pruned.pt

Install dependency
------------------
    pip install torch-pruning

Usage
-----
    python experiments/compression/phase1_pruning/train.py
    python experiments/compression/phase1_pruning/train.py --sparsity 0.9
    python experiments/compression/phase1_pruning/train.py --sparsity 0.9 --iterative-steps 6
    python experiments/compression/phase1_pruning/train.py --sensitive-threshold 0.30
    python experiments/compression/phase1_pruning/train.py --residual-penalty 0.3
    python experiments/compression/phase1_pruning/train.py --skip-sensitivity

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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from image.baseline.dataset import MATWIDataset
from image.baseline.model import build_resnet18_regressor
from src.metrics import mae

DATA_ROOT   = ROOT / "data" / "raw"
CKPT_DIR    = ROOT / "image" / "compression" / "checkpoints"
RESULTS_DIR = Path(__file__).parent / "results"

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_SPARSITY               = 0.80
DEFAULT_FINETUNE_EPOCHS        = 30

LR                  = 1e-5     # matches He et al. ResNet-50 fine-tune (1e-5)
WEIGHT_DECAY        = 1e-3
BATCH_SIZE          = 16
NUM_WORKERS         = 0
EARLY_STOP_PATIENCE = 10

# Fix 2 — switch to iterative pruning when target sparsity is aggressive.
# Li et al. §3.4: "Iterative pruning and retraining may yield better results
# … especially for very deep networks."
ITERATIVE_THRESHOLD            = 0.80    # use iterative if sparsity ≥ 0.80
DEFAULT_ITERATIVE_STEPS        = 4
INTERMEDIATE_FINETUNE_EPOCHS   = 3       # short fine-tune between prune steps

# Fix 4 — defaults for relative-MAE-increase thresholds.  These are *defaults*;
# the actual distribution is printed before pruning so the user can override.
DEFAULT_SENSITIVE_THRESHOLD       = 0.20
DEFAULT_VERY_SENSITIVE_THRESHOLD  = 1.00

# Fix 3 — Li et al. §4.2 recommends only pruning the inner conv of each
# residual block on ResNet.  We allow pruning of residual-touching layers
# but at a reduced sparsity ratio.
DEFAULT_RESIDUAL_PENALTY          = 0.5  # multiply sparsity by this factor


# ── Utilities ─────────────────────────────────────────────────────────────────

def model_size_report(model: nn.Module) -> dict:
    n = sum(p.numel() for p in model.parameters())
    return {
        "n_params": n,
        "fp32_kb":  round(n * 4   / 1024, 1),
        "int8_kb":  round(n * 1   / 1024, 1),
        "int4_kb":  round(n * 0.5 / 1024, 1),
    }


def print_size(label: str, stats: dict):
    print(
        f"{label:35s}  params={stats['n_params']:>9,}  "
        f"FP32={stats['fp32_kb']:>7.1f} KB  "
        f"INT8={stats['int8_kb']:>6.1f} KB  "
        f"INT4={stats['int4_kb']:>6.1f} KB"
    )


def validate(model, loader, device):
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for images, targets in loader:
            preds = model(images.to(device)).squeeze(1).cpu()
            preds_all.append(preds)
            targets_all.append(targets)
    p, t = torch.cat(preds_all), torch.cat(targets_all)
    return mae(p, t), (p - t).abs().std().item()


def is_residual_layer(layer_name: str) -> bool:
    """
    Identify layers whose *output channels* touch a residual skip connection.

    For torchvision ResNet18 BasicBlocks:
      - layerN.M.conv2     → output added to residual
      - layerN.M.downsample.0 → 1×1 projection in shortcut
    Pruning these forces the residual addition to lose channels too, so
    Li et al. (§4.2) recommends being conservative here.
    """
    return ".conv2" in layer_name or ".downsample." in layer_name


# ── Sensitivity analysis (Li et al. §3.2) ─────────────────────────────────────

def sensitivity_analysis(model, val_loader, device, probe_sparsity: float):
    """
    Soft-prune each conv layer independently at *probe_sparsity* (= target
    sparsity, per Fix 1) and measure the resulting MAE increase on the
    validation set.  Weights are restored after each layer test.

    Returns
    -------
    results       : list of dicts {layer, out_channels, mae_increase,
                                   relative_increase, sensitivity_class}
    baseline_mae  : float
    """
    baseline_mae, _ = validate(model, val_loader, device)
    results = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue

        original = module.weight.data.clone()

        # ℓ₁-norm per output filter (Li et al. Eq. s_j = Σ Σ |K_l|)
        norms   = module.weight.data.abs().sum(dim=[1, 2, 3])
        n_prune = int(probe_sparsity * len(norms))
        if n_prune == 0:
            continue

        prune_idx = norms.argsort()[:n_prune]
        module.weight.data[prune_idx] = 0.0

        layer_mae, _ = validate(model, val_loader, device)
        mae_increase = layer_mae - baseline_mae
        relative     = mae_increase / (baseline_mae + 1e-8)

        results.append({
            "layer":             name,
            "out_channels":      module.out_channels,
            "mae_increase":      round(mae_increase, 2),
            "relative_increase": round(relative, 4),
            "is_residual":       is_residual_layer(name),
        })

        module.weight.data = original   # restore

        marker = " (residual)" if is_residual_layer(name) else ""
        print(
            f"  {name:40s}  C={module.out_channels:4d}  "
            f"ΔMAE={mae_increase:+6.2f} µm  ({relative:+.1%}){marker}"
        )

    return results, baseline_mae


def report_sensitivity_distribution(results, sensitive_t, very_sensitive_t):
    """Fix 4 — show the distribution so users can judge their thresholds."""
    rels = [r["relative_increase"] for r in results]
    rels_sorted = sorted(rels)
    n = len(rels)
    print(
        f"\nRelative-MAE-increase distribution across {n} conv layers:\n"
        f"  min      : {min(rels):+.1%}\n"
        f"  q1  (25%): {rels_sorted[n // 4]:+.1%}\n"
        f"  median   : {rels_sorted[n // 2]:+.1%}\n"
        f"  q3  (75%): {rels_sorted[3 * n // 4]:+.1%}\n"
        f"  max      : {max(rels):+.1%}\n"
        f"  mean     : {statistics.mean(rels):+.1%}\n"
        f"\nThresholds in use:\n"
        f"  sensitive       (halve sparsity)  : > {sensitive_t:+.1%}\n"
        f"  very_sensitive  (skip layer)      : > {very_sensitive_t:+.1%}"
    )


def classify_and_assign(results, target_sparsity, sensitive_t, very_sensitive_t, residual_penalty):
    """
    Map each layer's relative MAE increase → a sparsity ratio.

    Fix 3 — for residual-touching layers, multiply the assigned sparsity by
    `residual_penalty` (default 0.5), making pruning more conservative on
    layers whose outputs flow into a skip connection.
    """
    mapping = {}
    classes = []
    for r in results:
        rel = r["relative_increase"]
        if rel > very_sensitive_t:
            cls, ratio = "very_sensitive", 0.0
        elif rel > sensitive_t:
            cls, ratio = "sensitive",      target_sparsity * 0.5
        else:
            cls, ratio = "insensitive",    target_sparsity

        if r["is_residual"] and ratio > 0:
            ratio = ratio * residual_penalty       # Fix 3

        mapping[r["layer"]] = ratio
        classes.append({**r, "sensitivity_class": cls, "assigned_sparsity": round(ratio, 4)})
    return mapping, classes


# ── Target-params scaling ────────────────────────────────────────────────────

def scale_sparsity_map_to_target(
    layer_sparsity_map: dict,
    sensitivity_results: list,
    target_params: int,
    model: nn.Module,
    device,
    max_sparsity: float = 0.92,
) -> dict:
    """
    Binary-search a global scale factor `m` such that pruning with
    `min(assigned_sparsity * m, max_sparsity)` per layer yields a model with
    ≤ target_params parameters.

    Layers previously assigned 0% (very_sensitive) are given a floor of
    `min(0.30 * m, max_sparsity)` so they also get pruned when the budget
    forces it.

    A deepcopy dry-run is used for each candidate `m` so the live model is
    untouched.  Dry-runs are done on CPU regardless of training device.
    """
    import copy
    try:
        import torch_pruning as tp
    except ImportError:
        print("  [target-params] torch-pruning not available — skipping scaling.")
        return layer_sparsity_map

    layer_out_channels = {r["layer"]: r["out_channels"] for r in sensitivity_results}

    def build_scaled_map(m: float) -> dict:
        scaled = {}
        for name, s in layer_sparsity_map.items():
            if s == 0.0:
                # Previously protected — allow pruning with a growing floor
                scaled[name] = min(0.30 * m, max_sparsity)
            else:
                scaled[name] = min(s * m, max_sparsity)
        return scaled

    def dry_run_params(m: float) -> int:
        scaled_map = build_scaled_map(m)
        mc = copy.deepcopy(model).cpu().eval()
        example_cpu = torch.randn(1, 3, 224, 224)
        name_to_mod = {n: mod for n, mod in mc.named_modules()}
        sd = {name_to_mod[n]: v for n, v in scaled_map.items() if n in name_to_mod}
        try:
            pruner = tp.pruner.MetaPruner(
                mc, example_cpu,
                importance=tp.importance.MagnitudeImportance(p=1),
                iterative_steps=1,
                ch_sparsity=0.0,
                ch_sparsity_dict=sd,
                ignored_layers=[mc.fc],
            )
            pruner.step()
            n = sum(p.numel() for p in mc.parameters())
        except Exception as e:
            print(f"    [dry-run m={m:.2f}] error: {e}")
            n = int(1e9)
        del mc
        return n

    print(f"\n  Scaling sparsities to reach ≤{target_params:,} params ...")

    # Quick check — is target already met at m=1?
    n0 = dry_run_params(1.0)
    print(f"    m=1.00 → {n0:,} params")
    if n0 <= target_params:
        print("    Already within target at m=1.00 — no scaling needed.")
        return layer_sparsity_map

    # Binary search m in [1.0, 25.0]
    lo, hi, best_m = 1.0, 25.0, 25.0
    for _ in range(12):
        mid = (lo + hi) / 2
        n = dry_run_params(mid)
        print(f"    m={mid:.2f} → {n:,} params")
        if n <= target_params:
            best_m = mid
            hi = mid
        else:
            lo = mid

    print(f"  Selected m={best_m:.2f} (estimated params after scaling)")
    return build_scaled_map(best_m)


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def finetune(model, train_loader, val_loader, device, epochs,
             save_best=True, label="finetune", ckpt_name="pruned.pt",
             history_name="finetune_history.json",
             initial_val_mae=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion        = nn.MSELoss()
    train_ds_len     = len(train_loader.dataset)
    patience_counter = 0
    history          = []

    # Seed best_val_mae with the pre-fine-tune MAE (if provided) and save
    # the current weights immediately — this ensures we never end up worse
    # than the post-prune starting point even if the LR overshoots.
    if initial_val_mae is not None and save_best:
        best_val_mae = initial_val_mae
        CKPT_DIR.mkdir(exist_ok=True)
        torch.save(model, CKPT_DIR / ckpt_name)
        print(f"  Pre-fine-tune checkpoint saved ({best_val_mae:.2f} µm)")
    else:
        best_val_mae = float("inf")

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
            train_loss += loss.item() * len(images)
        train_loss /= train_ds_len

        val_mae_val, val_std = validate(model, val_loader, device)
        scheduler.step(val_mae_val)
        lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_mae": round(val_mae_val, 4), "val_mae_std": round(val_std, 4),
            "lr": lr,
        })
        print(
            f"  [{label}] Epoch {epoch:3d}  train_loss={train_loss:.2f}  "
            f"val_mae={val_mae_val:.2f} ± {val_std:.2f} µm  lr={lr:.2e}"
        )

        if val_mae_val < best_val_mae:
            best_val_mae = val_mae_val
            patience_counter = 0
            if save_best:
                torch.save(model, CKPT_DIR / ckpt_name)
                print(f"    ✓ New best: {best_val_mae:.2f} µm  (saved {ckpt_name})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    if save_best:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / history_name, "w") as f:
            json.dump(history, f, indent=2)

    return best_val_mae


# ── FLOP estimation (informational) ───────────────────────────────────────────

def estimate_flops(model, example_input):
    """
    Rough FLOP count via torch-pruning's helper.  Reports MACs (multiply-accumulates).
    """
    try:
        import torch_pruning as tp
        macs, params = tp.utils.count_ops_and_params(model, example_input)
        return int(macs), int(params)
    except Exception:
        return None, sum(p.numel() for p in model.parameters())


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    try:
        import torch_pruning as tp
    except ImportError:
        sys.exit("torch-pruning not found.\nInstall with:  pip install torch-pruning")

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    sparsity        = args.sparsity
    finetune_epochs = args.finetune_epochs
    skip_sensitivity = args.skip_sensitivity
    residual_penalty = args.residual_penalty
    sensitive_t      = args.sensitive_threshold
    very_sensitive_t = args.very_sensitive_threshold
    iterative_steps  = args.iterative_steps if args.iterative_steps > 0 else (
        DEFAULT_ITERATIVE_STEPS if sparsity >= ITERATIVE_THRESHOLD else 1
    )
    suffix = args.output_suffix
    # Checkpoint / results filenames — suffix="" preserves legacy names
    ckpt_name       = f"resnet_pruned{suffix}.pt" if suffix else "pruned.pt"
    results_name    = f"pruning_results{suffix}.json"
    history_name    = f"finetune_history{suffix}.json"
    sensitivity_name = f"sensitivity_results{suffix}.json"

    print(f"Device                : {device}")
    print(f"Target sparsity       : {sparsity:.0%}")
    print(f"Iterative prune steps : {iterative_steps}  "
          f"({'enabled' if iterative_steps > 1 else 'one-shot'})")
    print(f"Final fine-tune epochs: {finetune_epochs}")
    print(f"Sensitivity analysis  : {'skipped' if skip_sensitivity else 'enabled'}")
    print(f"Residual penalty      : ×{residual_penalty}")
    print()

    # ── Load baseline ─────────────────────────────────────────────────────────
    phase1_ckpt = ROOT / "image" / "baseline" / "checkpoints" / "phase1_best.pt"
    if not phase1_ckpt.exists():
        sys.exit(f"Phase-1 checkpoint not found: {phase1_ckpt}")

    model = build_resnet18_regressor().to(device)
    model.load_state_dict(torch.load(phase1_ckpt, map_location=device))
    print_size("Baseline (phase1_best.pt)", model_size_report(model))

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = MATWIDataset(DATA_ROOT, split="train")
    val_ds   = MATWIDataset(DATA_ROOT, split="val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    val_mae_before, _ = validate(model, val_loader, device)
    print(f"Val MAE before pruning: {val_mae_before:.2f} µm\n")

    example_input = torch.randn(1, 3, 224, 224).to(device)
    macs_before, _ = estimate_flops(model, example_input)

    # ── Sensitivity analysis at TARGET sparsity (Fix 1) ───────────────────────
    ch_sparsity_dict = {}
    sensitivity_classified = []

    if not skip_sensitivity:
        print("=" * 70)
        print(f"SENSITIVITY ANALYSIS  (probe at target sparsity = {sparsity:.0%})")
        print("=" * 70)

        sensitivity_results, _ = sensitivity_analysis(
            model, val_loader, device, probe_sparsity=sparsity
        )

        report_sensitivity_distribution(
            sensitivity_results, sensitive_t, very_sensitive_t
        )

        layer_sparsity_map, sensitivity_classified = classify_and_assign(
            sensitivity_results, sparsity,
            sensitive_t, very_sensitive_t, residual_penalty,
        )

        # Optional: scale all sparsities up to hit a hard parameter budget
        if args.target_params > 0:
            layer_sparsity_map = scale_sparsity_map_to_target(
                layer_sparsity_map, sensitivity_results,
                args.target_params, model, device,
            )
            # Rebuild sensitivity_classified to reflect scaled values (for logging/JSON)
            name_to_scaled = layer_sparsity_map
            for r in sensitivity_classified:
                r["assigned_sparsity"] = round(name_to_scaled.get(r["layer"], r["assigned_sparsity"]), 4)

        # Map names → modules for torch-pruning
        name_to_module = {n: m for n, m in model.named_modules()}
        for layer_name, ratio in layer_sparsity_map.items():
            mod = name_to_module.get(layer_name)
            if mod is not None:
                ch_sparsity_dict[mod] = ratio

        print("\nPer-layer sparsity assignments:")
        for r in sensitivity_classified:
            tag = " (residual)" if r["is_residual"] else ""
            print(
                f"  {r['layer']:40s}  [{r['sensitivity_class']:14s}]"
                f"{tag:11s}  → {r['assigned_sparsity']:.0%}"
            )

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / sensitivity_name, "w") as f:
            json.dump({
                "target_sparsity":          sparsity,
                "probe_sparsity":           sparsity,
                "baseline_val_mae":         round(val_mae_before, 2),
                "sensitive_threshold":      sensitive_t,
                "very_sensitive_threshold": very_sensitive_t,
                "residual_penalty":         residual_penalty,
                "layers":                   sensitivity_classified,
            }, f, indent=2)
        print(f"\nSensitivity results saved.")
    else:
        print("Sensitivity analysis skipped — using uniform sparsity for all layers.")

    # ── Pruning ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"PRUNING  ({iterative_steps} step{'s' if iterative_steps > 1 else ''})")
    print("=" * 70)

    importance = tp.importance.MagnitudeImportance(p=1)   # ℓ₁-norm
    pruner = tp.pruner.MetaPruner(
        model,
        example_input,
        importance=importance,
        iterative_steps=iterative_steps,
        ch_sparsity=sparsity,
        ch_sparsity_dict=ch_sparsity_dict,
        ignored_layers=[model.fc],
    )

    CKPT_DIR.mkdir(exist_ok=True)

    for step in range(iterative_steps):
        pruner.step()
        cumulative = (step + 1) / iterative_steps * sparsity
        post_mae, _ = validate(model, val_loader, device)
        stats = model_size_report(model)
        print(
            f"\nStep {step + 1}/{iterative_steps} done — cumulative sparsity ≈ {cumulative:.0%}"
        )
        print_size(f"  After step {step + 1}", stats)
        print(f"  Val MAE post-prune: {post_mae:.2f} µm")

        if step < iterative_steps - 1:
            # Intermediate fine-tune (no save)
            print(f"  Intermediate fine-tune ({INTERMEDIATE_FINETUNE_EPOCHS} epochs)")
            finetune(
                model, train_loader, val_loader, device,
                epochs=INTERMEDIATE_FINETUNE_EPOCHS,
                save_best=False, label=f"step{step + 1}",
            )

    pruned_stats     = model_size_report(model)
    macs_after, _    = estimate_flops(model, example_input)
    val_mae_post_prune, _ = validate(model, val_loader, device)
    print()
    print_size("After all pruning", pruned_stats)
    print(f"Val MAE before final fine-tune: {val_mae_post_prune:.2f} µm")
    if macs_before and macs_after:
        flop_reduction = (1 - macs_after / macs_before) * 100
        print(f"FLOPs (MACs)  : {macs_before:,} → {macs_after:,}  ({flop_reduction:.1f}% reduction)")

    # ── Final long fine-tune (saves best) ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"FINAL FINE-TUNE  ({finetune_epochs} epochs max)")
    print("=" * 70)
    best_val_mae = finetune(
        model, train_loader, val_loader, device,
        epochs=finetune_epochs, save_best=True, label="final",
        ckpt_name=ckpt_name, history_name=history_name,
        initial_val_mae=val_mae_post_prune,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("PRUNING SUMMARY")
    print("─" * 70)
    print_size("Baseline", {
        "n_params": 11_173_962,
        "fp32_kb":  round(11_173_962 * 4   / 1024, 1),
        "int8_kb":  round(11_173_962 * 1   / 1024, 1),
        "int4_kb":  round(11_173_962 * 0.5 / 1024, 1),
    })
    print_size("Pruned",   pruned_stats)
    print(f"\nVal MAE  : {val_mae_before:.2f} µm  →  {best_val_mae:.2f} µm")
    if macs_before and macs_after:
        print(f"FLOPs    : {flop_reduction:.1f}% reduction")
    print(f"INT8 size: {pruned_stats['int8_kb']:.1f} KB  (target: ≤2048 KB flash — NXP FRDM-MCXN947)")
    if pruned_stats["int8_kb"] <= 2048:
        print("✓ INT8 size fits within 2 MB flash budget")
    else:
        print(f"✗ Still {pruned_stats['int8_kb'] - 2048:.1f} KB over INT8 budget  "
              f"({pruned_stats['int4_kb']:.1f} KB projected INT4)")
    print("─" * 70)

    results = {
        "sparsity":                sparsity,
        "iterative_steps":         iterative_steps,
        "sensitivity_analysis":    not skip_sensitivity,
        "sensitive_threshold":     sensitive_t,
        "very_sensitive_threshold": very_sensitive_t,
        "residual_penalty":        residual_penalty,
        "finetune_epochs":         finetune_epochs,
        **pruned_stats,
        "macs_before":             macs_before,
        "macs_after":              macs_after,
        "flop_reduction_pct":      round(flop_reduction, 2) if macs_before else None,
        "val_mae_before_pruning":  round(val_mae_before, 2),
        "val_mae_post_prune":      round(val_mae_post_prune, 2),
        "val_mae_after_finetune":  round(best_val_mae, 2),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / results_name, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / results_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparsity",         type=float, default=DEFAULT_SPARSITY,
                        help="Target fraction of channels to remove (default: 0.80)")
    parser.add_argument("--finetune-epochs",  type=int,   default=DEFAULT_FINETUNE_EPOCHS,
                        help="Final fine-tune epochs after pruning (default: 30)")
    parser.add_argument("--iterative-steps",  type=int,   default=0,
                        help=("Number of iterative prune-and-fine-tune steps. "
                              f"0 = auto ({DEFAULT_ITERATIVE_STEPS} when sparsity >= "
                              f"{int(ITERATIVE_THRESHOLD * 100)}%%, else 1)"))
    parser.add_argument("--sensitive-threshold",       type=float,
                        default=DEFAULT_SENSITIVE_THRESHOLD,
                        help="Relative MAE-increase threshold for 'sensitive' "
                             "classification (default: 0.20 = 20%%)")
    parser.add_argument("--very-sensitive-threshold",  type=float,
                        default=DEFAULT_VERY_SENSITIVE_THRESHOLD,
                        help="Relative MAE-increase threshold for 'very_sensitive' "
                             "classification (default: 1.00 = 100%%)")
    parser.add_argument("--residual-penalty", type=float,
                        default=DEFAULT_RESIDUAL_PENALTY,
                        help="Multiplier applied to sparsity for layers whose "
                             "output touches a residual skip connection "
                             "(default: 0.5)")
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="Skip sensitivity analysis and use uniform sparsity")
    parser.add_argument("--target-params", type=int, default=0,
                        help="If set, binary-search a sparsity scale factor so the pruned "
                             "model has at most this many parameters. Overrides per-layer "
                             "sensitivity assignments by scaling them up proportionally. "
                             "E.g. --target-params 2000000 targets ≤2M params (≤2048 KB INT8). "
                             "0 = disabled (default).")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Suffix appended to checkpoint and results filenames "
                             "(e.g. '_90' → resnet_pruned_90.pt, pruning_results_90.json). "
                             "Empty string preserves legacy names (pruned.pt).")
    args = parser.parse_args()
    run(args)
