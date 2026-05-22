"""
Aggregate all compression results into a summary table and two thesis plots.

Reads
-----
  experiments/compression/resnet/phase1_pruning/results/pruning_results*.json
  experiments/compression/resnet/phase2_distillation/results/distillation_results*.json
  experiments/compression/resnet/phase3_quantization/results/quantization_results*.json
  experiments/compression/resnet/phase3_quantization/results/qat_results*.json

Writes
------
  experiments/compression/resnet/results/compression_summary.json
  experiments/compression/resnet/results/plot_pareto.png
  experiments/compression/resnet/results/plot_sweep.png

Usage
-----
    python experiments/compression/resnet/aggregate_results.py

Run from the thesis root.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BASE      = Path(__file__).parent
OUT_DIR   = BASE / "results"

PRUNE_DIR = BASE / "phase1_pruning"    / "results"
DIST_DIR  = BASE / "phase2_distillation" / "results"
QUANT_DIR = BASE / "phase3_quantization" / "results"

# Unpruned baseline (Phase 1)
BASELINE_TEST_MAE = 23.17
BASELINE_PARAMS   = 11_173_962
BASELINE_INT8_KB  = round(BASELINE_PARAMS / 1024, 1)

ON_DEVICE_KB = 2048   # INT8 flash budget


def _suffix_from_file(path: Path) -> str:
    """
    Extract the run-identifier suffix from result filenames.
      pruning_results.json         → "85"   (legacy file = 85% run)
      pruning_results_50.json      → "50"
      pruning_results_90.json      → "90"
      pruning_results_budget.json  → "budget"
    """
    stem = path.stem                    # e.g. "pruning_results_budget"
    parts = stem.rsplit("_", 1)
    last = parts[-1] if len(parts) == 2 else ""
    # "results" / "history" are part of the base filename, not a run suffix
    if last in ("results", "history", ""):
        return "85"     # legacy 85% file (no explicit suffix)
    return last         # "50", "85", "90", "95", "budget", …


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def collect_rows() -> list[dict]:
    rows = {}  # keyed by sparsity string

    # ── Pruning results ───────────────────────────────────────────────────────
    for p in sorted(PRUNE_DIR.glob("pruning_results*.json")):
        suf = _suffix_from_file(p)
        d   = _load_json(p)
        sparsity = d.get("sparsity", int(suf) / 100 if suf.isdigit() else None)
        rows.setdefault(suf, {"sparsity": sparsity})
        rows[suf].update({
            "n_params":        d.get("n_params"),
            "int8_kb":         d.get("int8_kb"),
            "int4_kb":         d.get("int4_kb"),
            "flop_reduction":  d.get("flop_reduction_pct"),
            "mae_pruned_val":  d.get("val_mae_after_finetune"),
        })

    # ── Distillation results ──────────────────────────────────────────────────
    for p in sorted(DIST_DIR.glob("distillation_results*.json")):
        suf = _suffix_from_file(p)
        d   = _load_json(p)
        rows.setdefault(suf, {})
        rows[suf]["mae_distilled_val"] = d.get("val_mae_after")

    # ── PTQ quantization results ──────────────────────────────────────────────
    for p in sorted(QUANT_DIR.glob("quantization_results*.json")):
        suf = _suffix_from_file(p)
        d   = _load_json(p)
        rows.setdefault(suf, {})
        rows[suf].update({
            "mae_ptq_val":    d.get("val_mae_int8"),
            "mae_ptq_test":   d.get("test_mae_int8"),
            "mae_fp32_test":  d.get("test_mae_fp32"),
            "on_device_int8": d.get("on_device_int8"),
        })

    # ── QAT results ───────────────────────────────────────────────────────────
    for p in sorted(QUANT_DIR.glob("qat_results*.json")):
        suf = _suffix_from_file(p)
        d   = _load_json(p)
        rows.setdefault(suf, {})
        rows[suf].update({
            "mae_qat_val":  d.get("val_mae_qat"),
            "mae_qat_test": d.get("test_mae_qat"),
        })

    # Sort by sparsity; budget row sorts last (after all numeric sparsity levels)
    def _sort_key(item):
        suf, row = item
        if suf == "budget":
            return 999.0
        return float(row.get("sparsity", 0))

    sorted_rows = []
    for suf, row in sorted(rows.items(), key=_sort_key):
        row["on_device_int8"] = row.get("on_device_int8",
                                        (row.get("int8_kb", 1e9) <= ON_DEVICE_KB))
        # Tag budget rows so the table can label them differently
        if suf == "budget":
            row.setdefault("sparsity", 0.95)
            row["_label"] = "budget"
        sorted_rows.append(row)
    return sorted_rows


def print_table(rows: list[dict]):
    print("\n" + "═" * 110)
    print(f"{'Sparsity':>10} {'Params':>10} {'INT8 KB':>8} {'INT4 KB':>8} "
          f"{'Pruned val':>12} {'Distil val':>12} {'PTQ test':>10} {'QAT test':>10} {'On-device':>10}")
    print("─" * 110)

    # Baseline row
    print(f"{'0% (base)':>10} {BASELINE_PARAMS:>10,} {BASELINE_INT8_KB:>8.1f} "
          f"{BASELINE_INT8_KB/2:>8.1f} {'—':>12} {'—':>12} "
          f"{BASELINE_TEST_MAE:>10.2f} {'—':>10} {'✗':>10}")

    for r in rows:
        sp  = r.get("_label", f"{r.get('sparsity', 0):.0%}")
        def fmt(v): return f"{v:.2f}" if v is not None else "—"
        od  = "✓" if r.get("on_device_int8") else "✗"
        print(
            f"{sp:>10} "
            f"{r.get('n_params', 0):>10,} "
            f"{r.get('int8_kb', 0):>8.1f} "
            f"{r.get('int4_kb', 0):>8.1f} "
            f"{fmt(r.get('mae_pruned_val')):>12} "
            f"{fmt(r.get('mae_distilled_val')):>12} "
            f"{fmt(r.get('mae_ptq_test')):>10} "
            f"{fmt(r.get('mae_qat_test')):>10} "
            f"{od:>10}"
        )
    print("═" * 110)
    print("All MAE values in µm.  On-device = INT8 model ≤ 2048 KB flash.")


def plot_pareto(rows: list[dict], out_path: Path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Baseline point
    ax.scatter(BASELINE_INT8_KB, BASELINE_TEST_MAE,
               marker="*", s=200, color="black", zorder=5, label="Unpruned FP32")

    colors = {"pruned": "#4C72B0", "distilled": "#DD8452",
              "ptq": "#55A868", "qat": "#C44E52"}
    markers = {"pruned": "o", "distilled": "s", "ptq": "^", "qat": "D"}

    for r in rows:
        kb  = r.get("int8_kb")
        sp  = r.get("sparsity", 0)
        lbl = f"{sp:.0%}"
        if kb is None:
            continue

        if r.get("mae_ptq_test") is not None:
            ax.scatter(kb, r["mae_ptq_test"], marker=markers["ptq"],
                       color=colors["ptq"], s=80, zorder=4)
            ax.annotate(lbl, (kb, r["mae_ptq_test"]),
                        textcoords="offset points", xytext=(6, 3), fontsize=8)

        if r.get("mae_qat_test") is not None:
            ax.scatter(kb, r["mae_qat_test"], marker=markers["qat"],
                       color=colors["qat"], s=100, zorder=4)

        if r.get("mae_fp32_test") is not None:
            ax.scatter(kb, r["mae_fp32_test"], marker=markers["distilled"],
                       color=colors["distilled"], s=80, zorder=4)

    # Device threshold line
    ax.axvline(ON_DEVICE_KB, color="red", linestyle="--", linewidth=1.2,
               label=f"{ON_DEVICE_KB} KB flash limit")

    # Baseline accuracy line
    ax.axhline(BASELINE_TEST_MAE, color="grey", linestyle=":", linewidth=1,
               label=f"Unpruned baseline ({BASELINE_TEST_MAE} µm)")

    # Legend proxies
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="*",  color="black",          linestyle="None", markersize=10, label="Unpruned FP32"),
        Line2D([0], [0], marker="s",  color=colors["distilled"], linestyle="None", markersize=7, label="Distilled (FP32)"),
        Line2D([0], [0], marker="^",  color=colors["ptq"],    linestyle="None", markersize=7, label="INT8 PTQ"),
        Line2D([0], [0], marker="D",  color=colors["qat"],    linestyle="None", markersize=7, label="INT8 QAT"),
        Line2D([0], [0], color="red", linestyle="--", label=f"{ON_DEVICE_KB} KB flash limit"),
        Line2D([0], [0], color="grey", linestyle=":", label=f"Unpruned baseline"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right")

    ax.set_xlabel("INT8 model size (KB)", fontsize=11)
    ax.set_ylabel("Test MAE (µm)", fontsize=11)
    ax.set_title("Compression Pareto Frontier — ResNet18 Wear Prediction", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_sweep(rows: list[dict], out_path: Path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        return

    sparsities   = [r["sparsity"] * 100 for r in rows if r.get("sparsity")]
    mae_pruned   = [r.get("mae_pruned_val")    for r in rows if r.get("sparsity")]
    mae_distil   = [r.get("mae_distilled_val") for r in rows if r.get("sparsity")]
    mae_ptq      = [r.get("mae_ptq_test")      for r in rows if r.get("sparsity")]

    fig, ax = plt.subplots(figsize=(8, 5))

    def plot_line(xs, ys, label, color, marker):
        pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if pairs:
            px, py = zip(*pairs)
            ax.plot(px, py, marker=marker, color=color, label=label, linewidth=1.8)

    plot_line(sparsities, mae_pruned, "Pruned (val)",        "#4C72B0", "o")
    plot_line(sparsities, mae_distil, "Distilled (val)",     "#DD8452", "s")
    plot_line(sparsities, mae_ptq,    "INT8 PTQ (test)",     "#55A868", "^")

    # QAT star markers
    for r in rows:
        if r.get("mae_qat_test") and r.get("sparsity"):
            ax.scatter(r["sparsity"] * 100, r["mae_qat_test"],
                       marker="D", s=120, color="#C44E52", zorder=5,
                       label="INT8 QAT (test)")

    ax.axhline(BASELINE_TEST_MAE, color="grey", linestyle=":", linewidth=1.2,
               label=f"Unpruned baseline ({BASELINE_TEST_MAE} µm)")

    ax.set_xlabel("Pruning sparsity (%)", fontsize=11)
    ax.set_ylabel("MAE (µm)", fontsize=11)
    ax.set_title("Accuracy vs Sparsity — ResNet18 Wear Prediction", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = collect_rows()
    if not rows:
        print("No results found. Run the compression pipeline first.")
        return

    print_table(rows)

    summary_path = OUT_DIR / "compression_summary.json"
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    plot_pareto(rows, OUT_DIR / "plot_pareto.png")
    plot_sweep(rows,  OUT_DIR / "plot_sweep.png")


if __name__ == "__main__":
    run()
