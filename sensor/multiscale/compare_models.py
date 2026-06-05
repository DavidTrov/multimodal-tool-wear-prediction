"""
Per-sample comparison: best sensor-only CNN vs image-only ResNet18.

For every test (or val/train) sample that has BOTH a valid image and a
pre-computed CWT scalogram, the script runs inference with both models,
records the absolute error of each, and identifies the winner per sample.

Outputs
-------
results/comparison_{split}.csv          — per-sample DataFrame
results/comparison_{split}_scatter.png  — error scatter with parity line
results/comparison_{split}_wear_dist.png— wear distribution: sensor-wins vs image-wins
results/comparison_{split}_by_set.png   — grouped bar: wins per Set

Usage
-----
    python experiments/phase4_sensor_cnn/compare_models.py
    python experiments/phase4_sensor_cnn/compare_models.py --split val
    python experiments/phase4_sensor_cnn/compare_models.py --arch multiscale --optim sgdm
    python experiments/phase4_sensor_cnn/compare_models.py --no-plots

Run from the thesis root.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from image.baseline.dataset import MATWIDataset
from sensor.cnn.dataset import MATWISensorScalogramDataset
from image.baseline.model import build_resnet18_regressor
from sensor.multiscale.model import MultiScaleSensorCNN
from sensor.cnn.model import SensorCNNRegressor

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT      = ROOT / "data" / "raw"
SCALOGRAM_DIR  = ROOT / "data" / "processed" / "scalograms"
FEATURES_PATH  = ROOT / "data" / "processed" / "sensor_features_physics.parquet"
CKPT_DIR       = ROOT / "sensor" / "multiscale" / "checkpoints"
RESULTS_DIR    = Path(__file__).parent / "results"

ARCH_REGISTRY = {
    "baseline":   SensorCNNRegressor,
    "multiscale": MultiScaleSensorCNN,
}

# ── Indexed dataset wrappers ───────────────────────────────────────────────────

class _IndexedImageDataset(MATWIDataset):
    """MATWIDataset that also returns the original labels.csv row index."""

    def __init__(self, data_root, split, idx_map: dict):
        super().__init__(data_root, split)
        # idx_map: {ImageFile_string → labels_idx}
        self.labels["labels_idx"] = self.labels["ImageFile"].map(idx_map)

    def __getitem__(self, i):
        image, wear = super().__getitem__(i)
        return image, wear, int(self.labels.iloc[i]["labels_idx"])


class _IndexedSensorDataset(MATWISensorScalogramDataset):
    """MATWISensorScalogramDataset that also returns labels_idx from self.meta."""

    def __getitem__(self, i):
        scalogram, wear = super().__getitem__(i)
        return scalogram, wear, int(self.meta.iloc[i]["labels_idx"])


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(dataset, model, device: str) -> pd.DataFrame:
    """Run model over all samples; return DataFrame with labels_idx, wear, pred."""
    records = []
    model.eval()
    for i in range(len(dataset)):
        x, wear, labels_idx = dataset[i]
        pred = model(x.unsqueeze(0).to(device)).squeeze().item()
        records.append({
            "labels_idx": labels_idx,
            "wear":        wear.item(),
            "pred":        pred,
        })
    return pd.DataFrame(records)


# ── Printing helpers ───────────────────────────────────────────────────────────

def _sep(char="─", width=70):
    print(char * width)


def print_top(df: pd.DataFrame, title: str, n: int = 10):
    _sep()
    print(title)
    _sep("·")
    cols = ["labels_idx", "Set", "wear", "img_err", "sensor_err", "margin"]
    sub = df[cols].head(n)
    print(sub.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(df: pd.DataFrame, split: str, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    # ── 1. Scatter: img_err vs sensor_err, coloured by wear ───────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        df["img_err"], df["sensor_err"],
        c=df["wear"], cmap="viridis", alpha=0.7, edgecolors="none", s=25,
    )
    plt.colorbar(sc, ax=ax, label="Ground-truth wear (µm)")
    lim = max(df["img_err"].max(), df["sensor_err"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="Parity (equal error)")
    ax.set_xlabel("Image model |error| (µm)")
    ax.set_ylabel("Sensor model |error| (µm)")
    ax.set_title(f"Per-sample error comparison — {split} split\n"
                 f"(points below diagonal = sensor wins)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    fig.tight_layout()
    p = out_dir / f"comparison_{split}_scatter.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p}")

    # ── 2. Wear distribution: sensor-wins vs image-wins ────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(df["wear"].min(), df["wear"].max(), 25)
    ax.hist(df.loc[df["winner"] == "sensor", "wear"], bins=bins,
            alpha=0.6, label="Sensor wins", color="steelblue")
    ax.hist(df.loc[df["winner"] == "image",  "wear"], bins=bins,
            alpha=0.6, label="Image wins",  color="tomato")
    ax.set_xlabel("Ground-truth wear (µm)")
    ax.set_ylabel("Count")
    ax.set_title(f"Wear distribution by winner — {split} split")
    ax.legend()
    fig.tight_layout()
    p = out_dir / f"comparison_{split}_wear_dist.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p}")

    # ── 3. Grouped bar: wins per Set ───────────────────────────────────────────
    sets = sorted(df["Set"].unique())
    sensor_counts = [len(df[(df["Set"] == s) & (df["winner"] == "sensor")]) for s in sets]
    image_counts  = [len(df[(df["Set"] == s) & (df["winner"] == "image")])  for s in sets]
    x = range(len(sets))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - w/2 for i in x], sensor_counts, w, label="Sensor wins", color="steelblue")
    ax.bar([i + w/2 for i in x], image_counts,  w, label="Image wins",  color="tomato")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Set {s}" for s in sets])
    ax.set_ylabel("Number of samples")
    ax.set_title(f"Wins by experimental set — {split} split")
    ax.legend()
    fig.tight_layout()
    p = out_dir / f"comparison_{split}_by_set.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(split: str, arch: str, optim_name: str, no_plots: bool):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device  : {device}  (image model forced to CPU — dynamic INT8 quant)")
    print(f"Split   : {split}")
    print(f"Image   : resnet_qat_int8_2m.pt")
    print(f"Sensor  : {arch} + {optim_name}\n")

    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Build labels_idx map from raw labels.csv ───────────────────────────────
    raw_labels = pd.read_csv(DATA_ROOT / "labels.csv")
    raw_labels["labels_idx"] = raw_labels.index
    idx_map = dict(zip(raw_labels["ImageFile"], raw_labels["labels_idx"]))

    # ── Datasets ───────────────────────────────────────────────────────────────
    img_ds    = _IndexedImageDataset(DATA_ROOT, split, idx_map)
    sensor_ds = _IndexedSensorDataset(SCALOGRAM_DIR, FEATURES_PATH, split)
    print(f"Image dataset  : {len(img_ds)} samples")
    print(f"Sensor dataset : {len(sensor_ds)} samples")

    # ── Models ─────────────────────────────────────────────────────────────────
    img_ckpt    = ROOT / "image" / "compression" / "checkpoints" / "resnet_qat_int8_2m.pt"
    sensor_ckpt = CKPT_DIR / f"phase4_{arch}_{optim_name}_best.pt"

    for p in (img_ckpt, sensor_ckpt):
        if not p.exists():
            sys.exit(f"Checkpoint not found: {p}")

    # QAT INT8 models are saved as full serialized objects and only run on CPU
    torch.backends.quantized.engine = "qnnpack"
    img_model = torch.load(img_ckpt, map_location="cpu", weights_only=False)
    img_model.eval()
    img_device = "cpu"

    sensor_model = ARCH_REGISTRY[arch]().to(device)
    sensor_model.load_state_dict(torch.load(sensor_ckpt, map_location=device, weights_only=True))

    # ── Inference ──────────────────────────────────────────────────────────────
    print("\nRunning image model inference ...")
    img_df = run_inference(img_ds, img_model, img_device)

    print("Running sensor model inference ...")
    sensor_df = run_inference(sensor_ds, sensor_model, device)

    # ── Merge on labels_idx (intersection only) ────────────────────────────────
    merged = img_df.merge(
        sensor_df[["labels_idx", "pred"]].rename(columns={"pred": "sensor_pred"}),
        on="labels_idx", how="inner",
    ).rename(columns={"pred": "img_pred"})

    merged["img_err"]    = (merged["img_pred"]    - merged["wear"]).abs()
    merged["sensor_err"] = (merged["sensor_pred"] - merged["wear"]).abs()
    merged["margin"]     = merged["img_err"] - merged["sensor_err"]   # +ve = sensor wins
    merged["winner"]     = merged["margin"].apply(
        lambda m: "sensor" if m > 0 else ("image" if m < 0 else "tie")
    )

    # Join Set from raw_labels
    merged = merged.merge(
        raw_labels[["labels_idx", "Set"]], on="labels_idx", how="left"
    )

    n = len(merged)
    sensor_wins = (merged["winner"] == "sensor").sum()
    image_wins  = (merged["winner"] == "image").sum()

    # ── Console summary ────────────────────────────────────────────────────────
    n_img    = len(img_ds)
    n_sensor = len(sensor_ds)
    _sep("═")
    print(f"  Summary — {split} split")
    print(f"  Image  dataset : {n_img} samples  |  Sensor dataset : {n_sensor} samples")
    print(f"  Intersection   : {n} samples  "
          f"(image-only excl.: {n_img - n}, sensor-only excl.: {n_sensor - n})")
    print(f"  NOTE: MAEs below are on the {n}-sample intersection, not the full split.")
    _sep("═")
    print(f"  Image  model MAE : {merged['img_err'].mean():.2f} µm")
    print(f"  Sensor model MAE : {merged['sensor_err'].mean():.2f} µm")
    print()
    print(f"  Sensor wins : {sensor_wins:3d} samples ({100*sensor_wins/n:.1f}%)"
          f"   avg margin: {merged.loc[merged['winner']=='sensor','margin'].mean():.1f} µm")
    print(f"  Image  wins : {image_wins:3d} samples ({100*image_wins/n:.1f}%)"
          f"   avg margin: {merged.loc[merged['winner']=='image','margin'].abs().mean():.1f} µm")

    _sep()
    print("  By Set")
    _sep("·")
    hdr = f"  {'Set':>4}  {'N':>5}  {'Sensor wins':>11}  {'Image wins':>10}  "
    hdr += f"{'Sensor MAE':>10}  {'Image MAE':>9}"
    print(hdr)
    _sep("·")
    for s in sorted(merged["Set"].unique()):
        sub = merged[merged["Set"] == s]
        sw  = (sub["winner"] == "sensor").sum()
        iw  = (sub["winner"] == "image").sum()
        print(f"  {int(s):>4}  {len(sub):>5}  {sw:>11}  {iw:>10}  "
              f"{sub['sensor_err'].mean():>10.2f}  {sub['img_err'].mean():>9.2f}")

    _sep()
    print("  Wear range by winner")
    _sep("·")
    for w_label, group in merged.groupby("winner"):
        print(f"  {w_label:>6}-win: "
              f"mean={group['wear'].mean():.1f} µm  "
              f"range=[{group['wear'].min():.1f}–{group['wear'].max():.1f}] µm  "
              f"n={len(group)}")

    print_top(
        merged.sort_values("margin", ascending=False),
        f"  Top 10: sensor beats image (largest margin)",
    )
    print_top(
        merged.sort_values("margin", ascending=True),
        f"  Top 10: image beats sensor (largest margin)",
    )
    _sep("═")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / f"comparison_{split}.csv"
    merged.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n  CSV saved: {csv_path}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    if not no_plots:
        make_plots(merged, split, RESULTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-sample comparison: sensor CNN vs image ResNet18"
    )
    parser.add_argument("--split",    default="test", choices=["train", "val", "test"])
    parser.add_argument("--arch",     default="multiscale", choices=list(ARCH_REGISTRY))
    parser.add_argument("--optim",    default="sgdm",       choices=["adam", "sgdm"])
    parser.add_argument("--no-plots", action="store_true",  help="Skip matplotlib output")
    args = parser.parse_args()
    run(split=args.split, arch=args.arch, optim_name=args.optim, no_plots=args.no_plots)
