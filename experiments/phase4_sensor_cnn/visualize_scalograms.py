"""
CWT scalogram inspector — side-by-side with raw sensor signal.

For each selected measurement produces one PNG with two rows:

  Row 1 (time series)  — raw signal for each of the 5 channels.
                          The region used to compute the scalogram (post aircut-
                          gating) is shaded green; discarded aircut in light red.

  Row 2 (scalograms)   — 2D CWT power heatmap (64 freq. bins × 64 time bins)
                          loaded directly from the pre-computed .pt file.
                          y-axis = scale index (low = high freq, high = low freq)
                          x-axis = time within cutting segment (sub-sampled)

Outputs saved to:
    experiments/phase4_sensor_cnn/scalogram_viz/<setN>_<sensorID>_wear<W>um.png

Usage:
    python experiments/phase4_sensor_cnn/visualize_scalograms.py

Run from the thesis root.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.aircut_mask import cutting_mask

RAW_DIR       = ROOT / "data" / "raw"
SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
LABELS_CSV    = RAW_DIR / "labels.csv"
OUT_DIR       = Path(__file__).parent / "scalogram_viz"

ALL_COLS    = ["acc", "acoustic", "fx", "fy", "fz", "datetime"]
CHANNELS    = ["acc", "acoustic", "fx", "fy", "fz"]
CHAN_LABELS = ["Acc (g)", "Acoustic (V)", "Fx (N)", "Fy (N)", "Fz (N)"]
CHAN_COLORS = ["tab:green", "tab:purple", "tab:blue", "tab:orange", "tab:red"]

TARGET_SETS = [1, 5, 9, 10, 13]
N_PER_SET   = 3   # early / mid / late wear
FS          = 1600


def pick_representatives(labels: pd.DataFrame, set_id: int, n: int) -> pd.DataFrame:
    sub = labels[labels["Set"] == set_id].dropna(subset=["wear"]).sort_values("wear")
    sub = sub.reset_index()          # keeps original integer index as column "index"
    if len(sub) == 0:
        return sub
    indices = np.linspace(0, len(sub) - 1, min(n, len(sub)), dtype=int)
    return sub.iloc[indices]


def plot_measurement(
    sensor_path: Path,
    scalogram_path: Path,
    set_id: int,
    sensor_id: int,
    wear: float,
    out_path: Path,
):
    # ── Load raw signal ────────────────────────────────────────────────────────
    df = pd.read_csv(sensor_path, header=None, names=ALL_COLS)
    n  = len(df)
    t  = np.arange(n) / FS

    fx   = df["fx"].to_numpy(float)
    fy   = df["fy"].to_numpy(float)
    fz   = df["fz"].to_numpy(float)
    mask = cutting_mask(fx, fy, fz)          # True = cutting segment

    # ── Load pre-computed scalogram ────────────────────────────────────────────
    scalogram = torch.load(scalogram_path, weights_only=True).numpy()  # (5, 64, 64)

    # ── Layout: 2 rows, 5 cols ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 8))
    fig.suptitle(
        f"Set {set_id}  |  SensorID {sensor_id}  |  Wear = {wear:.0f} µm\n"
        f"{sensor_path.name}",
        fontsize=11,
    )

    gs = gridspec.GridSpec(
        2, 5,
        figure=fig,
        hspace=0.45,
        wspace=0.35,
        top=0.88, bottom=0.07,
        left=0.06, right=0.98,
    )

    for col, (ch, label, color) in enumerate(zip(CHANNELS, CHAN_LABELS, CHAN_COLORS)):

        signal = df[ch].to_numpy(float)

        # ── Row 0: time series ─────────────────────────────────────────────────
        ax_ts = fig.add_subplot(gs[0, col])

        # Aircut region (light red background)
        ax_ts.fill_between(t, signal.min(), signal.max(),
                           where=~mask, color="#ffcccc", alpha=0.6,
                           label="aircut")
        # Cutting region (light green background)
        ax_ts.fill_between(t, signal.min(), signal.max(),
                           where=mask, color="#ccffcc", alpha=0.4,
                           label="cutting")
        ax_ts.plot(t, signal, lw=0.3, color=color)
        ax_ts.set_title(ch, fontsize=9, pad=3)
        ax_ts.set_ylabel(label, fontsize=7)
        ax_ts.set_xlabel("Time (s)", fontsize=7)
        ax_ts.tick_params(labelsize=6)
        ax_ts.grid(True, lw=0.25, alpha=0.5)

        # Small legend only on first column
        if col == 0:
            ax_ts.legend(fontsize=6, loc="upper right")

        # ── Row 1: CWT scalogram ───────────────────────────────────────────────
        ax_cwt = fig.add_subplot(gs[1, col])

        cwt_img = scalogram[col]          # (64, 64)  — (scale, time)
        im = ax_cwt.imshow(
            cwt_img,
            aspect="auto",
            origin="lower",
            cmap="inferno",
            vmin=0.0, vmax=1.0,
            interpolation="nearest",
        )
        ax_cwt.set_title(f"{ch} CWT", fontsize=9, pad=3)
        ax_cwt.set_xlabel("Time bin (cutting segment)", fontsize=7)
        ax_cwt.set_ylabel("Scale (low→high freq)", fontsize=7)
        ax_cwt.tick_params(labelsize=6)

        # Colourbar
        cbar = fig.colorbar(im, ax=ax_cwt, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(ROOT)}")


def main():
    labels = pd.read_csv(LABELS_CSV)

    for set_id in TARGET_SETS:
        reps = pick_representatives(labels, set_id, N_PER_SET)
        if reps.empty:
            print(f"Set {set_id}: no entries — skipping")
            continue
        print(f"\nSet {set_id} — {len(reps)} representative measurements")

        for _, row in reps.iterrows():
            labels_idx = int(row["index"])
            sensor_id  = int(row["SensorID"])
            wear       = float(row["wear"])

            rel_path     = str(row["SensorFile"]).replace("MATWI/", "")
            sensor_path  = RAW_DIR / rel_path
            scalogram_path = SCALOGRAM_DIR / f"{labels_idx}.pt"

            if not sensor_path.exists():
                print(f"  MISSING sensor: {sensor_path.name} — skipping")
                continue
            if not scalogram_path.exists():
                print(f"  MISSING scalogram: {scalogram_path} — skipping")
                continue

            out_name = f"set{set_id:02d}_sid{sensor_id:04d}_wear{wear:.0f}um.png"
            plot_measurement(
                sensor_path, scalogram_path,
                set_id, sensor_id, wear,
                OUT_DIR / out_name,
            )

    print(f"\nDone. Plots in: {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
