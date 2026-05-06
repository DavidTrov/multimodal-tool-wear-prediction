"""
Aircut diagnostic visualizer.

For representative measurements across Sets 1, 5, 9, 10, 13 (early / mid / late
wear), produces one PNG per file showing:
    Row 1: Raw force channels  fx, fy, fz
    Row 2: Force resultant  Fr = sqrt(fx² + fy² + fz²)
    Row 3: Accelerometer
    Row 4: Sliding-window AC-RMS of Fr   (window=512 samples, ~0.32 s)

Outputs saved to:
    experiments/phase4_sensor_cnn/aircut_diagnostics/<setN>_<sensorID>_wear<W>um.png

Usage:
    python experiments/phase4_sensor_cnn/diagnose_aircuts.py

Run from the thesis root.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RAW_DIR    = ROOT / "data" / "raw"
LABELS_CSV = RAW_DIR / "labels.csv"
OUT_DIR    = Path(__file__).parent / "aircut_diagnostics"

# Sets to inspect and how many wear-quantile representatives to pull
TARGET_SETS   = [1, 5, 9, 10, 13]
N_PER_SET     = 3          # early / mid / late wear quantiles
AC_WINDOW     = 512        # samples for sliding AC-RMS  (~0.32 s at 1.6 kHz)
FS            = 1600       # approximate sample rate (Hz) — used for x-axis only


def ac_rms(signal: np.ndarray, window: int) -> np.ndarray:
    """Sliding-window AC-RMS: remove local mean then compute RMS."""
    out = np.full(len(signal), np.nan)
    half = window // 2
    for i in range(half, len(signal) - half):
        seg = signal[i - half : i + half]
        out[i] = np.sqrt(np.mean((seg - seg.mean()) ** 2))
    return out


def plot_measurement(sensor_path: Path, set_id: int, sensor_id: int, wear: float, out_path: Path):
    df = pd.read_csv(
        sensor_path,
        header=None,
        names=["acc", "acoustic", "fx", "fy", "fz", "datetime"],
    )
    n = len(df)
    t = np.arange(n) / FS   # seconds

    fx = df["fx"].to_numpy(dtype=float)
    fy = df["fy"].to_numpy(dtype=float)
    fz = df["fz"].to_numpy(dtype=float)
    fr = np.sqrt(fx**2 + fy**2 + fz**2)
    ac = df["acc"].to_numpy(dtype=float)
    acrms = ac_rms(fr, AC_WINDOW)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Set {set_id}  |  SensorID {sensor_id}  |  Wear = {wear:.0f} µm\n"
        f"{sensor_path.name}",
        fontsize=11,
    )

    # --- Row 1: force channels ---
    ax = axes[0]
    ax.plot(t, fx, lw=0.4, label="fx")
    ax.plot(t, fy, lw=0.4, label="fy")
    ax.plot(t, fz, lw=0.4, label="fz")
    ax.set_ylabel("Force (N)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(True, lw=0.3, alpha=0.5)

    # --- Row 2: resultant ---
    ax = axes[1]
    ax.plot(t, fr, lw=0.4, color="tab:orange", label="Fr")
    ax.set_ylabel("Fr (N)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.5)

    # --- Row 3: accelerometer ---
    ax = axes[2]
    ax.plot(t, ac, lw=0.3, color="tab:green", label="acc")
    ax.set_ylabel("Acc (g)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.5)

    # --- Row 4: AC-RMS of Fr ---
    ax = axes[3]
    ax.plot(t, acrms, lw=0.5, color="tab:red", label=f"AC-RMS Fr (win={AC_WINDOW})")
    # Mark 20th / 80th percentile thresholds for visual reference
    p20 = np.nanpercentile(acrms, 20)
    p80 = np.nanpercentile(acrms, 80)
    ax.axhline(p20, ls="--", lw=0.8, color="grey", alpha=0.7, label=f"p20={p20:.3f}")
    ax.axhline(p80, ls=":",  lw=0.8, color="grey", alpha=0.7, label=f"p80={p80:.3f}")
    ax.set_ylabel("AC-RMS (N)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(True, lw=0.3, alpha=0.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(ROOT)}")


def pick_representatives(labels: pd.DataFrame, set_id: int, n: int) -> pd.DataFrame:
    sub = labels[labels["Set"] == set_id].sort_values("wear").reset_index(drop=True)
    if len(sub) == 0:
        return sub
    # Evenly spaced quantile indices
    indices = np.linspace(0, len(sub) - 1, min(n, len(sub)), dtype=int)
    return sub.iloc[indices]


def main():
    labels = pd.read_csv(LABELS_CSV)
    OUT_DIR.mkdir(exist_ok=True)

    for set_id in TARGET_SETS:
        reps = pick_representatives(labels, set_id, N_PER_SET)
        if reps.empty:
            print(f"Set {set_id}: no entries in labels.csv — skipping")
            continue
        print(f"\nSet {set_id} — {len(reps)} representative measurements")

        for _, row in reps.iterrows():
            sensor_id = int(row["SensorID"])
            wear      = float(row["wear"])
            # SensorFile column stores "MATWI/Set{N}/sensordata/<fname>"
            rel_path  = str(row["SensorFile"]).replace("MATWI/", "")
            sensor_path = RAW_DIR / rel_path

            if not sensor_path.exists():
                print(f"  MISSING: {sensor_path} — skipping")
                continue

            out_name = f"set{set_id:02d}_sid{sensor_id:04d}_wear{wear:.0f}um.png"
            plot_measurement(sensor_path, set_id, sensor_id, wear, OUT_DIR / out_name)

    print(f"\nDone. Plots in: {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
