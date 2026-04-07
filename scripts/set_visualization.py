from pathlib import Path
import pandas as pd
import numpy as np

# ----------------------------
# CONFIG
# ----------------------------
# Example: "data/raw/Set1"
SET_DIR = Path("data/raw/Set1")
SENSOR_DIR = SET_DIR / "sensordata"

OUT_DIR = Path("data/interim")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sensor CSV format: 6 cols, no header, comma-separated
COLS = ["acc", "acoustic", "fx", "fy", "fz", "time"]
CHANNELS = COLS[:-1]

# Optional: for outlier counting
Z_THRESH = 5.0

# ----------------------------
# HELPERS
# ----------------------------
def load_sensor(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] != 6:
        raise ValueError(f"{path.name}: expected 6 cols, got {df.shape[1]}")
    df.columns = COLS
    # force numeric for channels (timestamps ignored here)
    for c in CHANNELS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def channel_stats(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) == 0:
        return {
            "n": 0, "min": np.nan, "p01": np.nan, "p05": np.nan,
            "median": np.nan, "mean": np.nan, "p95": np.nan, "p99": np.nan,
            "max": np.nan, "std": np.nan, "iqr": np.nan, "outliers_z5": np.nan
        }

    q01, q05, q25, q50, q75, q95, q99 = np.quantile(x, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    mu = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    iqr = float(q75 - q25)

    if sd > 0:
        outliers = int((np.abs((x - mu) / sd) > Z_THRESH).sum())
    else:
        outliers = 0

    return {
        "n": int(len(x)),
        "min": float(x.min()),
        "p01": float(q01),
        "p05": float(q05),
        "median": float(q50),
        "mean": mu,
        "p95": float(q95),
        "p99": float(q99),
        "max": float(x.max()),
        "std": sd,
        "iqr": iqr,
        "outliers_z5": outliers,
    }

# ----------------------------
# MAIN
# ----------------------------
csv_files = sorted(SENSOR_DIR.glob("*.csv"))
if not csv_files:
    raise SystemExit(f"No .csv files found in: {SENSOR_DIR}")

rows_long = []
skipped = []

for fp in csv_files:
    try:
        df = load_sensor(fp)
        for ch in CHANNELS:
            s = channel_stats(df[ch])
            rows_long.append({
                "set": SET_DIR.name,
                "file": fp.name,
                "channel": ch,
                **s
            })
    except Exception as e:
        skipped.append({"set": SET_DIR.name, "file": fp.name, "error": str(e)})

stats_long = pd.DataFrame(rows_long)

# Wide format: one row per file, columns like mean__acc, max__fx, etc.
stats_wide = (
    stats_long
    .pivot_table(index=["set", "file"], columns="channel",
                 values=["n","min","p01","p05","median","mean","p95","p99","max","std","iqr","outliers_z5"])
)
stats_wide.columns = [f"{metric}__{ch}" for metric, ch in stats_wide.columns]
stats_wide = stats_wide.reset_index()

# Save outputs
out_long = OUT_DIR / f"{SET_DIR.name}_set_stats_long.csv"
out_wide = OUT_DIR / f"{SET_DIR.name}_set_stats_wide.csv"
stats_long.to_csv(out_long, index=False)
stats_wide.to_csv(out_wide, index=False)

print("Processed files:", len(csv_files))
print("Rows in long table:", len(stats_long))
print("Saved:", out_long)
print("Saved:", out_wide)

if skipped:
    skipped_df = pd.DataFrame(skipped)
    out_skipped = OUT_DIR / f"{SET_DIR.name}_skipped_files.csv"
    skipped_df.to_csv(out_skipped, index=False)
    print("Skipped files:", len(skipped))
    print("Saved:", out_skipped)

# ----------------------------
# FULL RANKING PER SENSOR CHANNEL
# ----------------------------
print("\n=== FULL RANKING PER SENSOR CHANNEL (sorted by max) ===")

for ch in CHANNELS:
    col_max = f"max__{ch}"
    col_mean = f"mean__{ch}"
    col_std = f"std__{ch}"
    col_out = f"outliers_z5__{ch}"

    if col_max in stats_wide.columns:
        print(f"\n--- Ranking by {col_max} ---")
        ranked = stats_wide.sort_values(col_max, ascending=False)
        print(
            ranked[["file", col_max, col_mean, col_std, col_out]]
            .to_string(index=False)
        )
