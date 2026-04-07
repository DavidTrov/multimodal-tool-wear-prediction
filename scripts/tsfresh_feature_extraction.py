from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- tsfresh ---
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

# -----------------------------
# CONFIG
# -----------------------------
# Adjust these paths
DATA_ROOT = Path("data/raw")           # contains Set1, Set2, ...
LABELS_CSV = DATA_ROOT / "labels.csv"  # dataset master table

# Column names per README (6 columns, no header)  [oai_citation:1‡README-3.md](sediment://file_00000000198872468faaf900ecec5581)
SENSOR_COLS = ["acc", "acoustic", "fx", "fy", "fz", "time"]

# Pick one sensor CSV to explore (either set this OR let the script pick the first paired row)
EXAMPLE_SENSOR_CSV = DATA_ROOT / "Set1" / "sensordata" / "File_name_2022-09-09T13_30_37.534347.csv"  # e.g. Path("data/raw/Set1/sensordata/File_name_....csv")

# If you want to run on ALL paired rows (tsfresh for all files), set True
RUN_ALL = False

# -----------------------------
# HELPERS
# -----------------------------
def load_sensor_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] != 6:
        raise ValueError(f"{path} has {df.shape[1]} cols, expected 6")
    df.columns = SENSOR_COLS
    # parse timestamp column (can be nanosecond-looking strings)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df

def explore_one_sensor(df: pd.DataFrame, title: str):
    # Basic sanity
    print("\n--- SENSOR OVERVIEW ---")
    print("Rows:", len(df))
    print("Missing time:", df["time"].isna().sum())
    print("Any NaNs numeric:", df[SENSOR_COLS[:-1]].isna().any().to_dict())

    # Numeric stats
    print("\n--- NUMERIC STATS ---")
    print(df[SENSOR_COLS[:-1]].describe().T[["min", "max", "mean", "std"]])

    # Plot each channel (downsample for speed)
    step = max(len(df) // 10000, 1)   # ~10k points max
    d = df.iloc[::step].reset_index(drop=True)

    for col in SENSOR_COLS[:-1]:
        plt.figure(figsize=(10, 3))
        plt.plot(d[col])
        plt.title(f"{title} | {col} (downsample step={step})")
        plt.xlabel("time index (downsampled)")
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()

def tsfresh_one_sensor(df: pd.DataFrame, sample_id: int = 0) -> pd.DataFrame:
    # tsfresh expects: [id, sort] + value columns
    work = df.copy()
    work["id"] = sample_id
    work["t"] = np.arange(len(work), dtype=np.int32)

    # Minimal feature set first (fast + stable). You can expand later.
    settings = MinimalFCParameters()

    feats = extract_features(
        work[["id", "t", "acc", "acoustic", "fx", "fy", "fz"]],
        column_id="id",
        column_sort="t",
        default_fc_parameters=settings,
        n_jobs=0,  # 0 = no multiprocessing (more predictable on laptops)
        disable_progressbar=False,
    )

    # tsfresh can produce NaNs -> impute for ML
    impute(feats)
    return feats

# -----------------------------
# MAIN
# -----------------------------
labels = pd.read_csv(LABELS_CSV)

# Keep only rows that have both files (some are missing due to sync issues)  [oai_citation:2‡README-3.md](sediment://file_00000000198872468faaf900ecec5581)
paired = labels.dropna(subset=["ImageFile", "SensorFile"]).copy()
paired = paired[paired["SensorFile"].astype(str).str.len() > 0]

print("Total label rows:", len(labels))
print("Paired rows (image+sensor):", len(paired))

# Pick one sensor file to explore
if EXAMPLE_SENSOR_CSV is None:
    # SensorFile in labels.csv is usually relative path
    sensor_path = DATA_ROOT / str(paired.iloc[0]["SensorFile"])
else:
    sensor_path = Path(EXAMPLE_SENSOR_CSV)

print("\nExploring sensor file:", sensor_path)

df = load_sensor_csv(sensor_path)
explore_one_sensor(df, title=sensor_path.name)

print("\n--- TSFRESH (one file, minimal features) ---")
feats = tsfresh_one_sensor(df, sample_id=0)
print("Feature shape:", feats.shape)
print(feats.T.sort_values(by=0, ascending=False).head(30))  # show top 30 by value (just for inspection)

# OPTIONAL: run tsfresh for all paired rows (produces one feature row per sensor file)
if RUN_ALL:
    all_feat_rows = []
    ids = []

    for i, r in paired.reset_index(drop=True).iterrows():
        sp = DATA_ROOT / str(r["SensorFile"])
        try:
            s = load_sensor_csv(sp)
            f = tsfresh_one_sensor(s, sample_id=i)
            all_feat_rows.append(f)
            ids.append(i)
        except Exception as e:
            print("Skipping", sp, "reason:", e)

    X = pd.concat(all_feat_rows, axis=0)
    # Align labels
    y_wear = paired.iloc[ids]["wear"].reset_index(drop=True)
    y_type = paired.iloc[ids]["type"].reset_index(drop=True)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    X.to_parquet(out_dir / "sensor_tsfresh_minimal.parquet")
    y_wear.to_csv(out_dir / "y_wear.csv", index=False)
    y_type.to_csv(out_dir / "y_type.csv", index=False)

    print("\nSaved features to:", out_dir / "sensor_tsfresh_minimal.parquet")