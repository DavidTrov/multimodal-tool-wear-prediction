"""
Phase 2 – tsfresh feature extraction for all sensor files.

Reads labels.csv, extracts MinimalFCParameters features for every row
that has a valid sensor file and wear label, and saves the result to
data/processed/sensor_features.parquet.

Usage:
    python experiments/phase2_sensor_only/extract_features.py

Run from the thesis root.
This is slow (~1–2 min per sample). Run once and reuse the parquet.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

ROOT      = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "raw"
OUT_PATH  = ROOT / "data" / "processed" / "sensor_features.parquet"

SENSOR_COLS = ["acc", "acoustic", "fx", "fy", "fz", "time"]


def load_sensor_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] != 6:
        raise ValueError(f"{path} has {df.shape[1]} cols, expected 6")
    df.columns = SENSOR_COLS
    return df


def extract_one(df: pd.DataFrame, sample_id: int) -> pd.DataFrame:
    work = df[["acc", "acoustic", "fx", "fy", "fz"]].copy()
    work["id"] = sample_id
    work["t"]  = np.arange(len(work), dtype=np.int32)

    feats = extract_features(
        work[["id", "t", "acc", "acoustic", "fx", "fy", "fz"]],
        column_id="id",
        column_sort="t",
        default_fc_parameters=MinimalFCParameters(),
        n_jobs=0,
        disable_progressbar=True,
        show_warnings=False,
    )
    impute(feats)
    return feats


def run():
    labels = pd.read_csv(DATA_ROOT / "labels.csv")

    # Keep only rows with a sensor file and a valid wear label
    df = labels.dropna(subset=["SensorFile", "wear"]).copy()
    df = df[df["SensorFile"].astype(str).str.len() > 0].reset_index(drop=True)
    print(f"Rows with sensor + wear label: {len(df)}")

    all_features = []
    meta_rows    = []
    skipped      = 0

    for i, row in df.iterrows():
        rel_path  = str(row["SensorFile"]).replace("MATWI/", "", 1)
        sensor_path = DATA_ROOT / rel_path

        if not sensor_path.exists():
            skipped += 1
            continue

        try:
            sensor_df = load_sensor_csv(sensor_path)
            feats     = extract_one(sensor_df, sample_id=i)
            all_features.append(feats)
            meta_rows.append({
                "idx":  i,
                "Set":  int(row["Set"]),
                "wear": float(row["wear"]),
            })
        except Exception as e:
            print(f"  Skipping {sensor_path.name}: {e}")
            skipped += 1

        done = len(all_features)
        if done % 50 == 0 and done > 0:
            print(f"  Extracted {done}/{len(df)} ...")

    print(f"\nExtracted: {len(all_features)}  |  Skipped: {skipped}")

    X    = pd.concat(all_features, axis=0).reset_index(drop=True)
    meta = pd.DataFrame(meta_rows).reset_index(drop=True)

    # Attach metadata as columns
    X["Set"]  = meta["Set"].values
    X["wear"] = meta["wear"].values

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(OUT_PATH)
    print(f"Saved to {OUT_PATH}  ({X.shape[0]} samples, {X.shape[1]} columns)")


if __name__ == "__main__":
    run()
