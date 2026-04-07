"""
Phase 2 – sensor-only baseline (XGBoost).

Loads pre-extracted tsfresh features, applies the paper's train/val/test
split by Set, trains an XGBoost regressor, and evaluates with MAE.

Usage:
    python experiments/phase2_sensor_only/train.py

Run from the thesis root.
Run extract_features.py first if sensor_features.parquet doesn't exist.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor

ROOT        = Path(__file__).resolve().parents[2]
FEATURES    = ROOT / "data" / "processed" / "sensor_features.parquet"
RESULTS_DIR = Path(__file__).parent / "results"
CKPT_DIR    = ROOT / "checkpoints"

TRAIN_SETS = [1, 2, 5, 7, 8, 10, 11]
VAL_SETS   = [3, 6, 12]
TEST_SETS  = [4, 9, 13]


def split_data(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in ("Set", "wear")]

    train = df[df["Set"].isin(TRAIN_SETS)]
    val   = df[df["Set"].isin(VAL_SETS)]
    test  = df[df["Set"].isin(TEST_SETS)]

    X_train, y_train = train[feature_cols].values, train["wear"].values
    X_val,   y_val   = val[feature_cols].values,   val["wear"].values
    X_test,  y_test  = test[feature_cols].values,  test["wear"].values

    return X_train, y_train, X_val, y_val, X_test, y_test


def report(name: str, y_true, y_pred) -> dict:
    errors = np.abs(y_true - y_pred)
    result = {
        "split":    name,
        "n":        len(y_true),
        "mae":      round(float(errors.mean()), 2),
        "mae_std":  round(float(errors.std()), 2),
        "mae_min":  round(float(errors.min()), 2),
        "mae_max":  round(float(errors.max()), 2),
    }
    print(f"{name:5s}  n={result['n']:4d}  MAE={result['mae']:.2f} ± {result['mae_std']:.2f} µm"
          f"  (min={result['mae_min']:.2f}, max={result['mae_max']:.2f})")
    return result


def run():
    if not FEATURES.exists():
        print(f"Features not found at {FEATURES}")
        print("Run extract_features.py first.")
        sys.exit(1)

    df = pd.read_parquet(FEATURES)
    print(f"Loaded features: {df.shape[0]} samples, {df.shape[1] - 2} feature columns\n")

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)
    print(f"Train: {len(y_train)}  |  Val: {len(y_val)}  |  Test: {len(y_test)}\n")

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="mae",
    )
    print("Training XGBoost ...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    print("Done.\n")

    results = {}
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        preds = model.predict(X)
        results[name] = report(name, y, preds)

    print(f"\nPhase 1 image-only test MAE:  23.17 µm")
    print(f"Paper baseline:               19.00 µm")

    # Save model + results
    CKPT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    joblib.dump(model, CKPT_DIR / "phase2_xgb.joblib")
    with open(RESULTS_DIR / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nModel saved to: {CKPT_DIR / 'phase2_xgb.joblib'}")
    print(f"Results saved to: {RESULTS_DIR / 'eval_results.json'}")


if __name__ == "__main__":
    run()
