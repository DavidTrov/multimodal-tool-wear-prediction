from pathlib import Path
import pandas as pd

CSV_DIR = Path("data/raw/Set1/sensordata")   # change this to your directory
OUTPUT_DIR = Path("data/interim/bad_rows")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

summary = []

for csv_file in CSV_DIR.glob("*.csv"):
    print(f"Checking: {csv_file.name}")

    try:
        df = pd.read_csv(csv_file)

        # Detect missing values (NaN or empty string)
        missing_mask = df.isna().any(axis=1) | (df == "").any(axis=1)
        bad_rows = df[missing_mask]

        n_bad = len(bad_rows)
        summary.append((csv_file.name, len(df), n_bad))

        if n_bad > 0:
            out_path = OUTPUT_DIR / f"{csv_file.stem}_missing_rows.csv"
            bad_rows.to_csv(out_path, index=False)
            print(f"  -> Found {n_bad} bad rows. Saved to {out_path.name}")
        else:
            print("  -> No missing values.")

    except Exception as e:
        print(f"  -> ERROR reading file: {e}")
        summary.append((csv_file.name, 0, "ERROR"))

# Save global summary
summary_df = pd.DataFrame(summary, columns=["file", "total_rows", "rows_with_missing"])
summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)

print("\nDone.")