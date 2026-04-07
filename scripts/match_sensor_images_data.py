from pathlib import Path
import pandas as pd
import re

DATA_ROOT = Path("data/raw")
MAX_GAP_SECONDS = 1.5

def extract_timestamp(name: str):
    """
    Extract timestamp like:
    2022-09-12T12_27_16.916823
    """
    match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}\.\d+", name)
    if not match:
        return pd.NaT
    
    ts = match.group()
    ts = ts.replace("_", ":")  # convert HH_MM_SS → HH:MM:SS
    return pd.to_datetime(ts, errors="coerce")

all_results = []

for set_dir in sorted(DATA_ROOT.glob("Set*")):

    img_dir = set_dir / "images"
    sen_dir = set_dir / "sensordata"

    if not img_dir.exists() or not sen_dir.exists():
        print(f"Missing folder in {set_dir.name}")
        continue

    images = []
    sensors = []

    # Collect image timestamps
    for img_path in sorted(img_dir.glob("*.jpg")):
        ts = extract_timestamp(img_path.name)
        if pd.notna(ts):
            images.append((img_path, ts))

    # Collect sensor timestamps
    for sen_path in sorted(sen_dir.glob("*.csv")):
        ts = extract_timestamp(sen_path.name)
        if pd.notna(ts):
            sensors.append((sen_path, ts))

    sensors.sort(key=lambda x: x[1])

    for img_path, img_ts in images:

        closest = None
        min_delta = float("inf")

        for sen_path, sen_ts in sensors:

            delta = (img_ts - sen_ts).total_seconds()

            abs_delta = abs(delta)

            if abs_delta < min_delta:
                min_delta = abs_delta
                closest = (sen_path, sen_ts, delta)

            elif abs_delta == min_delta:
                # earlier wins → sensor timestamp must be earlier than image
                if delta > 0:
                    closest = (sen_path, sen_ts, delta)

        if closest and min_delta <= MAX_GAP_SECONDS:
            all_results.append({
                "set": set_dir.name,
                "image": img_path.name,
                "sensor": closest[0].name,
                "delta_seconds": closest[2],
                "matched": True
            })
        else:
            all_results.append({
                "set": set_dir.name,
                "image": img_path.name,
                "sensor": None,
                "delta_seconds": None,
                "matched": False
            })

results_df = pd.DataFrame(all_results)

print("\nTotal images:", len(results_df))
print("Matched:", results_df["matched"].sum())
print("Unmatched:", (~results_df["matched"]).sum())

# Save unmatched list
results_df[~results_df["matched"]].to_csv(
    "unmatched_samples.csv",
    index=False
)