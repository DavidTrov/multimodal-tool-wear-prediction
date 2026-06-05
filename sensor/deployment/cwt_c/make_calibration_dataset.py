# experiments/cwt_c/make_calibration_dataset.py
"""
Generate ~100 calibration scalograms from the training set for INT8 PTQ.
Saves as a directory of .npy files that eIQ Toolkit can ingest.
"""
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

SCALOGRAM_DIR = ROOT / "data" / "processed" / "scalograms"
LABELS        = ROOT / "data" / "raw" / "labels.csv"
OUT_DIR       = Path(__file__).parent / "calibration_data"

OUT_DIR.mkdir(exist_ok=True)
df = pd.read_csv(LABELS).dropna(subset=["wear"]).reset_index(drop=True)
rng = np.random.default_rng(42)
idxs = rng.choice(len(df), size=min(100, len(df)), replace=False)

for i, idx in enumerate(idxs):
    pt = SCALOGRAM_DIR / f"{idx}.pt"
    if pt.exists():
        arr = torch.load(pt, weights_only=True).numpy().astype(np.float32)
        np.save(OUT_DIR / f"sample_{i:03d}.npy", arr[np.newaxis])  # (1,5,64,64)

print(f"Saved {len(list(OUT_DIR.glob('*.npy')))} calibration samples → {OUT_DIR}")