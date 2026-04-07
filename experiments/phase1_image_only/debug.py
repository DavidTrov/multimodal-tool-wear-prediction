"""Quick sanity check before training."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import MATWIDataset

ds = MATWIDataset(ROOT / "data" / "raw", split="train")

wears = torch.stack([ds[i][1] for i in range(len(ds))])
images_nan = sum(1 for i in range(len(ds)) if torch.isnan(ds[i][0]).any())

print(f"Samples:        {len(ds)}")
print(f"Wear  – min:    {wears.min():.1f} µm")
print(f"Wear  – max:    {wears.max():.1f} µm")
print(f"Wear  – mean:   {wears.mean():.1f} µm")
print(f"Wear  – NaN:    {torch.isnan(wears).sum().item()}")
print(f"Images w/ NaN:  {images_nan}")
