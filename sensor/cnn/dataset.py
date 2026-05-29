"""
Sensor-only dataset for the CWT scalogram CNN.

Each sample returns:
    scalogram : (5, 64, 64)  float32  — stacked CWT power scalograms
    wear      : scalar float32         — VB wear in µm

Pre-compute scalograms with:
    python experiments/phase3_fusion/precompute_scalograms.py
"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import SPLIT_MAP


class MATWISensorScalogramDataset(Dataset):
    """
    Args
    ----
    scalogram_dir  : path to data/processed/scalograms/
    features_path  : path to the physics parquet (provides labels_idx, Set, wear)
    split          : "train", "val", or "test"
    """

    def __init__(
        self,
        scalogram_dir: str | Path,
        features_path: str | Path,
        split:         str,
    ):
        assert split in SPLIT_MAP, f"split must be one of {list(SPLIT_MAP)}"
        self.scalogram_dir = Path(scalogram_dir)

        feats = pd.read_parquet(features_path)
        feats = feats[feats["Set"].isin(SPLIT_MAP[split])]

        mask = feats["labels_idx"].apply(
            lambda idx: (self.scalogram_dir / f"{int(idx)}.pt").exists()
        )
        feats = feats[mask].reset_index(drop=True)

        self.meta = feats[["labels_idx", "Set", "wear"]].copy()

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        row        = self.meta.iloc[idx]
        labels_idx = int(row["labels_idx"])

        scalogram = torch.load(
            self.scalogram_dir / f"{labels_idx}.pt",
            weights_only=True,
        )   # (5, 64, 64)

        wear = torch.tensor(float(row["wear"]), dtype=torch.float32)
        return scalogram, wear
