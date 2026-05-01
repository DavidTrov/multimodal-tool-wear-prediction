"""
Sensor-only dataset for the CWT scalogram CNN.

Each sample returns:
    scalogram : (5, 64, 64)  float32  — stacked CWT power scalograms
    wear      : scalar float32         — VB wear in µm

Pre-compute scalograms with:
    python experiments/phase3_fusion/precompute_scalograms.py

Augmentation (training split only)
----------------------------------
SpecAugment-style augmentation, adapted for CWT scalograms.  Applied with
probability `aug_p` per sample, each operation independently:
  - Time shift  : cyclic roll along the time axis (±max_time_shift)
  - Freq mask   : zero out a random band of frequency rows
  - Time mask   : zero out a random band of time columns
  - Channel drop: occasionally zero out one of the 5 sensor channels (forces
                  the network to use information from multiple sensors)

These mitigate overfitting at the small sample sizes typical of milling
TCM datasets (≈650 training samples here).
"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.dataset import SPLIT_MAP


class ScalogramAugment:
    """SpecAugment for stacked CWT scalograms."""

    def __init__(
        self,
        freq_mask_max: int  = 12,
        time_mask_max: int  = 12,
        time_shift_max: int = 8,
        channel_drop_p: float = 0.1,
        op_p: float = 0.5,
    ):
        self.freq_mask_max  = freq_mask_max
        self.time_mask_max  = time_mask_max
        self.time_shift_max = time_shift_max
        self.channel_drop_p = channel_drop_p
        self.op_p = op_p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C, H, W) — C=5 channels, H=64 freq, W=64 time
        x = x.clone()
        C, H, W = x.shape

        # Cyclic time shift
        if self.time_shift_max > 0 and torch.rand(1).item() < self.op_p:
            shift = int(torch.randint(-self.time_shift_max, self.time_shift_max + 1, (1,)).item())
            x = torch.roll(x, shifts=shift, dims=-1)

        # Frequency mask
        if self.freq_mask_max > 0 and torch.rand(1).item() < self.op_p:
            f  = int(torch.randint(1, self.freq_mask_max + 1, (1,)).item())
            f0 = int(torch.randint(0, max(1, H - f), (1,)).item())
            x[:, f0:f0 + f, :] = 0.0

        # Time mask
        if self.time_mask_max > 0 and torch.rand(1).item() < self.op_p:
            t  = int(torch.randint(1, self.time_mask_max + 1, (1,)).item())
            t0 = int(torch.randint(0, max(1, W - t), (1,)).item())
            x[:, :, t0:t0 + t] = 0.0

        # Channel drop — encourages multi-sensor robustness
        if self.channel_drop_p > 0 and torch.rand(1).item() < self.channel_drop_p:
            c = int(torch.randint(0, C, (1,)).item())
            x[c] = 0.0

        return x


class MATWISensorScalogramDataset(Dataset):
    """
    Args
    ----
    scalogram_dir  : path to data/processed/scalograms/
    features_path  : path to the physics parquet (provides labels_idx, Set, wear)
    split          : "train", "val", or "test"
    augment        : if None, augmentation is enabled iff split == "train"
                     pass False to disable, or a ScalogramAugment instance to
                     override defaults.
    """

    def __init__(
        self,
        scalogram_dir: str | Path,
        features_path: str | Path,
        split:         str,
        augment:       bool | ScalogramAugment | None = None,
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

        # Default: augment train, do not augment val/test
        if augment is None:
            self.augment = ScalogramAugment() if split == "train" else None
        elif augment is False:
            self.augment = None
        elif augment is True:
            self.augment = ScalogramAugment()
        else:
            self.augment = augment

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        row        = self.meta.iloc[idx]
        labels_idx = int(row["labels_idx"])

        scalogram = torch.load(
            self.scalogram_dir / f"{labels_idx}.pt",
            weights_only=True,
        )   # (5, 64, 64)

        if self.augment is not None:
            scalogram = self.augment(scalogram)

        wear = torch.tensor(float(row["wear"]), dtype=torch.float32)
        return scalogram, wear
