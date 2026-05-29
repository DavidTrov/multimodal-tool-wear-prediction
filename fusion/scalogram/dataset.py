"""
Multimodal dataset that pairs tool images with CWT scalograms.

Each sample returns:
    image     : (3, 224, 224) float32  — cropped & normalised tool photograph
    scalogram : (5, 64, 64)  float32  — stacked CWT power scalograms
    wear      : scalar float32         — VB wear in µm

Pre-compute scalograms with:
    python experiments/phase3_fusion/precompute_scalograms.py

Only rows that have both a valid ImageFile AND a pre-computed scalogram file
are included.
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import SPLIT_MAP, IMAGENET_MEAN, IMAGENET_STD, _parse_crop


class MATWIScalogramDataset(Dataset):
    """
    Args
    ----
    data_root      : path to data/raw/
    scalogram_dir  : path to data/processed/scalograms/
    features_path  : path to the physics parquet (provides labels_idx, Set,
                     wear, ImageFile columns)
    split          : "train", "val", or "test"
    image_size     : resize target for tool images (default 224)
    """

    def __init__(
        self,
        data_root:     str | Path,
        scalogram_dir: str | Path,
        features_path: str | Path,
        split:         str,
        image_size:    int = 224,
    ):
        assert split in SPLIT_MAP, f"split must be one of {list(SPLIT_MAP)}"
        self.data_root     = Path(data_root)
        self.scalogram_dir = Path(scalogram_dir)

        # Load physics parquet for metadata (labels_idx, Set, wear, ImageFile)
        feats = pd.read_parquet(features_path)
        feats = feats[feats["ImageFile"].str.len() > 0]
        feats = feats[feats["Set"].isin(SPLIT_MAP[split])]

        # Keep only rows where the pre-computed scalogram exists
        mask = feats["labels_idx"].apply(
            lambda idx: (self.scalogram_dir / f"{int(idx)}.pt").exists()
        )
        feats = feats[mask].reset_index(drop=True)

        # Crop lookup from sets.csv
        sets = pd.read_csv(self.data_root / "sets.csv", index_col=0)
        self._crops: dict[int, tuple] = {}
        for idx_label, row in sets.iterrows():
            set_num = int(str(idx_label).replace("Set ", "").strip())
            self._crops[set_num] = _parse_crop(row["crop"])

        self.feats = feats

        # Match Phase-1 augmentation for training, plain transform for val/test
        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self.feats)

    def __getitem__(self, idx: int):
        row        = self.feats.iloc[idx]
        set_num    = int(row["Set"])
        labels_idx = int(row["labels_idx"])

        # ── Tool image ──────────────────────────────────────────────────────
        rel_path = str(row["ImageFile"]).replace("MATWI/", "", 1)
        image    = Image.open(self.data_root / rel_path).convert("RGB")
        image    = image.crop(self._crops[set_num])
        image    = self.transform(image)

        # ── CWT scalogram ───────────────────────────────────────────────────
        scalogram = torch.load(
            self.scalogram_dir / f"{labels_idx}.pt",
            weights_only=True,
        )   # (5, 64, 64)

        # ── Wear label ──────────────────────────────────────────────────────
        wear = torch.tensor(float(row["wear"]), dtype=torch.float32)

        return image, scalogram, wear
