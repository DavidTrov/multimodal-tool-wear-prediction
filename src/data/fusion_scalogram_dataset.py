"""
Joint image + scalogram dataset for Phase 5 fusion experiments.

Each sample returns:
    image     : (3, 224, 224) float32 — ImageNet-normalised, per-set cropped
    scalogram : (5, 64, 64)   float32 — stacked CWT power scalograms
    wear      : scalar float32         — VB wear in µm

Only samples that have BOTH a valid ImageFile and a pre-computed .pt scalogram
are included (intersection).  On the test split this is 225 of 247 samples.

Pre-compute scalograms with:
    python experiments/phase3_fusion/precompute_scalograms.py --force
"""

import ast
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.dataset import SPLIT_MAP, IMAGENET_MEAN, IMAGENET_STD


def _parse_crop(crop_str: str) -> tuple:
    values = ast.literal_eval(f"({crop_str})")
    return tuple(int(v) for v in values)


class MATWIFusionScalogramDataset(Dataset):
    """
    Args
    ----
    data_root      : path to data/raw/  (contains labels.csv, sets.csv, Set1/, …)
    scalogram_dir  : path to data/processed/scalograms/
    features_path  : path to sensor_features_physics.parquet  (provides labels_idx)
    split          : "train", "val", or "test"  (ignored when `sets` is provided)
    image_size     : resize target for images (default 224)
    sets           : explicit list of set numbers; overrides split/SPLIT_MAP
    train_mode     : if provided, controls augmentation; if None, infers from split
    """

    def __init__(
        self,
        data_root:     str | Path,
        scalogram_dir: str | Path,
        features_path: str | Path,
        split:         str | None  = None,
        image_size:    int         = 224,
        sets:          list | None = None,
        train_mode:    bool | None = None,
    ):
        assert split is not None or sets is not None, "Provide split or sets"
        if split is not None and sets is None:
            assert split in SPLIT_MAP, f"split must be one of {list(SPLIT_MAP)}"
        self.data_root     = Path(data_root)
        self.scalogram_dir = Path(scalogram_dir)

        active_sets = sets if sets is not None else SPLIT_MAP[split]

        # ── Image-side metadata ────────────────────────────────────────────────
        labels = pd.read_csv(self.data_root / "labels.csv")
        labels["labels_idx"] = labels.index           # preserve original row index
        labels = labels[labels["Set"].isin(active_sets)]
        labels = labels[labels["ImageFile"].notna()]
        labels = labels[labels["wear"].notna()]

        # ── Scalogram-side metadata ────────────────────────────────────────────
        feats = pd.read_parquet(features_path)
        feats = feats[feats["Set"].isin(active_sets)]
        scalogram_exists = feats["labels_idx"].apply(
            lambda idx: (self.scalogram_dir / f"{int(idx)}.pt").exists()
        )
        feats = feats[scalogram_exists][["labels_idx"]].copy()

        # ── Intersection on labels_idx ─────────────────────────────────────────
        merged = labels.merge(feats, on="labels_idx", how="inner")
        self.meta = merged[["labels_idx", "Set", "wear", "ImageFile"]].reset_index(drop=True)

        # ── Per-set crop lookup ────────────────────────────────────────────────
        sets_df = pd.read_csv(self.data_root / "sets.csv", index_col=0)
        self._crops: dict[int, tuple] = {}
        for idx_label, row in sets_df.iterrows():
            set_num = int(str(idx_label).replace("Set ", "").strip())
            self._crops[set_num] = _parse_crop(row["crop"])

        # ── Image transforms ───────────────────────────────────────────────────
        use_aug = train_mode if train_mode is not None else (split == "train")
        if use_aug:
            self.img_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        row        = self.meta.iloc[idx]
        labels_idx = int(row["labels_idx"])
        set_num    = int(row["Set"])

        # ── Image ──────────────────────────────────────────────────────────────
        rel_path = str(row["ImageFile"]).replace("MATWI/", "", 1)
        img_path = self.data_root / rel_path
        image    = Image.open(img_path).convert("RGB")
        image    = image.crop(self._crops[set_num])
        image    = self.img_transform(image)

        # ── Scalogram ──────────────────────────────────────────────────────────
        scalogram = torch.load(
            self.scalogram_dir / f"{labels_idx}.pt",
            weights_only=True,
        )   # (5, 64, 64)

        wear = torch.tensor(float(row["wear"]), dtype=torch.float32)
        return image, scalogram, wear
