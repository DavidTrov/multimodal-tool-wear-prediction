import ast
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

TRAIN_SETS = [1, 2, 5, 7, 8, 10, 11]
VAL_SETS   = [3, 6, 12]
TEST_SETS  = [4, 9, 13]

SPLIT_MAP = {
    "train": TRAIN_SETS,
    "val":   VAL_SETS,
    "test":  TEST_SETS,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _parse_crop(crop_str: str) -> tuple[int, int, int, int]:
    """Parse 'left, top, right, bottom' string from sets.csv into a 4-tuple."""
    values = ast.literal_eval(f"({crop_str})")
    return tuple(int(v) for v in values)  # (left, top, right, bottom)


class MATWIDataset(Dataset):
    """
    Image-only dataset for MATWI tool wear regression.

    Args:
        data_root:  path to data/raw/ (contains labels.csv, sets.csv, Set1/, ...)
        split:      "train", "val", or "test"  (ignored when `sets` is provided)
        image_size: resize target (default 224)
        sets:       explicit list of set numbers to include; overrides split/SPLIT_MAP
        train_mode: if provided, controls augmentation; if None, infers from split
    """

    def __init__(
        self,
        data_root:  str | Path,
        split:      str | None  = None,
        image_size: int         = 224,
        sets:       list | None = None,
        train_mode: bool | None = None,
    ):
        assert split is not None or sets is not None, "Provide split or sets"
        if split is not None and sets is None:
            assert split in SPLIT_MAP, f"split must be one of {list(SPLIT_MAP)}"
        self.data_root = Path(data_root)

        labels = pd.read_csv(self.data_root / "labels.csv")
        sets_  = pd.read_csv(self.data_root / "sets.csv", index_col=0)

        # Keep only rows that belong to the requested sets
        active_sets = sets if sets is not None else SPLIT_MAP[split]
        labels = labels[labels["Set"].isin(active_sets)].reset_index(drop=True)

        # Drop rows without an image or without a valid wear label
        labels = labels[labels["ImageFile"].notna()].reset_index(drop=True)
        labels = labels[labels["wear"].notna()].reset_index(drop=True)

        # Build per-set crop lookup: {set_number: (left, top, right, bottom)}
        self._crops: dict[int, tuple] = {}
        for idx_label, row in sets_.iterrows():
            # sets.csv index is "Set 1", "Set 2", etc.
            set_num = int(str(idx_label).replace("Set ", "").strip())
            self._crops[set_num] = _parse_crop(row["crop"])

        self.labels = labels

        # Augmentation: explicit train_mode overrides split-based inference
        use_aug = train_mode if train_mode is not None else (split == "train")
        if use_aug:
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
        return len(self.labels)

    def __getitem__(self, idx: int):
        row = self.labels.iloc[idx]
        set_num = int(row["Set"])

        # ImageFile in labels.csv is "MATWI/Set1/images/..." → strip "MATWI/"
        rel_path = str(row["ImageFile"]).replace("MATWI/", "", 1)
        img_path = self.data_root / rel_path

        image = Image.open(img_path).convert("RGB")

        # Apply per-set crop
        crop = self._crops[set_num]  # (left, top, right, bottom)
        image = image.crop(crop)

        image = self.transform(image)
        wear  = torch.tensor(float(row["wear"]), dtype=torch.float32)

        return image, wear
