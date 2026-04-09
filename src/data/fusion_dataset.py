import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.dataset import SPLIT_MAP, IMAGENET_MEAN, IMAGENET_STD, _parse_crop

FEATURE_COLS = [
    "acc__sum_values", "acc__median", "acc__mean", "acc__length",
    "acc__standard_deviation", "acc__variance", "acc__root_mean_square",
    "acc__maximum", "acc__absolute_maximum", "acc__minimum",
    "acoustic__sum_values", "acoustic__median", "acoustic__mean", "acoustic__length",
    "acoustic__standard_deviation", "acoustic__variance", "acoustic__root_mean_square",
    "acoustic__maximum", "acoustic__absolute_maximum", "acoustic__minimum",
    "fx__sum_values", "fx__median", "fx__mean", "fx__length",
    "fx__standard_deviation", "fx__variance", "fx__root_mean_square",
    "fx__maximum", "fx__absolute_maximum", "fx__minimum",
    "fy__sum_values", "fy__median", "fy__mean", "fy__length",
    "fy__standard_deviation", "fy__variance", "fy__root_mean_square",
    "fy__maximum", "fy__absolute_maximum", "fy__minimum",
    "fz__sum_values", "fz__median", "fz__mean", "fz__length",
    "fz__standard_deviation", "fz__variance", "fz__root_mean_square",
    "fz__maximum", "fz__absolute_maximum", "fz__minimum",
]


class MATWIFusionDataset(Dataset):
    """
    Multimodal dataset for MATWI: returns (image tensor, sensor features, wear).

    Only includes rows that have both a valid image and sensor features.

    Args:
        data_root:     path to data/raw/
        features_path: path to data/processed/sensor_features.parquet
        split:         "train", "val", or "test"
        image_size:    resize target (default 224)
    """

    def __init__(
        self,
        data_root: str | Path,
        features_path: str | Path,
        split: str,
        image_size: int = 224,
    ):
        assert split in SPLIT_MAP, f"split must be one of {list(SPLIT_MAP)}"
        self.data_root = Path(data_root)

        # Load sensor features — only rows with a valid ImageFile
        feats = pd.read_parquet(features_path)
        feats = feats[feats["ImageFile"].str.len() > 0]
        feats = feats[feats["Set"].isin(SPLIT_MAP[split])].reset_index(drop=True)

        # Crop lookup from sets.csv
        sets = pd.read_csv(self.data_root / "sets.csv", index_col=0)
        self._crops: dict[int, tuple] = {}
        for idx_label, row in sets.iterrows():
            set_num = int(str(idx_label).replace("Set ", "").strip())
            self._crops[set_num] = _parse_crop(row["crop"])

        self.feats = feats
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.feats)

    def __getitem__(self, idx: int):
        row     = self.feats.iloc[idx]
        set_num = int(row["Set"])

        # Load image
        rel_path = str(row["ImageFile"]).replace("MATWI/", "", 1)
        image    = Image.open(self.data_root / rel_path).convert("RGB")
        image    = image.crop(self._crops[set_num])
        image    = self.transform(image)

        # Sensor features
        sensor = torch.tensor(row[FEATURE_COLS].values.astype("float32"), dtype=torch.float32)

        wear = torch.tensor(float(row["wear"]), dtype=torch.float32)

        return image, sensor, wear
