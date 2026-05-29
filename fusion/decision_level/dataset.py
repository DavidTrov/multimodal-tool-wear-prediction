import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import SPLIT_MAP, IMAGENET_MEAN, IMAGENET_STD, _parse_crop

_CHANNELS   = ["acc", "acoustic", "fx", "fy", "fz"]
_TIME_FEATS = ["rms", "kurtosis", "crest_factor", "skewness", "shape_factor"]
_FFT_FEATS  = [f"fft_band_{i}" for i in range(8)] + ["spectral_centroid", "hf_energy_ratio"]
_WAV_FEATS  = ["wavelet_a4", "wavelet_d4", "wavelet_d3", "wavelet_d2", "wavelet_d1"]

FEATURE_COLS = [
    f"{ch}__{feat}"
    for ch in _CHANNELS
    for feat in _TIME_FEATS + _FFT_FEATS + _WAV_FEATS
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

        # Sensor features — nan_to_num guards against any NaN/Inf in extracted features
        sensor = torch.tensor(row[FEATURE_COLS].values.astype("float32"), dtype=torch.float32)
        sensor = torch.nan_to_num(sensor, nan=0.0, posinf=0.0, neginf=0.0)

        wear = torch.tensor(float(row["wear"]), dtype=torch.float32)

        return image, sensor, wear
