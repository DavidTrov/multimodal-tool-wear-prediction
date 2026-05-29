"""
Shared configuration constants used across all modalities.
"""

import ast

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
