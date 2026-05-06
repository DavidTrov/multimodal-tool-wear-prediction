"""
Multiscale pyramid CNN for CWT scalogram wear regression.

Adapts Zhang 2023 (Sensors 23(10), 4595) inception-style entry block to the
continuous-regression setting on MATWI (5 channels, (5, 64, 64) scalograms).

Key differences from the baseline SensorCNNRegressor:
  - Parallel 1×1 / 3×3 / 5×5 paths at the input (multi-scale feature extraction)
  - Dropout(0.1) before the regression head [Zhang §4.1]
  - ~170K parameters — ~2.8× the baseline; INT8 ≈ 166 KB (fits 2MB flash)

Reference
---------
Zhang Y., Qi X., Wang T., He Y.  "Tool Wear State Recognition Based on Multi-Scale
Convolutional Neural Network with Coordinate Attention."  Sensors 23(10), 4595, 2023.
"""

import torch
import torch.nn as nn


class MultiScaleSensorCNN(nn.Module):
    """
    Inception-style multiscale CNN regressor on (5, 64, 64) CWT scalograms.

    Architecture
    ------------
    Entry block — three parallel paths (no spatial downsampling):
        path_1x1 : Conv(5→16, 1×1) → BN → ReLU
        path_3x3 : Conv(5→8, 1×1) → BN → ReLU → Conv(8→24, 3×3) → BN → ReLU
        path_5x5 : Conv(5→4, 1×1) → BN → ReLU → Conv(4→8, 5×5) → BN → ReLU
        → concat → 48 channels, 64×64

    Feature extractor (standard VGG-style after the entry):
        MaxPool(2) → 48, 32×32
        Conv(48→64, 3×3) → BN → ReLU
        MaxPool(2) → 64, 16×16
        Conv(64→96, 3×3) → BN → ReLU
        MaxPool(2) → 96, 8×8
        Conv(96→96, 3×3) → BN → ReLU
        AdaptiveAvgPool2d(1) → 96
        Flatten → 96

    Head:
        Dropout(0.1) → Linear(96, 1)
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()

        # ── Multi-scale entry ──────────────────────────────────────────────────
        self.path_1x1 = nn.Sequential(
            nn.Conv2d(5, 16, 1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )
        self.path_3x3 = nn.Sequential(
            nn.Conv2d(5, 8, 1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24), nn.ReLU(inplace=True),
        )
        self.path_5x5 = nn.Sequential(
            nn.Conv2d(5, 4, 1, bias=False),
            nn.BatchNorm2d(4), nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, 5, padding=2, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
        )
        # concat → 16 + 24 + 8 = 48 channels

        # ── Feature extractor ──────────────────────────────────────────────────
        self.features = nn.Sequential(
            nn.MaxPool2d(2),                                    # (48, 32, 32)

            nn.Conv2d(48, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                    # (64, 16, 16)

            nn.Conv2d(64, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                    # (96, 8, 8)

            nn.Conv2d(96, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),         # (96, 8, 8)

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),                                       # (96,)
        )

        # ── Regression head ────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.path_1x1(x), self.path_3x3(x), self.path_5x5(x)], dim=1)
        return self.head(self.features(x))
