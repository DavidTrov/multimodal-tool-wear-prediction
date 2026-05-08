"""
Multiscale pyramid CNN for CWT scalogram wear regression.

Adapts Zhang 2023 (Sensors 23(10), 4595) inception-style entry block to the
continuous-regression setting on MATWI (5 channels, (5, 64, 64) scalograms).

Changes from the original baseline:
  - Parallel 1×1 / 3×3 / 5×5 paths at the input [Zhang 2023]
  - GroupNorm(G=8) replaces BatchNorm throughout — BN degrades under SGDM at
    batch sizes 16-32 due to noisy batch statistics (NeurIPS 2021)
  - One residual block in the feature extractor — skip connections provide
    gradient highways for SGDM's noisier updates (Keskar et al. 2017)
  - Dropout(0.1) before the regression head [Zhang §4.1]
  - ~174K parameters — fits 2MB flash at INT8
"""

import torch
import torch.nn as nn


def _gn(channels: int) -> nn.GroupNorm:
    """GroupNorm with G=8, falling back for small channel counts."""
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)


class _ResBlock(nn.Module):
    """Two-conv residual block with GroupNorm. Input and output channels are equal."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _gn(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _gn(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class MultiScaleSensorCNN(nn.Module):
    """
    Inception-style multiscale CNN regressor on (5, 64, 64) CWT scalograms.

    Architecture
    ------------
    Entry block — three parallel paths (no spatial downsampling):
        path_1x1 : Conv(5→16, 1×1) → GN → ReLU
        path_3x3 : Conv(5→8,  1×1) → GN → ReLU → Conv(8→24,  3×3) → GN → ReLU
        path_5x5 : Conv(5→4,  1×1) → GN → ReLU → Conv(4→8,   5×5) → GN → ReLU
        → concat → 48 channels, 64×64

    Feature extractor:
        MaxPool(2)                              → (48, 32, 32)
        Conv(48→64, 3×3) → GN → ReLU
        MaxPool(2)                              → (64, 16, 16)
        ResBlock(64)           ← skip connection
        Conv(64→96, 3×3) → GN → ReLU
        MaxPool(2)                              → (96, 8, 8)
        Conv(96→96, 3×3) → GN → ReLU           → (96, 8, 8)
        AdaptiveAvgPool2d(1) → Flatten          → 96

    Head:
        Dropout(0.1) → Linear(96, 1)
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()

        # ── Multi-scale entry ──────────────────────────────────────────────────
        self.path_1x1 = nn.Sequential(
            nn.Conv2d(5, 16, 1, bias=False),
            _gn(16), nn.ReLU(inplace=True),
        )
        self.path_3x3 = nn.Sequential(
            nn.Conv2d(5, 8, 1, bias=False),
            _gn(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 24, 3, padding=1, bias=False),
            _gn(24), nn.ReLU(inplace=True),
        )
        self.path_5x5 = nn.Sequential(
            nn.Conv2d(5, 4, 1, bias=False),
            _gn(4), nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, 5, padding=2, bias=False),
            _gn(8), nn.ReLU(inplace=True),
        )
        # concat → 16 + 24 + 8 = 48 channels

        # ── Feature extractor ──────────────────────────────────────────────────
        self.features = nn.Sequential(
            nn.MaxPool2d(2),                                    # (48, 32, 32)

            nn.Conv2d(48, 64, 3, padding=1, bias=False),
            _gn(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                    # (64, 16, 16)

            _ResBlock(64),                                      # skip connection

            nn.Conv2d(64, 96, 3, padding=1, bias=False),
            _gn(96), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                    # (96, 8, 8)

            nn.Conv2d(96, 96, 3, padding=1, bias=False),
            _gn(96), nn.ReLU(inplace=True),                    # (96, 8, 8)

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
