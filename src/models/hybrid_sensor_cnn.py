"""
Hybrid sensor CNN: CWT scalogram branch + force scalar features.

Uses MultiScaleSensorCNN on acc+acoustic CWT scalograms (2, 64, 64) for
high-frequency vibration patterns, and a small MLP on force scalar features
(RMS, kurtosis, FFT bands, etc.) for magnitude/trend information that CWT
cannot capture (frequencies below 25 Hz).

The two branches are concatenated before the regression head.
"""

import torch
import torch.nn as nn

from src.models.multiscale_sensor_cnn import (
    _gn, _ResBlock, _CBAM,
)


class HybridSensorCNN(nn.Module):
    """
    Architecture
    ------------
    CWT branch (identical to MultiScaleSensorCNN feature extractor):
        Multi-scale entry → features → GAP → 96-dim

    Force branch:
        Linear(n_force, 32) → ReLU → Linear(32, 16) → ReLU → 16-dim

    Fusion:
        Concat(96 + 16 = 112) → Dropout(0.3) → Linear(112, 1)
    """

    def __init__(self, in_channels: int = 2, n_force: int = 69, dropout: float = 0.3):
        super().__init__()

        # ── CWT branch (acc + acoustic) ────────────────────────────────────────
        self.path_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 1, bias=False),
            _gn(16), nn.ReLU(inplace=True),
        )
        self.path_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 8, 1, bias=False),
            _gn(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 24, 3, padding=1, bias=False),
            _gn(24), nn.ReLU(inplace=True),
        )
        self.path_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 4, 1, bias=False),
            _gn(4), nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, 5, padding=2, bias=False),
            _gn(8), nn.ReLU(inplace=True),
        )

        self.features = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1, bias=False),
            _gn(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            _ResBlock(64),
            _CBAM(64),
            nn.Conv2d(64, 96, 3, padding=1, bias=False),
            _gn(96), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 96, 3, padding=1, bias=False),
            _gn(96), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # ── Force scalar branch ────────────────────────────────────────────────
        self.force_branch = nn.Sequential(
            nn.Linear(n_force, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
        )

        # ── Regression head ────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(96 + 16, 1),
        )

    def forward(self, scalogram: torch.Tensor, force_feats: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.path_1x1(scalogram), self.path_3x3(scalogram), self.path_5x5(scalogram)], dim=1)
        cnn_out = self.features(x)
        force_out = self.force_branch(force_feats)
        return self.head(torch.cat([cnn_out, force_out], dim=1))
