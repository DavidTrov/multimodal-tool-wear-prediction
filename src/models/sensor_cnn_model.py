"""
Sensor-only CNN regressor on CWT scalograms.

Architecture: simple 4-block VGG-style stack (~61 K params).
Uses GroupNorm(G=8) instead of BatchNorm — BN degrades at batch sizes 16-32
under SGDM due to noisy batch statistics (NeurIPS 2021 unified normalisation study).
BatchNorm was tested in Phase 4 experiment 12 and produced worse results (40.45 µm vs 30.36 µm).
"""

import torch
import torch.nn as nn


def _gn(channels: int) -> nn.GroupNorm:
    """GroupNorm with G=8, falling back to G=4 or G=channels for small channel counts."""
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)


class SensorCNNRegressor(nn.Module):
    """Sensor-only wear regressor on (5, 64, 64) CWT scalograms."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1, bias=False),
            _gn(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            _gn(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            _gn(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            _gn(64), nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
