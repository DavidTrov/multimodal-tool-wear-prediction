"""
Sensor-only CNN regressor on CWT scalograms.

Designed to be deployable to STM32F401RC (256 KB flash, 64 KB SRAM):
  - INT8 weight footprint after quantization fits flash
  - Peak activation in SRAM fits 64 KB
  - Input tensor (5, 64, 64) at INT8: 20 KB

Architecture: simple 4-block VGG-style stack (~61 K params).
"""

import torch
import torch.nn as nn


class SensorCNNRegressor(nn.Module):
    """Sensor-only wear regressor on (5, 64, 64) CWT scalograms."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
