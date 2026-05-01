"""
Sensor-only CNN regressors on CWT scalograms.

Designed to be deployable to STM32F401RC (256 KB flash, 64 KB SRAM):
  - INT8 weight footprint after quantization fits flash
  - Peak activation in SRAM fits 64 KB
  - Input tensor (5, 64, 64) at INT8: 20 KB

Two architectures are provided:

  SensorCNNRegressor      — simple 4-block VGG-style stack (~61 K params)
  MultiScaleSensorCNN     — Inception-style multiscale feature pyramid
                            adapted from Zhang et al. (Sensors 2023, 23, 4595)
                            for scalar wear regression (~28 K params)

The multiscale variant captures both fine local texture (3×3 branch) and
broader time-frequency structure (5×5 branch) at the same spatial scale,
fused with the stem output.  The original paper used this scheme for
3-class wear-stage classification on PHM 2010 force signals; here it is
adapted for continuous wear regression on MATWI's 5-channel sensor stack.
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Simple 4-block VGG-style stack (kept for comparison)
# ─────────────────────────────────────────────────────────────────────────────
class SensorCNNRegressor(nn.Module):
    """Simple sensor-only wear regressor on (5, 64, 64) CWT scalograms."""

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


# ─────────────────────────────────────────────────────────────────────────────
# Multiscale feature pyramid (adapted from Zhang et al. 2023)
# ─────────────────────────────────────────────────────────────────────────────
class MultiScaleSensorCNN(nn.Module):
    """
    Multiscale feature pyramid for CWT scalogram regression.

    Architecture
    ------------
    Stem        :  Conv(5→16, 1×1, s=2) + BN + MaxPool(2) + ReLU
                   (5, 64, 64) → (16, 16, 16)

    Three parallel paths from the stem output:
      • Skip path : MaxPool(2) only                 → (16, 8, 8)
      • 3×3 branch: Conv(1×1) → Conv(3×3) + MaxPool → (16, 8, 8)
      • 5×5 branch: Conv(1×1) → Conv(5×5) + MaxPool → (16, 8, 8)

    Fusion      : concat along channel axis        → (48, 8, 8)

    Head        : Conv(48→32, 3×3) + BN + ReLU + Dropout(0.1)
                  Conv(32→16, 3×3) + BN + ReLU
                  AdaptiveAvgPool(1×1) + Flatten   → (16,)
                  Linear(16 → 1)                   → (1,)

    Total parameters: ~28 K  →  ~28 KB at INT8
    """

    def __init__(self):
        super().__init__()

        # ── Stem ──────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
        )   # (5, 64, 64) → (16, 16, 16)

        # ── 3×3 branch (1×1 bottleneck → 3×3 spatial) ─────────────────────────
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
        )   # (16, 16, 16) → (16, 8, 8)

        # ── 5×5 branch (1×1 bottleneck → 5×5 spatial) ─────────────────────────
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
        )   # (16, 16, 16) → (16, 8, 8)

        # ── Skip path: just downsample the stem to (16, 8, 8) ─────────────────
        self.skip_pool = nn.MaxPool2d(2)

        # ── Head: convs after concat (48 channels) ────────────────────────────
        self.head_convs = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.stem(x)                # (B, 16, 16, 16)

        y1 = self.branch_3x3(y0)         # (B, 16,  8,  8)
        y2 = self.branch_5x5(y0)         # (B, 16,  8,  8)
        y0 = self.skip_pool(y0)          # (B, 16,  8,  8)

        # Two-step concatenation matches Zhang et al.: (skip, 3×3), then (·, 5×5)
        z = torch.cat([y0, y1], dim=1)   # (B, 32, 8, 8)
        z = torch.cat([z,  y2], dim=1)   # (B, 48, 8, 8)

        z = self.head_convs(z)            # (B, 16)
        return self.head(z)               # (B,  1)
