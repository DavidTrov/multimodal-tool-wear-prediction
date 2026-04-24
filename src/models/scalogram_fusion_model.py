"""
Scalogram fusion model — Phase 3b  (mid-level fusion).

Architecture
------------
Image encoder   : pretrained ResNet18, conv layers frozen after Phase-1 weights
                  are loaded, regression head replaced by Identity.
                  Input  : (B, 3, 224, 224)  →  f_img    : (B, 512)

Sensor encoder  : lightweight SensorCNN operating on stacked CWT scalograms.
                  Input  : (B, 5, 64, 64)    →  f_sensor : (B, 64)

Normalisation   : LayerNorm applied independently to f_img and f_sensor before
                  concatenation, so both modalities enter the shared head on a
                  comparable scale regardless of their initialisation state.

Shared head     : Linear(576 → 1)  — learns the joint regression from image +
                  sensor features simultaneously, with gradient flowing back into
                  the sensor encoder from the very first step.

Why mid-level instead of late (decision-level) fusion
------------------------------------------------------
In late fusion each branch must independently output a calibrated wear value
(µm).  The blend is initialised with w_sensor = 0, which blocks gradient to
the sensor branch initially and biases the model towards the image prediction
even after training.  Mid-level fusion removes this asymmetry: the head sees
both feature vectors at once, gradient flows to the sensor CNN from step one,
and the head can discover cross-modal interactions.

Trainable parameters  (image conv layers are frozen)
----------------------------------------------------
  SensorCNN    ≈ 24 K
  LayerNorm ×2 ≈  1 K
  Head Linear  ≈  0.6 K
  Total        ≈ 26 K  — well within safe range for ~650 training samples.

SensorCNN design (outputs 64-d features, no internal regression head)
----------------------------------------------------------------------
  Conv(5→16,  3×3) + BN + ReLU + MaxPool(2)  →  (B, 16, 32, 32)
  Conv(16→32, 3×3) + BN + ReLU + MaxPool(2)  →  (B, 32, 16, 16)
  Conv(32→64, 3×3) + BN + ReLU + MaxPool(2)  →  (B, 64,  8,  8)
  Conv(64→64, 3×3) + BN + ReLU               →  (B, 64,  8,  8)
  AdaptiveAvgPool(1×1) + Flatten              →  (B, 64)
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

IMAGE_FEAT_DIM  = 512   # ResNet18 penultimate layer
SENSOR_FEAT_DIM = 64    # SensorCNN output


class SensorCNN(nn.Module):
    """Lightweight CNN for a (5, 64, 64) stacked CWT scalogram.
    Outputs a 64-dimensional feature vector (no regression head).
    """

    def __init__(self, out_dim: int = SENSOR_FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1  →  (B, 16, 32, 32)
            nn.Conv2d(5, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2  →  (B, 32, 16, 16)
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3  →  (B, 64, 8, 8)
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4  →  (B, 64, 8, 8)
            nn.Conv2d(64, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            # Global pooling
            nn.AdaptiveAvgPool2d(1),    # (B, out_dim, 1, 1)
            nn.Flatten(),               # (B, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScalogramFusionModel(nn.Module):
    """
    Mid-level fusion: image feature vector + sensor feature vector →
    LayerNorm → concat → shared Linear head → wear prediction (µm).
    """

    def __init__(self):
        super().__init__()

        # ── Image encoder ─────────────────────────────────────────────────────
        # We build with a real fc first so Phase-1 weights can be loaded
        # cleanly.  The training script replaces fc → Identity after loading.
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)   # matches Phase-1
        self.image_encoder = backbone

        # ── Sensor encoder ────────────────────────────────────────────────────
        self.sensor_cnn = SensorCNN(out_dim=SENSOR_FEAT_DIM)

        # ── Per-modality normalisation ────────────────────────────────────────
        # Keeps image features (already scaled by Phase-1 training) and sensor
        # features (random at init) on a comparable scale so the shared head
        # can weight them fairly from the very first step.
        self.image_norm  = nn.LayerNorm(IMAGE_FEAT_DIM)
        self.sensor_norm = nn.LayerNorm(SENSOR_FEAT_DIM)

        # ── Shared regression head ────────────────────────────────────────────
        self.head = nn.Linear(IMAGE_FEAT_DIM + SENSOR_FEAT_DIM, 1)

    def load_phase1_weights(self, ckpt_path, device="cpu"):
        """
        Load Phase-1 state dict (backbone + regression head), then swap the
        fc layer for nn.Identity so the encoder outputs a 512-d feature vector.
        Call this before freezing the image encoder.
        """
        self.image_encoder.load_state_dict(
            torch.load(ckpt_path, map_location=device)
        )
        self.image_encoder.fc = nn.Identity()

    def freeze_image_encoder(self):
        """Freeze all image encoder parameters."""
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        image:     torch.Tensor,   # (B, 3, 224, 224)
        scalogram: torch.Tensor,   # (B, 5,  64,  64)
    ) -> torch.Tensor:             # (B, 1)

        f_img    = self.image_encoder(image)       # (B, 512)
        f_sensor = self.sensor_cnn(scalogram)      # (B,  64)

        # Normalise each modality independently before fusing
        f_img    = self.image_norm(f_img)          # (B, 512)
        f_sensor = self.sensor_norm(f_sensor)      # (B,  64)

        combined = torch.cat([f_img, f_sensor], dim=1)   # (B, 576)
        return self.head(combined)                        # (B,   1)
