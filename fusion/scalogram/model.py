"""
Scalogram fusion model — mid-level fusion with auxiliary sensor loss.

Architecture
------------
Image encoder   : pretrained ResNet18, fc replaced with Identity after
                  Phase-1 weights are loaded.  Frozen during Phase-3 training.
                  Input  : (B, 3, 224, 224)  →  f_img    : (B, 512)

Sensor encoder  : lightweight SensorCNN on stacked CWT scalograms.
                  Input  : (B, 5, 64, 64)    →  f_sensor : (B, 64)

Fusion          : LayerNorm per modality → concat(576) → Linear(576→1) → P_final

Auxiliary head  : Linear(64→1) on raw sensor features → P_aux
                  Used only during training to force the sensor encoder to
                  independently learn wear-relevant features, preventing the
                  head from ignoring the sensor branch in favour of the already-
                  calibrated image features (modality laziness).
                  Discarded at evaluation — only P_final is used.

Training loss   : MSE(P_final, y)  +  λ * MSE(P_aux, y)   (λ = 0.2)

forward() always returns (P_final, P_aux).
At eval time call model(img, scal)[0] to get only P_final.
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

IMAGE_FEAT_DIM  = 512
SENSOR_FEAT_DIM = 64


class SensorCNN(nn.Module):
    """Lightweight CNN for a (5, 64, 64) stacked CWT scalogram.
    Outputs a 64-dimensional feature vector.
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
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),               # (B, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScalogramFusionModel(nn.Module):
    """
    Mid-level fusion with auxiliary sensor loss.
    Returns (P_final, P_aux) from forward().
    """

    def __init__(self):
        super().__init__()

        # ── Image encoder ─────────────────────────────────────────────────────
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)  # matches Phase-1
        self.image_encoder = backbone

        # ── Sensor encoder ────────────────────────────────────────────────────
        self.sensor_cnn = SensorCNN(out_dim=SENSOR_FEAT_DIM)

        # ── Per-modality normalisation ────────────────────────────────────────
        self.image_norm  = nn.LayerNorm(IMAGE_FEAT_DIM)
        self.sensor_norm = nn.LayerNorm(SENSOR_FEAT_DIM)

        # ── Shared regression head ────────────────────────────────────────────
        self.head = nn.Linear(IMAGE_FEAT_DIM + SENSOR_FEAT_DIM, 1)

        # ── Auxiliary sensor head (training only) ─────────────────────────────
        # Forces the sensor encoder to independently predict wear so it cannot
        # be ignored by the shared head in favour of the frozen image branch.
        self.sensor_aux_head = nn.Linear(SENSOR_FEAT_DIM, 1)

    def load_phase1_weights(self, ckpt_path, device="cpu"):
        """Load Phase-1 state dict then replace fc with Identity."""
        self.image_encoder.load_state_dict(
            torch.load(ckpt_path, map_location=device)
        )
        self.image_encoder.fc = nn.Identity()

    def freeze_image_encoder(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        image:     torch.Tensor,   # (B, 3, 224, 224)
        scalogram: torch.Tensor,   # (B, 5,  64,  64)
    ):
        f_img    = self.image_encoder(image)       # (B, 512)
        f_sensor = self.sensor_cnn(scalogram)      # (B,  64)

        # Main prediction
        combined = torch.cat(
            [self.image_norm(f_img), self.sensor_norm(f_sensor)], dim=1
        )                                          # (B, 576)
        p_final = self.head(combined)              # (B,   1)

        # Auxiliary prediction (sensor branch only)
        p_aux = self.sensor_aux_head(f_sensor)     # (B,   1)

        return p_final, p_aux
