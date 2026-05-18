"""
Phase 5 — improved scalogram fusion model.

Architecture
------------
Image encoder  : pretrained ResNet18 (Phase-1 weights), fc → Identity (frozen)
                 Input  : (B, 3, 224, 224)  →  f_img    : (B, 512)

Sensor encoder : MultiScaleSensorCNN.extract_features()
                 warm-start from Phase-4 checkpoint
                 Input  : (B, 5,  64,  64)  →  f_sensor : (B,  96)

Normalisation  : LayerNorm(512) on f_img, LayerNorm(96) on f_sensor

Fusion head    : Linear(608, 1)  →  P_final

Auxiliary head : Linear(96, 1)   →  P_aux  (training only)
                 Forces the sensor branch to independently predict wear,
                 preventing modality laziness where the frozen image branch
                 dominates and the sensor branch is ignored.

Training loss  : HuberLoss(δ=20)(P_final, y) + λ * HuberLoss(δ=20)(P_aux, y)
                 λ = 0.2 (same as Phase 3b)

forward() always returns (P_final, P_aux).
At eval time use model(img, scal)[0].
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from src.models.multiscale_sensor_cnn import MultiScaleSensorCNN

IMAGE_FEAT_DIM  = 512
SENSOR_FEAT_DIM = 96   # MultiScaleSensorCNN features before head


class MultiScaleFusionModel(nn.Module):

    def __init__(self):
        super().__init__()

        # ── Image encoder ──────────────────────────────────────────────────────
        # fc is Identity here so forward() always outputs (B, 512).
        # load_phase1_weights() temporarily swaps in Linear(512→1), loads the
        # Phase-1 state dict, then restores Identity.
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.image_encoder = backbone

        # ── Sensor encoder ─────────────────────────────────────────────────────
        self.sensor_cnn = MultiScaleSensorCNN()

        # ── Per-modality normalisation ─────────────────────────────────────────
        self.image_norm  = nn.LayerNorm(IMAGE_FEAT_DIM)
        self.sensor_norm = nn.LayerNorm(SENSOR_FEAT_DIM)

        # ── Fusion head ────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(IMAGE_FEAT_DIM + SENSOR_FEAT_DIM, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        # ── Auxiliary sensor head (training only) ──────────────────────────────
        self.aux_head = nn.Linear(SENSOR_FEAT_DIM, 1)

    # ── Weight loading helpers ─────────────────────────────────────────────────

    def load_phase1_weights(self, ckpt_path, device: str = "cpu"):
        """Load Phase-1 state dict (which has fc as Linear(512→1)), then restore Identity."""
        self.image_encoder.fc = nn.Linear(512, 1)   # match Phase-1 structure
        self.image_encoder.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        self.image_encoder.fc = nn.Identity()        # restore for feature extraction

    def load_phase4_weights(self, ckpt_path, device: str = "cpu"):
        """Warm-start sensor encoder from Phase-4 multiscale checkpoint."""
        self.sensor_cnn.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )

    def freeze_image_encoder(self):
        """Freeze all ResNet18 parameters (called after load_phase1_weights)."""
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def freeze_sensor_encoder(self):
        """Freeze all MultiScaleSensorCNN parameters (called after load_phase4_weights)."""
        for param in self.sensor_cnn.parameters():
            param.requires_grad = False

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        image:     torch.Tensor,   # (B, 3, 224, 224)
        scalogram: torch.Tensor,   # (B, 5,  64,  64)
    ):
        f_img    = self.image_encoder(image)                    # (B, 512)
        f_sensor = self.sensor_cnn.extract_features(scalogram) # (B,  96)

        combined = torch.cat(
            [self.image_norm(f_img), self.sensor_norm(f_sensor)], dim=1
        )                                                       # (B, 608)

        p_final = self.head(combined)                           # (B,   1)
        p_aux   = self.aux_head(f_sensor)                       # (B,   1)

        return p_final, p_aux
