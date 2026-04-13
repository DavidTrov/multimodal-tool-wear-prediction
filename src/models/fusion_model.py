import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

SENSOR_INPUT_DIM = 50


class SensorEncoder(nn.Module):
    """Small MLP that embeds tsfresh features into a dense vector."""

    def __init__(self, input_dim: int = SENSOR_INPUT_DIM, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModalityGating(nn.Module):
    """
    Learns a scalar gate per modality: how much to trust image vs sensor.

    Given both feature vectors, outputs two weights that sum to 1 (softmax).
    Each modality is scaled by its weight before concatenation.

    With only 2050 parameters this adds negligible size.
    """

    def __init__(self, img_dim: int, sensor_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(img_dim + sensor_dim, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, img_feat: torch.Tensor, sensor_feat: torch.Tensor):
        weights    = self.gate(torch.cat([img_feat, sensor_feat], dim=1))  # (B, 2)
        img_w      = weights[:, 0].unsqueeze(1)   # (B, 1)
        sensor_w   = weights[:, 1].unsqueeze(1)   # (B, 1)
        return img_feat * img_w, sensor_feat * sensor_w


class FusionModel(nn.Module):
    """
    Multimodal fusion model for tool wear regression.

    Image branch:  pretrained ResNet18 → 512-dim (unchanged)
    Sensor branch: normalise inputs → MLP → 64-dim → projection → 128-dim
    Gating:        learned per-sample weights → scale each modality
    Fusion head:   concat(512 + 128) → MLP → scalar wear prediction

    sensor_mean / sensor_std are pre-computed from the training set and
    registered as buffers so normalisation is baked into the model.
    """

    def __init__(
        self,
        sensor_input_dim: int = SENSOR_INPUT_DIM,
        sensor_embed_dim: int = 64,
        proj_dim: int = 128,
        sensor_mean: torch.Tensor | None = None,
        sensor_std:  torch.Tensor | None = None,
    ):
        super().__init__()

        # Image branch — strip the classification head
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        image_out_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.image_encoder = backbone

        # Sensor input normalisation — stored as buffers (not trained)
        # Falls back to no-op if statistics are not provided
        if sensor_mean is None:
            sensor_mean = torch.zeros(sensor_input_dim)
        if sensor_std is None:
            sensor_std = torch.ones(sensor_input_dim)
        self.register_buffer("sensor_mean", sensor_mean)
        self.register_buffer("sensor_std",  sensor_std)

        # Sensor branch
        self.sensor_encoder = SensorEncoder(sensor_input_dim, sensor_embed_dim)

        # Project sensor to 128-dim
        self.sensor_proj = nn.Sequential(
            nn.Linear(sensor_embed_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # LayerNorm — normalise both branches to the same scale before fusion
        self.image_norm  = nn.LayerNorm(image_out_dim)
        self.sensor_norm = nn.LayerNorm(proj_dim)

        # Gating — learned per-sample modality weights
        self.gating = ModalityGating(image_out_dim, proj_dim)

        # Fusion head
        self.fusion_head = nn.Sequential(
            nn.Linear(image_out_dim + proj_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, image: torch.Tensor, sensor: torch.Tensor) -> torch.Tensor:
        # Normalise sensor inputs using training set statistics
        sensor = (sensor - self.sensor_mean) / (self.sensor_std + 1e-8)

        img_feat    = self.image_norm(self.image_encoder(image))                      # (B, 512)
        sensor_feat = self.sensor_norm(self.sensor_proj(self.sensor_encoder(sensor))) # (B, 128)
        img_feat, sensor_feat = self.gating(img_feat, sensor_feat)                   # scaled by learned weights
        fused       = torch.cat([img_feat, sensor_feat], dim=1)                       # (B, 640)
        return self.fusion_head(fused)                                                # (B, 1)
