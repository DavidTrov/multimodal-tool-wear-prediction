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


class FusionModel(nn.Module):
    """
    Multimodal fusion model for tool wear regression.

    Image branch:  pretrained ResNet18 → 512-dim feature vector
    Sensor branch: MLP on tsfresh features → 64-dim feature vector
    Fusion head:   concat(512 + 64) → MLP → scalar wear prediction
    """

    def __init__(self, sensor_input_dim: int = SENSOR_INPUT_DIM, sensor_embed_dim: int = 64):
        super().__init__()

        # Image branch — strip the classification head
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        image_out_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.image_encoder = backbone

        # Sensor branch
        self.sensor_encoder = SensorEncoder(sensor_input_dim, sensor_embed_dim)

        # Fusion head
        fused_dim = image_out_dim + sensor_embed_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, image: torch.Tensor, sensor: torch.Tensor) -> torch.Tensor:
        img_feat    = self.image_encoder(image)        # (B, 512)
        sensor_feat = self.sensor_encoder(sensor)      # (B, 64)
        fused       = torch.cat([img_feat, sensor_feat], dim=1)  # (B, 576)
        return self.fusion_head(fused)                 # (B, 1)
