import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

SENSOR_INPUT_DIM = 50


class SensorEncoder(nn.Module):
    """Small MLP that embeds tsfresh features into a 128-dim vector."""

    def __init__(self, input_dim: int = SENSOR_INPUT_DIM, embed_dim: int = 128):
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


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention, adapted from MDFNet.

    Applied to the concatenated [image + sensor] feature vector after fusion.
    Each of the `in_dim` channels gets its own independent sigmoid gate in
    [0, 1] — no zero-sum competition between modalities.

    Architecture:
        [in_dim] → Linear(in_dim → in_dim // reduction) → ReLU
                 → Linear(in_dim // reduction → in_dim) → Sigmoid
                 → elementwise scale of input
    """

    def __init__(self, in_dim: int, reduction: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(in_dim, in_dim // reduction),
            nn.ReLU(),
            nn.Linear(in_dim // reduction, in_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)   # elementwise scale — (B, in_dim)


class FusionModel(nn.Module):
    """
    Multimodal fusion model for tool wear regression — SE-attention variant.

    Inspired by MDFNet (Chen et al. 2022): concatenate modality features,
    then apply independent sigmoid channel attention (SE block) so each
    feature dimension is weighted on its own merit — no zero-sum softmax
    competition between modalities.

    Architecture
    ------------
    Image branch  : pretrained ResNet18 → LayerNorm → 512-dim
    Sensor branch : normalise → MLP → LayerNorm → 128-dim
    Fusion        : Concat [640] → SE channel attention → [640]
    Head          : MLP [640 → 256 → 64 → 1]

    sensor_mean / sensor_std are pre-computed from the training set and
    registered as buffers so normalisation is baked into the model.
    """

    def __init__(
        self,
        sensor_input_dim: int = SENSOR_INPUT_DIM,
        sensor_embed_dim: int = 128,
        sensor_mean: torch.Tensor | None = None,
        sensor_std:  torch.Tensor | None = None,
    ):
        super().__init__()

        # ── Image branch ──────────────────────────────────────────────────────
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        image_out_dim = backbone.fc.in_features   # 512
        backbone.fc = nn.Identity()
        self.image_encoder = backbone
        self.image_norm    = nn.LayerNorm(image_out_dim)

        # ── Sensor normalisation buffers ──────────────────────────────────────
        if sensor_mean is None:
            sensor_mean = torch.zeros(sensor_input_dim)
        if sensor_std is None:
            sensor_std = torch.ones(sensor_input_dim)
        self.register_buffer("sensor_mean", sensor_mean)
        self.register_buffer("sensor_std",  sensor_std)

        # ── Sensor branch ─────────────────────────────────────────────────────
        self.sensor_encoder = SensorEncoder(sensor_input_dim, sensor_embed_dim)
        self.sensor_norm    = nn.LayerNorm(sensor_embed_dim)

        # ── SE channel attention on fused vector ──────────────────────────────
        fused_dim = image_out_dim + sensor_embed_dim   # 640
        self.se_block = SEBlock(fused_dim, reduction=16)

        # ── Fusion head (640 → 1) ─────────────────────────────────────────────
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, image: torch.Tensor, sensor: torch.Tensor) -> torch.Tensor:
        # Normalise sensor inputs using training-set statistics
        sensor = (sensor - self.sensor_mean) / (self.sensor_std + 1e-8)

        img_feat    = self.image_norm(self.image_encoder(image))   # (B, 512)
        sensor_feat = self.sensor_norm(self.sensor_encoder(sensor)) # (B, 128)

        fused   = torch.cat([img_feat, sensor_feat], dim=1)        # (B, 640)
        fused   = self.se_block(fused)                             # (B, 640) — attended
        return self.fusion_head(fused)                             # (B, 1)
