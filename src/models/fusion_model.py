import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

SENSOR_INPUT_DIM = 100   # 20 physics features × 5 channels (FFT + wavelet + time-domain)


class FusionModel(nn.Module):
    """
    Decision-level fusion: each modality makes an independent scalar prediction,
    then a learned linear blend combines them.

    Architecture
    ------------
    Image branch  : pretrained ResNet18 + Phase-1 regression head (frozen)
                    → P_img  (scalar wear estimate, µm)
    Sensor branch : Linear(sensor_input_dim → 1)
                    → P_sensor (scalar wear estimate, µm)
    Blend         : Linear(2 → 1) — learns w_img, w_sensor, bias
                    → P_final

    Trainable parameters: sensor_input_dim + 1 (sensor head) + 3 (blend).
    With 100 physics features: 104 total trainable parameters.

    The blend is initialised so the model starts as pure image-only
    (w_img=1, w_sensor=0, bias=0).  Training moves away from that only
    if the sensor prediction genuinely reduces the loss.

    sensor_mean / sensor_std are registered as buffers for input normalisation.
    """

    def __init__(
        self,
        sensor_input_dim: int = SENSOR_INPUT_DIM,
        sensor_mean: torch.Tensor | None = None,
        sensor_std:  torch.Tensor | None = None,
    ):
        super().__init__()

        # ── Image branch (frozen after Phase-1 weights are loaded) ────────────
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
        self.image_model = backbone

        # ── Sensor normalisation buffers ──────────────────────────────────────
        if sensor_mean is None:
            sensor_mean = torch.zeros(sensor_input_dim)
        if sensor_std is None:
            sensor_std = torch.ones(sensor_input_dim)
        self.register_buffer("sensor_mean", sensor_mean)
        self.register_buffer("sensor_std",  sensor_std)

        # ── Sensor branch: single linear layer ────────────────────────────────
        self.sensor_head = nn.Linear(sensor_input_dim, 1)

        # ── Blend: learned weighted average of the two predictions ────────────
        # Initialised to image-only (w_img=1, w_sensor=0, bias=0)
        self.blend = nn.Linear(2, 1)
        nn.init.constant_(self.blend.weight, 0.0)
        nn.init.constant_(self.blend.bias,   0.0)
        with torch.no_grad():
            self.blend.weight[0, 0] = 1.0   # w_img = 1 at init

    def forward(self, image: torch.Tensor, sensor: torch.Tensor) -> torch.Tensor:
        sensor = (sensor - self.sensor_mean) / (self.sensor_std + 1e-8)

        p_img    = self.image_model(image)      # (B, 1)
        p_sensor = self.sensor_head(sensor)     # (B, 1)

        combined = torch.cat([p_img, p_sensor], dim=1)   # (B, 2)
        return self.blend(combined)                       # (B, 1)
