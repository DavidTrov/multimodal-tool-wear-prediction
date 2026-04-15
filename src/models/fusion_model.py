import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

SENSOR_INPUT_DIM = 50


class FusionModel(nn.Module):
    """
    Decision-level fusion: each modality makes an independent scalar prediction,
    then a learned linear blend combines them.

    Architecture
    ------------
    Image branch  : pretrained ResNet18 + Phase-1 regression head (frozen)
                    → P_img  (scalar wear estimate)
    Sensor branch : Linear(50 → 1)  (tiny, cannot overfit)
                    → P_sensor (scalar wear estimate)
    Blend         : Linear(2 → 1)   (learns w_img, w_sensor, bias — 3 params)
                    → P_final

    Trainable parameters: sensor head (51) + blend (3) = 54 total.
    With 647 training samples this ratio is safe by any standard.

    The blend layer is initialised so that at epoch 0 the model outputs
    the image-only prediction (w_img=1, w_sensor=0, bias=0).  Training
    then learns how much — if at all — to trust the sensor estimate.

    sensor_mean / sensor_std are registered as buffers (not trained).
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
        self.image_model = backbone        # full model, including regression head

        # ── Sensor normalisation buffers ──────────────────────────────────────
        if sensor_mean is None:
            sensor_mean = torch.zeros(sensor_input_dim)
        if sensor_std is None:
            sensor_std = torch.ones(sensor_input_dim)
        self.register_buffer("sensor_mean", sensor_mean)
        self.register_buffer("sensor_std",  sensor_std)

        # ── Sensor branch: single linear layer, 51 parameters ─────────────────
        self.sensor_head = nn.Linear(sensor_input_dim, 1)

        # ── Blend: learned weighted average of the two scalar predictions ─────
        # Initialised to image-only (w_img=1, w_sensor=0, bias=0) so training
        # starts from the known-good Phase-1 baseline and only moves away if
        # the sensor signal genuinely helps.
        self.blend = nn.Linear(2, 1)
        nn.init.constant_(self.blend.weight, 0.0)   # both weights start at 0
        nn.init.constant_(self.blend.bias,   0.0)
        with torch.no_grad():
            self.blend.weight[0, 0] = 1.0            # w_img  = 1
            # w_sensor stays 0 — model starts as image-only

    def forward(self, image: torch.Tensor, sensor: torch.Tensor) -> torch.Tensor:
        # Normalise sensor inputs
        sensor = (sensor - self.sensor_mean) / (self.sensor_std + 1e-8)

        p_img    = self.image_model(image)      # (B, 1)
        p_sensor = self.sensor_head(sensor)     # (B, 1)

        combined = torch.cat([p_img, p_sensor], dim=1)   # (B, 2)
        return self.blend(combined)                       # (B, 1)
