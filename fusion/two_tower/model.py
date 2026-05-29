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

    def __init__(self, image_feat_dim: int = IMAGE_FEAT_DIM):
        """
        Args
        ----
        image_feat_dim : dimensionality of the image encoder's penultimate
                         feature vector.  512 for standard ResNet18; use the
                         actual avgpool output size for pruned/compressed models
                         (e.g. 309 for the 2M-param budget model).
        """
        super().__init__()

        self._image_feat_dim = image_feat_dim

        # ── Image encoder ──────────────────────────────────────────────────────
        # fc is Identity here so forward() always outputs (B, image_feat_dim).
        # load_phase1_weights() temporarily swaps in Linear(512→1), loads the
        # Phase-1 state dict, then restores Identity.
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.image_encoder = backbone

        # ── Sensor encoder ─────────────────────────────────────────────────────
        self.sensor_cnn = MultiScaleSensorCNN()

        # ── Per-modality normalisation ─────────────────────────────────────────
        self.image_norm  = nn.LayerNorm(image_feat_dim)
        self.sensor_norm = nn.LayerNorm(SENSOR_FEAT_DIM)

        # ── Per-modality projection towers ─────────────────────────────────────
        # Each branch is projected to the same 128-dim space before merging.
        # This gives each modality its own non-linear transformation and ensures
        # equal gradient flow (128 dims each) regardless of original dimensionality.
        self.img_proj = nn.Sequential(
            nn.Linear(image_feat_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.sen_proj = nn.Sequential(
            nn.Linear(SENSOR_FEAT_DIM, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        # ── Fusion head ────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
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

    def load_pruned_sensor_encoder(self, ckpt_path, device: str = "cpu"):
        """Load a pruned/distilled sensor CNN saved as a full model object.

        Pruned models have non-standard channel counts and are saved with
        torch.save(model, path) rather than torch.save(model.state_dict(), path).
        This method loads the full object and installs it as self.sensor_cnn.

        The pruned model's extract_features() output is kept at 96-d by protecting
        model.head[1] during pruning, so sensor_norm and sen_proj need no changes.
        """
        obj = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(obj, dict):
            raise ValueError(
                f"{ckpt_path} contains a state_dict, not a full model object. "
                "Pruned sensor models must be saved with torch.save(model, path)."
            )
        self.sensor_cnn = obj.to(device)

    def load_compressed_image_encoder(self, ckpt_path, device: str = "cpu"):
        """Load a pruned/compressed ResNet saved as a full model object.

        Pruned models have non-standard channel counts and are saved with
        torch.save(model, path) rather than torch.save(model.state_dict(), path).
        This method loads the full object, strips the fc head (replacing it with
        Identity), and installs it as self.image_encoder.

        The caller must ensure MultiScaleFusionModel was constructed with the
        correct image_feat_dim matching the compressed model's avgpool output.
        """
        obj = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(obj, dict):
            raise ValueError(
                f"{ckpt_path} contains a state_dict, not a full model object. "
                "Compressed models must be saved with torch.save(model, path)."
            )
        obj.fc = nn.Identity()
        self.image_encoder = obj.to(device)

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

        h_img    = self.img_proj(self.image_norm(f_img))        # (B, 128)
        h_sensor = self.sen_proj(self.sensor_norm(f_sensor))    # (B, 128)

        combined = torch.cat([h_img, h_sensor], dim=1)          # (B, 256)

        p_final = self.head(combined)                           # (B,   1)
        p_aux   = self.aux_head(f_sensor)                       # (B,   1)

        return p_final, p_aux
