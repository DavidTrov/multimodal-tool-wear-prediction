import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18_regressor() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model
