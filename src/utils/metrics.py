import torch


def mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return (preds - targets).abs().mean().item()
