"""U-Net model factory for VDD semantic segmentation.

Uses segmentation_models_pytorch (smp), a well-maintained library that gives
us U-Net (and other architectures) with swappable backbones pretrained on ImageNet.

Reference:
    https://github.com/qubvel-org/segmentation_models.pytorch
"""
from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_unet(
    num_classes: int = 7,
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
) -> nn.Module:
    """Build a U-Net for semantic segmentation.

    Args:
        num_classes: number of semantic classes (VDD has 7).
        encoder_name: smp-supported backbone. Options include:
            - 'resnet18', 'resnet34' (lightweight, recommended default)
            - 'resnet50' (stronger, ~2x heavier)
            - 'efficientnet-b0', 'efficientnet-b3' (parameter-efficient)
            - 'mobilenet_v2' (very fast, less accurate)
            Full list: https://smp.readthedocs.io/en/latest/encoders.html
        encoder_weights: 'imagenet' (pretrained) or None (train from scratch).
        in_channels: usually 3 (RGB).

    Returns:
        A torch.nn.Module. Its forward(x) with x of shape (B, 3, H, W) returns
        logits of shape (B, num_classes, H, W).
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
    return model


def count_parameters(model: nn.Module) -> tuple:
    """Return (trainable_params, total_params) as integers."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def human_readable(n: int) -> str:
    """Format a large integer as '24.4M' or '1.3B'."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)
