"""Albumentations transform pipelines for VDD.

Two entry points:
    - get_train_transform(image_size): augmentation + normalize + tensor
    - get_eval_transform(image_size):  resize + normalize + tensor (deterministic)

We normalize with ImageNet statistics because we will use a U-Net with a backbone
pretrained on ImageNet (ResNet / EfficientNet). All backbones in segmentation_models_pytorch
expect this normalization by default.
"""
from typing import Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet statistics (for pretrained backbones)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transform(image_size: int = 512) -> A.Compose:
    """Training pipeline with geometric + photometric augmentations.

    Args:
        image_size: output square side (e.g., 512 -> images become 512x512).
    """
    return A.Compose(
        [
            # --- Geometric ---
            A.Resize(height=image_size, width=image_size, interpolation=1),  # bilinear for image
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),  # rotates by 0, 90, 180, or 270 degrees

            # --- Photometric (applied only to the image, not the mask) ---
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.3,
            ),

            # --- Final: normalize + to tensor ---
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_eval_transform(image_size: int = 512) -> A.Compose:
    """Deterministic pipeline for validation and testing: only resize + normalize."""
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size, interpolation=1),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def denormalize(tensor, mean: Tuple[float, ...] = IMAGENET_MEAN,
                std: Tuple[float, ...] = IMAGENET_STD):
    """Undo ImageNet normalization on a (C, H, W) or (B, C, H, W) tensor.

    Useful for visualizing a batch that has already been normalized.
    Returns a float tensor in roughly [0, 1] range.
    """
    import torch
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    mean_t = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    std_t = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    out = tensor * std_t + mean_t
    out = out.clamp(0.0, 1.0)
    return out.squeeze(0) if out.shape[0] == 1 else out
