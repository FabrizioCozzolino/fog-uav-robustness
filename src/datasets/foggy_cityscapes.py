"""Foggy Cityscapes paired dataset for Pix2Pix training.

The dataset has 500 street-level images, available in three fog conditions:
  - No_Fog/      (500 .png)   <-- the "input"  (clean image)
  - Medium_Fog/  (500 .png)   <-- one possible "target" (medium fog)
  - Dense_Fog/   (500 .png)   <-- the other possible "target" (dense fog)

Files are named identically across the three folders (`001.png`, `002.png`, ...
`500.png`), so the same basename in two folders is the same scene with different
fog level. This is "paired" image-to-image translation.

Key differences vs VDDDataset:
  - returns (clean_image, foggy_image), both 3-channel float tensors
  - both images are normalized to [-1, +1] (the Pix2Pix convention, since the
    generator's last layer is tanh)
  - augmentations must be applied IDENTICALLY to both images, otherwise we
    break the pair. Albumentations supports this via `additional_targets`.
"""
from pathlib import Path
from typing import Callable, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


# Tanh-friendly normalization (range [-1, +1])
TANH_MEAN = (0.5, 0.5, 0.5)
TANH_STD = (0.5, 0.5, 0.5)


class FoggyCityscapesPairedDataset(Dataset):
    """Paired clean->foggy translation dataset.

    Args:
        root: path to the Foggy_Cityscapes folder (the one that contains
              No_Fog/, Medium_Fog/, Dense_Fog/).
        fog_level: 'medium' -> uses Medium_Fog as target.
                   'dense'  -> uses Dense_Fog as target.
        transform: an Albumentations Compose. If None, uses the default
                   (resize to image_size + tanh-normalize + ToTensor).
        image_size: only used when transform=None.
        split: 'train' or 'val'. If 'train', uses the first 90% of the
               filenames; if 'val', the last 10%. Used to monitor GAN training.
    """

    def __init__(
        self,
        root: str,
        fog_level: str = "medium",
        transform: Optional[Callable] = None,
        image_size: int = 256,
        split: str = "train",
    ) -> None:
        if fog_level == "medium":
            target_subdir = "Medium_Fog"
        elif fog_level == "dense":
            target_subdir = "Dense_Fog"
        else:
            raise ValueError(f"fog_level must be 'medium' or 'dense', got '{fog_level}'")
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        self.root = Path(root)
        self.fog_level = fog_level
        self.split = split
        self.clean_dir = self.root / "No_Fog"
        self.foggy_dir = self.root / target_subdir

        if not self.clean_dir.is_dir():
            raise FileNotFoundError(f"Clean folder not found: {self.clean_dir}")
        if not self.foggy_dir.is_dir():
            raise FileNotFoundError(f"Foggy folder not found: {self.foggy_dir}")

        # Build sorted list of basenames available in BOTH folders
        clean_names = sorted(p.name for p in self.clean_dir.glob("*.png"))
        foggy_names = set(p.name for p in self.foggy_dir.glob("*.png"))
        paired = [n for n in clean_names if n in foggy_names]

        if not paired:
            raise RuntimeError(
                f"No paired images found between {self.clean_dir} and {self.foggy_dir}"
            )

        # Deterministic train/val split: 90% / 10%
        n_total = len(paired)
        n_train = int(round(n_total * 0.9))
        if split == "train":
            self.basenames = paired[:n_train]
        else:
            self.basenames = paired[n_train:]

        # Default transform if none provided
        if transform is None:
            transform = self._default_transform(image_size, train=(split == "train"))
        self.transform = transform

    @staticmethod
    def _default_transform(image_size: int, train: bool) -> A.Compose:
        """A reasonable default transform if the caller passes none.

        The augmentations must be applied identically to both images. We use
        Albumentations' `additional_targets={'foggy': 'image'}` to declare
        that the second key is also an image and should follow the same
        random transform.
        """
        ops = [A.Resize(height=image_size, width=image_size, interpolation=1)]
        if train:
            ops += [A.HorizontalFlip(p=0.5)]
        ops += [
            A.Normalize(mean=TANH_MEAN, std=TANH_STD),
            ToTensorV2(),
        ]
        return A.Compose(ops, additional_targets={"foggy": "image"})

    def __len__(self) -> int:
        return len(self.basenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self.basenames[idx]
        clean_path = self.clean_dir / name
        foggy_path = self.foggy_dir / name

        clean = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
        foggy = cv2.imread(str(foggy_path), cv2.IMREAD_COLOR)
        if clean is None:
            raise RuntimeError(f"Failed to read clean image: {clean_path}")
        if foggy is None:
            raise RuntimeError(f"Failed to read foggy image: {foggy_path}")

        # OpenCV reads BGR, the rest of our stack expects RGB
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        foggy = cv2.cvtColor(foggy, cv2.COLOR_BGR2RGB)

        # Albumentations will apply identical random transforms to both
        out = self.transform(image=clean, foggy=foggy)
        clean_t = out["image"]
        foggy_t = out["foggy"]
        # If transform didn't return tensors (shouldn't happen with our default), convert
        if not torch.is_tensor(clean_t):
            clean_t = torch.from_numpy(clean_t).permute(2, 0, 1).float() / 127.5 - 1.0
            foggy_t = torch.from_numpy(foggy_t).permute(2, 0, 1).float() / 127.5 - 1.0

        return clean_t, foggy_t


def denormalize_tanh(tensor: torch.Tensor) -> torch.Tensor:
    """Undo tanh-style normalization (from [-1, +1] back to [0, 1]).

    Useful to display / save images. Accepts (C, H, W) or (B, C, H, W).
    """
    out = (tensor + 1.0) / 2.0
    return out.clamp(0.0, 1.0)
