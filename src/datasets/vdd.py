"""VDD (Varied Drone Dataset) PyTorch Dataset class.

VDD has 7 semantic classes (masks are grayscale uint8, class ID per pixel):
    0 = other
    1 = wall
    2 = road
    3 = vegetation
    4 = vehicle
    5 = roof
    6 = water

Expected folder layout:
    root/
    ├── train/
    │   ├── src/  (.JPG images, 4000x3000)
    │   └── gt/   (.png grayscale masks, same basename as image)
    ├── val/
    │   ├── src/
    │   └── gt/
    └── test/
        ├── src/
        └── gt/
"""
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# Class ID -> human-readable name
VDD_CLASSES = {
    0: "other",
    1: "wall",
    2: "road",
    3: "vegetation",
    4: "vehicle",
    5: "roof",
    6: "water",
}

# Class ID -> RGB color (for visualization overlays)
VDD_COLOR_MAP = np.array(
    [
        [  0,   0,   0],  # 0 other       - black
        [128,   0,   0],  # 1 wall        - dark red
        [128,  64, 128],  # 2 road        - purple
        [  0, 128,   0],  # 3 vegetation  - green
        [ 64,   0, 128],  # 4 vehicle     - violet
        [128, 128,   0],  # 5 roof        - olive
        [  0,   0, 128],  # 6 water       - dark blue
    ],
    dtype=np.uint8,
)


class VDDDataset(Dataset):
    """Varied Drone Dataset for semantic segmentation."""

    NUM_CLASSES = 7
    IMG_EXT = ".JPG"
    MASK_EXT = ".png"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            root: Path to VDD root (contains train/, val/, test/ subfolders).
                  Example: 'data/raw/VDD/VDD'
            split: One of 'train', 'val', 'test'.
            transform: An Albumentations transform with signature
                       transform(image=np.ndarray, mask=np.ndarray) -> dict.
                       If None, returns raw tensors in CHW float/long.
        """
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of train/val/test, got '{split}'")

        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.img_dir = self.root / split / "src"
        self.mask_dir = self.root / split / "gt"

        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        # Build list of image paths and validate pairs exist
        self.img_paths = sorted(self.img_dir.glob(f"*{self.IMG_EXT}"))
        if len(self.img_paths) == 0:
            raise RuntimeError(
                f"No '{self.IMG_EXT}' files found in {self.img_dir}. "
                f"Check that the dataset is extracted correctly."
            )

        # Sanity check: every image must have a paired mask
        missing = [
            p.name for p in self.img_paths
            if not (self.mask_dir / (p.stem + self.MASK_EXT)).is_file()
        ]
        if missing:
            raise RuntimeError(
                f"{len(missing)} images without matching mask. First few: {missing[:5]}"
            )

    # ------------------------------------------------------------------ API

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.img_paths[idx]
        mask_path = self.mask_dir / (img_path.stem + self.MASK_EXT)

        # Load image as RGB (cv2 reads BGR by default)
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # HxWx3 uint8

        # Load mask as single-channel class IDs
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        if mask.ndim != 2:
            raise RuntimeError(
                f"Expected single-channel mask, got shape {mask.shape} for {mask_path}"
            )

        if self.transform is not None:
            # Albumentations pipeline (it handles Normalize + ToTensorV2)
            out = self.transform(image=image, mask=mask)
            image = out["image"]
            mask = out["mask"].long() if torch.is_tensor(out["mask"]) else torch.from_numpy(out["mask"]).long()
        else:
            # No transform: return plain CHW float tensors in [0, 1] with long masks
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask

    # ------------------------------------------------------------ helpers

    @staticmethod
    def decode_segmap(mask: np.ndarray) -> np.ndarray:
        """Convert a class-ID mask (H, W) into an RGB image (H, W, 3)."""
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got {mask.shape}")
        return VDD_COLOR_MAP[mask]

    def get_class_distribution(self) -> dict:
        """Compute per-class pixel counts across the split (slow: reads all masks)."""
        from collections import Counter
        counter: Counter = Counter()
        for p in self.img_paths:
            mask_path = self.mask_dir / (p.stem + self.MASK_EXT)
            m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            vals, counts = np.unique(m, return_counts=True)
            for v, c in zip(vals, counts):
                counter[int(v)] += int(c)
        total = sum(counter.values())
        return {VDD_CLASSES.get(k, f"class_{k}"): v / total for k, v in sorted(counter.items())}
