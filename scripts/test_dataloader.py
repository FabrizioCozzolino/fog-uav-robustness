"""End-to-end test of the VDD dataloader.

Does three things:
  1. Creates a DataLoader with the training transforms.
  2. Iterates through a configurable number of batches and measures loading time.
  3. Saves a visualization of the first batch (images + masks) to verify
     augmentations and normalization work correctly.

Usage:
    python scripts/test_dataloader.py
    python scripts/test_dataloader.py --batch-size 4 --num-batches 10
"""
import argparse
import sys
import time
from pathlib import Path

# Allow 'src.*' imports when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.vdd import VDDDataset, VDD_CLASSES
from src.utils.transforms import get_train_transform, get_eval_transform, denormalize


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/raw/VDD/VDD")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-batches", type=int, default=5,
                   help="How many batches to iterate for timing (0 = all)")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers. Keep 0 on Windows for first run.")
    p.add_argument("--out", default="outputs/figures/vdd_batch.png")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Build dataset + dataloader ---
    transform = (get_train_transform(args.image_size) if args.split == "train"
                 else get_eval_transform(args.image_size))
    ds = VDDDataset(root=args.root, split=args.split, transform=transform)
    print(f"[OK] Dataset '{args.split}': {len(ds)} samples, image size = {args.image_size}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(args.split == "train"),
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    print(f"[OK] DataLoader: batch_size={args.batch_size}, "
          f"num_workers={args.num_workers}, "
          f"#batches={len(loader)}")

    # --- Iterate and time ---
    num_to_iter = len(loader) if args.num_batches == 0 else min(args.num_batches, len(loader))
    print(f"\nIterating {num_to_iter} batch(es) for timing...")

    t0 = time.perf_counter()
    first_batch = None
    total_images = 0
    for i, (images, masks) in enumerate(loader):
        if i == 0:
            first_batch = (images.clone(), masks.clone())
            # Sanity checks on shapes/dtypes on the first batch
            assert images.ndim == 4 and images.shape[1] == 3, f"bad image shape {images.shape}"
            assert masks.ndim == 3, f"bad mask shape {masks.shape}"
            assert images.dtype == torch.float32, f"image dtype {images.dtype}"
            assert masks.dtype == torch.long, f"mask dtype {masks.dtype}"
            print(f"  batch 0 : images {tuple(images.shape)} {images.dtype}  |  "
                  f"masks {tuple(masks.shape)} {masks.dtype}")
            print(f"            image value range = [{images.min():.2f}, {images.max():.2f}]")
            print(f"            mask unique values = {torch.unique(masks).tolist()}")
        total_images += images.shape[0]
        if i + 1 >= num_to_iter:
            break
    elapsed = time.perf_counter() - t0

    print(f"\n[TIMING] {total_images} images in {elapsed:.2f}s "
          f"→ {total_images/elapsed:.1f} img/s "
          f"({elapsed/total_images*1000:.0f} ms/img)")

    # --- Visualize the first batch ---
    images, masks = first_batch
    images_denorm = denormalize(images)  # (B, 3, H, W) in [0, 1]

    n = images.shape[0]
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        img = images_denorm[i].permute(1, 2, 0).cpu().numpy()
        mask_np = masks[i].cpu().numpy()
        mask_rgb = VDDDataset.decode_segmap(mask_np)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image [{i}]  shape={img.shape}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask_rgb)
        unique = np.unique(mask_np).tolist()
        present = ", ".join(VDD_CLASSES.get(v, "?") for v in unique)
        axes[i, 1].set_title(f"Mask [{i}]  classes: {present}", fontsize=9)
        axes[i, 1].axis("off")

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=80, bbox_inches="tight")
    print(f"\n[OK] Saved batch visualization to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
