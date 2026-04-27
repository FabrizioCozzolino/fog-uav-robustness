"""Quick visual sanity check for the VDD dataset.

Displays a few image/mask pairs side-by-side, with an overlay view.
Saves the figure to outputs/figures/vdd_samples.png.

Usage:
    python scripts/visualize_vdd.py
    python scripts/visualize_vdd.py --split train --n 4
"""
import argparse
import sys
from pathlib import Path

# Allow 'src.*' imports when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.datasets.vdd import VDDDataset, VDD_CLASSES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/raw/VDD/VDD",
                   help="VDD root folder (the one with train/val/test subfolders)")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--n", type=int, default=4, help="Number of samples to display")
    p.add_argument("--out", default="outputs/figures/vdd_samples.png")
    return p.parse_args()


def main():
    args = parse_args()

    ds = VDDDataset(root=args.root, split=args.split, transform=None)
    print(f"[OK] Loaded VDD '{args.split}' split: {len(ds)} samples")

    # Equally spaced sample indices
    n = min(args.n, len(ds))
    indices = np.linspace(0, len(ds) - 1, n, dtype=int)

    fig, axes = plt.subplots(n, 3, figsize=(14, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for row, idx in enumerate(indices):
        image, mask = ds[idx]
        img_np = image.permute(1, 2, 0).numpy()     # H x W x 3, float [0, 1]
        mask_np = mask.numpy()                      # H x W, int
        mask_rgb = VDDDataset.decode_segmap(mask_np)  # H x W x 3 uint8

        # 50/50 blend
        overlay = (0.55 * img_np + 0.45 * (mask_rgb / 255.0)).clip(0, 1)

        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f"Image  (idx={idx}, {ds.img_paths[idx].name})", fontsize=10)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask_rgb)
        axes[row, 1].set_title("Mask (colorized)", fontsize=10)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay", fontsize=10)
        axes[row, 2].axis("off")

        present = np.unique(mask_np).tolist()
        present_names = [f"{v}={VDD_CLASSES.get(v, '?')}" for v in present]
        print(f"  sample {idx:3d}  image shape={img_np.shape}  classes={present_names}")

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=80, bbox_inches="tight")
    print(f"\n[OK] Saved figure to: {out_path.resolve()}")

    # Also print the color legend once for reference
    print("\nColor legend (class_id = name):")
    for k, v in VDD_CLASSES.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
