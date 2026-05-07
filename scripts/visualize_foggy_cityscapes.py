"""Visual sanity check for the paired Foggy Cityscapes dataset.

Loads N samples and saves a figure showing clean image, foggy image, and the
absolute difference (so you can see *where* the fog was added).

Usage:
    python scripts/visualize_foggy_cityscapes.py
    python scripts/visualize_foggy_cityscapes.py --fog-level dense --n 4
"""
import argparse
import sys
from pathlib import Path

# Allow `from src.*` imports when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.datasets.foggy_cityscapes import (
    FoggyCityscapesPairedDataset,
    denormalize_tanh,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/raw/foggy_cityscapes/Foggy_Cityscapes")
    p.add_argument("--fog-level", default="medium", choices=["medium", "dense"])
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--n", type=int, default=4, help="Number of samples to display")
    p.add_argument("--out", default=None,
                   help="Output figure path. Default: outputs/figures/foggy_<level>.png")
    return p.parse_args()


def main():
    args = parse_args()

    ds = FoggyCityscapesPairedDataset(
        root=args.root,
        fog_level=args.fog_level,
        image_size=args.image_size,
        split=args.split,
    )
    print(f"[OK] Loaded Foggy_Cityscapes paired ({args.fog_level}, {args.split}): "
          f"{len(ds)} samples")

    n = min(args.n, len(ds))
    indices = np.linspace(0, len(ds) - 1, n, dtype=int)

    fig, axes = plt.subplots(n, 3, figsize=(14, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for row, idx in enumerate(indices):
        clean_t, foggy_t = ds[idx]
        clean = denormalize_tanh(clean_t).permute(1, 2, 0).numpy()
        foggy = denormalize_tanh(foggy_t).permute(1, 2, 0).numpy()
        # Difference visualization (amplified)
        diff = np.abs(foggy - clean)
        diff = (diff / max(diff.max(), 1e-8)).clip(0, 1)

        axes[row, 0].imshow(clean)
        axes[row, 0].set_title(f"Clean (No_Fog)  idx={idx}  '{ds.basenames[idx]}'")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(foggy)
        axes[row, 1].set_title(f"Foggy ({args.fog_level})")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(diff)
        axes[row, 2].set_title("|foggy - clean|  (normalized)")
        axes[row, 2].axis("off")

        print(f"  sample {idx:3d}: shapes clean={clean.shape}, foggy={foggy.shape}; "
              f"diff_mean={diff.mean():.3f}")

    plt.tight_layout()
    if args.out is None:
        args.out = f"outputs/figures/foggy_{args.fog_level}_{args.split}.png"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=80, bbox_inches="tight")
    print(f"\n[OK] Saved figure to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
