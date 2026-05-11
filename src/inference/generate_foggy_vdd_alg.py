"""Apply Albumentations algorithmic fog to ALL VDD images (train + val + test).

Produces a foggy version of VDD using Albumentations.RandomFog with deterministic
parameters (no randomness across calls -- same input -> same output).

Output structure mirrors the original VDD:

    output_root/
        train/
            src/  (foggy .JPG)
            gt/   (.png masks, resized to save_size with NEAREST)
        val/
        test/

Two intensity presets matching the GAN ones:
  - medium: fog_coef in (0.3, 0.5), alpha_coef=0.08
  - dense : fog_coef in (0.6, 0.85), alpha_coef=0.12

Important: unlike the Pix2Pix-generated fog (which mimicked Koschmieder via the
training data's depth implicitly), Albumentations.RandomFog applies a UNIFORM
fog overlay -- it does NOT depend on depth. This is a limit of the method on
non-stereo / non-depth aerial imagery and should be discussed in the report.

Usage:
    python src/inference/generate_foggy_vdd_alg.py \\
        --vdd-root data/raw/VDD/VDD \\
        --output-root data/processed/VDD_foggy_alg_medium_768 \\
        --intensity medium \\
        --save-size 768

    # The same script handles dense by changing --intensity dense.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm


# Fog presets. We picked these by visual inspection so the algorithmic fog
# "looks comparable" to the GAN-generated fog at each severity. They are
# fixed (no randomness) to make the generation deterministic.
FOG_PRESETS = {
    "medium": {"fog_coef_lower": 0.3, "fog_coef_upper": 0.5, "alpha_coef": 0.08},
    "dense":  {"fog_coef_lower": 0.6, "fog_coef_upper": 0.85, "alpha_coef": 0.12},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vdd-root", default="data/raw/VDD/VDD")
    p.add_argument("--output-root", required=True)
    p.add_argument("--intensity", choices=list(FOG_PRESETS.keys()), required=True,
                   help="Fog intensity preset: 'medium' or 'dense'.")
    p.add_argument("--save-size", type=int, default=768,
                   help="Resolution at which foggy images AND masks are saved.")
    p.add_argument("--image-format", default="jpg", choices=["jpg", "png"])
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--seed", type=int, default=42,
                   help="Seed used to fix Albumentations randomness so the output "
                        "is reproducible across runs.")
    return p.parse_args()


def build_fog_transform(preset: dict, save_size: int) -> A.Compose:
    """Resize to save_size, then apply RandomFog with fixed parameters."""
    return A.Compose([
        A.Resize(save_size, save_size, interpolation=cv2.INTER_LINEAR),
        A.RandomFog(
            fog_coef_lower=preset["fog_coef_lower"],
            fog_coef_upper=preset["fog_coef_upper"],
            alpha_coef=preset["alpha_coef"],
            p=1.0,  # always apply
        ),
    ], additional_targets={})  # only image, no mask augmentation here


def resize_mask(mask: np.ndarray, save_size: int) -> np.ndarray:
    """NEAREST resize for class-ID masks (preserves discrete values)."""
    if mask.shape[0] == save_size and mask.shape[1] == save_size:
        return mask
    return cv2.resize(mask, (save_size, save_size), interpolation=cv2.INTER_NEAREST)


def process_split(
    split: str,
    transform: A.Compose,
    vdd_root: Path,
    output_root: Path,
    save_size: int,
    image_format: str,
    jpeg_quality: int,
    seed: int,
) -> int:
    """Apply fog to every image in the split, save with resized mask."""
    src_dir = vdd_root / split / "src"
    gt_dir = vdd_root / split / "gt"
    out_src_dir = output_root / split / "src"
    out_gt_dir = output_root / split / "gt"

    if not src_dir.is_dir():
        print(f"[skip] {src_dir} doesn't exist")
        return 0

    out_src_dir.mkdir(parents=True, exist_ok=True)
    out_gt_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(src_dir.glob("*.JPG"))
    if not img_paths:
        print(f"[skip] no .JPG files in {src_dir}")
        return 0

    print(f"\n[{split}] {len(img_paths)} images to process")

    if image_format == "jpg":
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        out_ext = ".JPG"
    else:
        encode_params = []
        out_ext = ".PNG"

    n_done = 0
    for src_path in tqdm(img_paths, desc=f"  fog[{split}]", ncols=100):
        # Read image (BGR)
        img_bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read: {src_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Apply fog (Albumentations expects RGB)
        # Seed Albumentations' rng deterministically per-image for reproducibility
        # We mix global seed with the file's stem hash so different files get
        # slightly different fog patterns but the same file always gets the same one.
        rng_seed = (seed * 1000003 + abs(hash(src_path.stem))) % (2**32)
        np.random.seed(rng_seed)
        out = transform(image=img_rgb)
        foggy_rgb = out["image"]

        # Back to BGR for cv2 save
        foggy_bgr = cv2.cvtColor(foggy_rgb, cv2.COLOR_RGB2BGR)
        out_img = out_src_dir / f"{src_path.stem}{out_ext}"
        ok = cv2.imwrite(str(out_img), foggy_bgr, encode_params)
        if not ok:
            raise RuntimeError(f"Failed to write {out_img}")

        # Resize and save mask
        mask_src = gt_dir / f"{src_path.stem}.png"
        if not mask_src.is_file():
            raise RuntimeError(f"Missing mask: {mask_src}")
        mask = cv2.imread(str(mask_src), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_src}")
        mask_resized = resize_mask(mask, save_size)
        mask_dst = out_gt_dir / f"{src_path.stem}.png"
        ok = cv2.imwrite(str(mask_dst), mask_resized)
        if not ok:
            raise RuntimeError(f"Failed to write mask {mask_dst}")

        n_done += 1
    return n_done


def copy_metadata(vdd_root: Path, output_root: Path) -> None:
    src = vdd_root / "metadata"
    if not src.is_dir():
        print(f"[note] No 'metadata' folder in {vdd_root}; skipping.")
        return
    dst = output_root / "metadata"
    if dst.is_dir():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"[ok] copied metadata to {dst}")


def main():
    args = parse_args()
    preset = FOG_PRESETS[args.intensity]
    print(f"[INFO] Intensity: {args.intensity}  preset: {preset}")
    print(f"[INFO] Save size : {args.save_size}")
    print(f"[INFO] Seed      : {args.seed}")

    vdd_root = Path(args.vdd_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    transform = build_fog_transform(preset, args.save_size)
    copy_metadata(vdd_root, output_root)

    t0 = time.perf_counter()
    total = 0
    for split in ["train", "val", "test"]:
        total += process_split(
            split=split,
            transform=transform,
            vdd_root=vdd_root,
            output_root=output_root,
            save_size=args.save_size,
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
            seed=args.seed,
        )
    elapsed = time.perf_counter() - t0

    print()
    print("=" * 60)
    print("ALGORITHMIC FOG GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total images: {total}")
    print(f"Total time  : {elapsed:.1f} s ({elapsed/max(total,1)*1000:.0f} ms/img)")
    print(f"Output root : {output_root.resolve()}")


if __name__ == "__main__":
    main()
