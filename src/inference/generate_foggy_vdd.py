"""Apply a trained Pix2Pix Generator to ALL VDD images (train + val + test).

Produces a "foggy" version of VDD with masks resized to match.

    output_root/
        train/
            src/  (foggy .JPG, generated)
            gt/   (.png masks, RESIZED with nearest-neighbor to save_size)
        val/   (same structure)
        test/  (same structure)

Important: masks are resized with INTER_NEAREST so class IDs are preserved.
This is mandatory: image and mask MUST have the same H x W when fed to the
segmentation pipeline (Albumentations enforces this).

Usage:
    python src/inference/generate_foggy_vdd.py \\
        --generator outputs/runs/pix2pix_medium_baseline/G_best.pth \\
        --vdd-root data/raw/VDD/VDD \\
        --output-root data/processed/VDD_foggy_medium_768 \\
        --apply-size 768 \\
        --save-size 768
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.models.gan.pix2pix import Pix2PixGenerator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--generator", required=True)
    p.add_argument("--vdd-root", default="data/raw/VDD/VDD")
    p.add_argument("--output-root", required=True)
    p.add_argument("--apply-size", type=int, default=768)
    p.add_argument("--save-size", type=int, default=768)
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-format", default="jpg", choices=["jpg", "png"])
    p.add_argument("--jpeg-quality", type=int, default=95)
    return p.parse_args()


def pick_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_generator(ckpt_path: Path, encoder_name: str, device: torch.device) -> Pix2PixGenerator:
    print(f"[INFO] Loading generator: {ckpt_path}")
    G = Pix2PixGenerator(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        out_channels=3,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "G_state_dict" not in ckpt:
        raise RuntimeError(f"Checkpoint {ckpt_path} doesn't have 'G_state_dict' key")
    G.load_state_dict(ckpt["G_state_dict"])
    G.eval()
    epoch = ckpt.get("epoch", "?")
    val_l1 = ckpt.get("val_L1", None)
    if isinstance(val_l1, float):
        print(f"[INFO] Generator loaded: epoch={epoch}, val_L1={val_l1:.4f}")
    else:
        print(f"[INFO] Generator loaded: epoch={epoch}")
    return G


def preprocess_batch(images_bgr, apply_size, device):
    batch = []
    for img_bgr in images_bgr:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (apply_size, apply_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = torch.from_numpy(img).permute(2, 0, 1)
        batch.append(img)
    return torch.stack(batch, dim=0).to(device, non_blocking=True)


def postprocess_batch(out_tensor, save_size):
    out = (out_tensor.detach().clamp(-1.0, 1.0) + 1.0) * 127.5
    out = out.cpu().numpy().astype(np.uint8)
    out_imgs = []
    for arr in out:
        arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[0] != save_size or arr.shape[1] != save_size:
            arr = cv2.resize(arr, (save_size, save_size), interpolation=cv2.INTER_LINEAR)
        out_imgs.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return out_imgs


def resize_mask(mask, save_size):
    """Resize a class-ID mask using NEAREST interpolation (preserves discrete values)."""
    if mask.shape[0] == save_size and mask.shape[1] == save_size:
        return mask
    return cv2.resize(mask, (save_size, save_size), interpolation=cv2.INTER_NEAREST)


@torch.no_grad()
def process_split(split, G, vdd_root, output_root, apply_size, save_size,
                  batch_size, device, image_format, jpeg_quality):
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
    pbar = tqdm(range(0, len(img_paths), batch_size),
                desc=f"  generate[{split}]", ncols=100)
    for i in pbar:
        paths_chunk = img_paths[i:i + batch_size]
        imgs_bgr = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths_chunk]
        if any(im is None for im in imgs_bgr):
            bad = [str(p) for p, im in zip(paths_chunk, imgs_bgr) if im is None]
            raise RuntimeError(f"Failed to read: {bad}")
        x = preprocess_batch(imgs_bgr, apply_size, device)
        y = G(x)
        out_bgr_list = postprocess_batch(y, save_size)
        for src_path, foggy_bgr in zip(paths_chunk, out_bgr_list):
            stem = src_path.stem
            out_img = out_src_dir / f"{stem}{out_ext}"
            ok = cv2.imwrite(str(out_img), foggy_bgr, encode_params)
            if not ok:
                raise RuntimeError(f"Failed to write {out_img}")
            mask_src = gt_dir / f"{stem}.png"
            if not mask_src.is_file():
                raise RuntimeError(f"Missing mask: {mask_src}")
            mask = cv2.imread(str(mask_src), cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise RuntimeError(f"Failed to read mask: {mask_src}")
            mask_resized = resize_mask(mask, save_size)
            mask_dst = out_gt_dir / f"{stem}.png"
            ok = cv2.imwrite(str(mask_dst), mask_resized)
            if not ok:
                raise RuntimeError(f"Failed to write mask {mask_dst}")
            n_done += 1

    return n_done


def copy_metadata(vdd_root, output_root):
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
    device = pick_device(args.device)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] apply_size = {args.apply_size}")
    print(f"[INFO] save_size  = {args.save_size}")

    vdd_root = Path(args.vdd_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    G = load_generator(Path(args.generator), args.encoder, device)
    copy_metadata(vdd_root, output_root)

    t0 = time.perf_counter()
    total = 0
    for split in ["train", "val", "test"]:
        total += process_split(
            split=split, G=G, vdd_root=vdd_root, output_root=output_root,
            apply_size=args.apply_size, save_size=args.save_size,
            batch_size=args.batch_size, device=device,
            image_format=args.image_format, jpeg_quality=args.jpeg_quality,
        )
    elapsed = time.perf_counter() - t0

    print()
    print("=" * 60)
    print("FOGGY VDD GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total images: {total}")
    print(f"Total time  : {elapsed:.1f} s ({elapsed/max(total,1)*1000:.0f} ms/img)")
    print(f"Output root : {output_root.resolve()}")


if __name__ == "__main__":
    main()