"""Apply a trained Pix2Pix Generator to ALL VDD images (train + val + test).

Produces a "foggy" version of VDD that mirrors the original folder structure:

    output_root/
        train/
            src/  (foggy .JPG, generated)
            gt/   (.png masks, copied unchanged from VDD)
        val/   (same structure)
        test/  (same structure)

The masks are NOT modified by the GAN: fog doesn't change what is road, vehicle,
or vegetation -- it only changes how visible they are. This means the
segmentation labels of VDD apply unchanged to the foggy version.

The Generator is run at `--apply-size` (256 or 768 typically) and the output
is then resized to `--save-size` for storage. Internally the original VDD
JPG is read at full resolution, downsampled to apply-size, fed to G, and the
G output is upsampled to save-size before saving as JPG (or PNG if you prefer).

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

# Allow `from src.*` imports when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.models.gan.pix2pix import Pix2PixGenerator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--generator", required=True,
                   help="Path to G_best.pth or G_last.pth from a Pix2Pix run")
    p.add_argument("--vdd-root", default="data/raw/VDD/VDD",
                   help="Original VDD root with train/val/test subfolders")
    p.add_argument("--output-root", required=True,
                   help="Where to write VDD_foggy (will create folders if needed)")
    p.add_argument("--apply-size", type=int, default=768,
                   help="Resolution at which the Generator is applied. Common: 256 (matches training) or 768 (matches U-Net eval).")
    p.add_argument("--save-size", type=int, default=768,
                   help="Resolution at which foggy images are saved. Default 768.")
    p.add_argument("--encoder", default="resnet34",
                   help="Encoder backbone used in training (must match)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=4,
                   help="Number of images processed in parallel by G")
    p.add_argument("--mask-format", default="png", choices=["png"],
                   help="Output format for masks (kept identical to VDD's)")
    p.add_argument("--image-format", default="jpg", choices=["jpg", "png"],
                   help="Output format for foggy images (jpg saves disk space)")
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
    """Load a trained Pix2PixGenerator from a checkpoint."""
    print(f"[INFO] Loading generator: {ckpt_path}")
    G = Pix2PixGenerator(
        encoder_name=encoder_name,
        encoder_weights=None,  # we'll load weights from checkpoint
        in_channels=3,
        out_channels=3,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "G_state_dict" not in ckpt:
        raise RuntimeError(f"Checkpoint {ckpt_path} doesn't have 'G_state_dict' key")
    G.load_state_dict(ckpt["G_state_dict"])
    G.eval()
    print(f"[INFO] Generator loaded: epoch={ckpt.get('epoch', '?')}, "
          f"val_L1={ckpt.get('val_L1', '?'):.4f}" if isinstance(ckpt.get('val_L1'), float)
          else f"[INFO] Generator loaded: epoch={ckpt.get('epoch', '?')}")
    return G


def preprocess_batch(images_bgr: list, apply_size: int, device: torch.device) -> torch.Tensor:
    """Stack a list of BGR uint8 images into a normalized batch tensor.

    Mirrors the dataset preprocessing during training:
      BGR -> RGB
      resize to apply_size x apply_size (bilinear)
      normalize to [-1, +1] (tanh range, mean=0.5, std=0.5)
      stack into (B, 3, H, W) float tensor.
    """
    batch = []
    for img_bgr in images_bgr:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (apply_size, apply_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0     # [0, 1]
        img = (img - 0.5) / 0.5                   # [-1, +1]
        img = torch.from_numpy(img).permute(2, 0, 1)  # CHW
        batch.append(img)
    return torch.stack(batch, dim=0).to(device, non_blocking=True)


def postprocess_batch(out_tensor: torch.Tensor, save_size: int) -> list:
    """Convert a (B, 3, H, W) tanh-normalized tensor back to a list of BGR uint8 images.

    De-normalizes from [-1, +1] to [0, 255], permutes to HWC, RGB->BGR for cv2 saving,
    and resizes to save_size x save_size if needed.
    """
    out = (out_tensor.detach().clamp(-1.0, 1.0) + 1.0) * 127.5  # [0, 255]
    out = out.cpu().numpy().astype(np.uint8)
    out_imgs = []
    for arr in out:                  # (3, H, W)
        arr = np.transpose(arr, (1, 2, 0))   # (H, W, 3) RGB
        if arr.shape[0] != save_size or arr.shape[1] != save_size:
            arr = cv2.resize(arr, (save_size, save_size), interpolation=cv2.INTER_LINEAR)
        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        out_imgs.append(arr_bgr)
    return out_imgs


@torch.no_grad()
def process_split(
    split: str,
    G: Pix2PixGenerator,
    vdd_root: Path,
    output_root: Path,
    apply_size: int,
    save_size: int,
    batch_size: int,
    device: torch.device,
    image_format: str,
    jpeg_quality: int,
) -> int:
    """Translate every image in the given split, copy masks. Return # images processed."""
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

    # JPEG encode params
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
        # 1. Read original BGR images at full resolution
        imgs_bgr = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths_chunk]
        if any(im is None for im in imgs_bgr):
            bad = [str(p) for p, im in zip(paths_chunk, imgs_bgr) if im is None]
            raise RuntimeError(f"Failed to read: {bad}")
        # 2. Preprocess (resize to apply_size, normalize, batch)
        x = preprocess_batch(imgs_bgr, apply_size, device)
        # 3. Run G
        y = G(x)  # (B, 3, apply_size, apply_size) in [-1, +1]
        # 4. Postprocess (denormalize, BGR, resize to save_size)
        out_bgr_list = postprocess_batch(y, save_size)
        # 5. Save foggy image, copy mask unchanged
        for src_path, foggy_bgr in zip(paths_chunk, out_bgr_list):
            stem = src_path.stem
            out_img = out_src_dir / f"{stem}{out_ext}"
            ok = cv2.imwrite(str(out_img), foggy_bgr, encode_params)
            if not ok:
                raise RuntimeError(f"Failed to write {out_img}")
            # Copy mask exactly: same name, same content, no resize (loader handles it)
            mask_src = gt_dir / f"{stem}.png"
            mask_dst = out_gt_dir / f"{stem}.png"
            if not mask_src.is_file():
                raise RuntimeError(f"Missing mask: {mask_src}")
            shutil.copyfile(mask_src, mask_dst)
            n_done += 1

    return n_done


def copy_metadata(vdd_root: Path, output_root: Path) -> None:
    """Copy the VDD metadata folder (train.txt etc.) so the new dataset is
    fully self-contained."""
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

    # --- Load Generator ---
    G = load_generator(Path(args.generator), args.encoder, device)

    # --- Copy metadata (train.txt, val.txt, test.txt) for completeness ---
    copy_metadata(vdd_root, output_root)

    # --- Process all splits ---
    t0 = time.perf_counter()
    total = 0
    for split in ["train", "val", "test"]:
        total += process_split(
            split=split,
            G=G,
            vdd_root=vdd_root,
            output_root=output_root,
            apply_size=args.apply_size,
            save_size=args.save_size,
            batch_size=args.batch_size,
            device=device,
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
        )
    elapsed = time.perf_counter() - t0

    print()
    print("=" * 60)
    print("FOGGY VDD GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total images: {total}")
    print(f"Total time  : {elapsed:.1f} s ({elapsed/max(total,1)*1000:.0f} ms/img)")
    print(f"Output root : {output_root.resolve()}")
    print()
    print("Next step: run evaluate.py with --data-root pointed at the new folder")
    print("to measure how the U-Net's mIoU drops on the foggy version.")


if __name__ == "__main__":
    main()