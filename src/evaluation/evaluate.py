"""Evaluate a trained checkpoint on a chosen split (val/test) of VDD.

Saves results as a JSON next to the checkpoint, plus prints them.

Usage:
    python src/evaluation/evaluate.py \\
        --checkpoint outputs/runs/unet_resnet34_clean_v2_weighted/best.pth \\
        --data-root data/raw/VDD/VDD \\
        --split test \\
        --image-size 768 \\
        --batch-size 4
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Allow `from src.*` imports when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.vdd import VDDDataset, VDD_CLASSES
from src.evaluation.metrics import SegmentationMetrics
from src.models.unet import build_unet
from src.utils.transforms import get_eval_transform


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pth checkpoint (e.g. outputs/runs/.../best.pth)")
    p.add_argument("--data-root", default="data/raw/VDD/VDD")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--image-size", type=int, default=768,
                   help="Must match the size used at training time.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--num-classes", type=int, default=7)
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--output", default=None,
                   help="Where to save JSON results. Default: <ckpt_dir>/<split>_results.json")
    return p.parse_args()


def pick_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"[INFO] Device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"[INFO] Checkpoint epoch={ckpt.get('epoch', '?')}, "
          f"val_mIoU={ckpt.get('mIoU', '?')}")

    # --- Model ---
    model = build_unet(
        num_classes=args.num_classes,
        encoder_name=args.encoder,
        encoder_weights=None,  # we'll load from checkpoint
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # --- Dataset ---
    transform = get_eval_transform(args.image_size)
    ds = VDDDataset(args.data_root, args.split, transform=transform)
    print(f"[INFO] Evaluating on '{args.split}' split: {len(ds)} samples, "
          f"image_size={args.image_size}")

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # --- Metrics ---
    class_names = [VDD_CLASSES[i] for i in range(args.num_classes)]
    metrics = SegmentationMetrics(args.num_classes, device, class_names)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0

    # --- Evaluation loop ---
    t0 = time.perf_counter()
    with torch.no_grad():
        for images, masks in tqdm(loader, desc=f"eval[{args.split}]", ncols=100):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            total_loss += criterion(logits, masks).item()
            n_batches += 1
            metrics.update(logits, masks)
    elapsed = time.perf_counter() - t0

    avg_loss = total_loss / max(n_batches, 1)
    results = metrics.compute()
    results["loss"] = avg_loss
    results["split"] = args.split
    results["num_samples"] = len(ds)
    results["image_size"] = args.image_size
    results["checkpoint"] = str(ckpt_path)
    results["eval_time_s"] = elapsed

    # --- Print ---
    print()
    print("=" * 60)
    print(f"EVALUATION RESULTS  -  split={args.split}")
    print("=" * 60)
    print(f"Samples evaluated : {len(ds)}")
    print(f"Eval time         : {elapsed:.1f} s ({elapsed/len(ds)*1000:.0f} ms/img)")
    print(f"Loss              : {avg_loss:.4f}")
    print(f"mIoU              : {results['mIoU']:.4f}")
    print(f"F1 (macro)        : {results['F1']:.4f}")
    print(f"Accuracy          : {results['accuracy']:.4f}")
    print(f"\nPer-class IoU:")
    for cls, iou in results["per_class_iou"].items():
        bar = "#" * int(iou * 40)
        print(f"  {cls:14s} {iou:.4f}  {bar}")

    # --- Save ---
    if args.output is None:
        args.output = ckpt_path.parent / f"{args.split}_results.json"
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved results to: {out_path}")


if __name__ == "__main__":
    main()
