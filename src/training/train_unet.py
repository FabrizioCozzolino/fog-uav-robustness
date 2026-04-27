"""Train a U-Net on VDD for semantic segmentation.

Logs everything to TensorBoard and saves:
    outputs/runs/<run_name>/
        ├── config.json       parameters used for this run
        ├── best.pth          model weights with best val mIoU
        ├── last.pth          model weights at the end of training
        ├── history.json      per-epoch metrics (for plots in the report)
        └── tb/               TensorBoard event files

Usage:
    # Real training (needs GPU in practice):
    python src/training/train_unet.py --epochs 30 --batch-size 8

    # Smoke test on CPU (2 epochs, 20 images):
    python src/training/train_unet.py --epochs 2 --subset 20 --batch-size 2

    # Launch TensorBoard (separate terminal):
    tensorboard --logdir outputs/runs
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Allow src.* imports when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.datasets.vdd import VDD_CLASSES, VDDDataset
from src.evaluation.metrics import SegmentationMetrics
from src.models.unet import build_unet, count_parameters, human_readable
from src.utils.transforms import get_eval_transform, get_train_transform


# ------------------------------------------------------------------ CLI


def parse_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--data-root", default="data/raw/VDD/VDD")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--subset", type=int, default=0,
                   help="If >0, use only first N samples of train AND val (smoke test)")

    # Model
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--num-classes", type=int, default=7)
    p.add_argument("--no-pretrained", action="store_true",
                   help="Train from scratch instead of ImageNet weights")

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--val-batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers. Keep 0 on Windows.")
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Device / seed
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)

    # I/O
    p.add_argument("--output-dir", default="outputs/runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--log-every", type=int, default=10,
                   help="Log batch-level loss to TB every N batches")

    return p.parse_args()


def pick_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------ train / val loops


def train_one_epoch(model, loader, optimizer, criterion, device, writer,
                    epoch: int, args):
    model.train()
    total_loss = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc=f"[train] epoch {epoch}", leave=False, ncols=100)
    for i, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (i + 1) % args.log_every == 0:
            step = (epoch - 1) * len(loader) + i
            writer.add_scalar("train/loss_batch", loss.item(), step)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    writer.add_scalar("train/loss_epoch", avg_loss, epoch)
    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
    return avg_loss


@torch.no_grad()
def validate(model, loader, criterion, metrics: SegmentationMetrics,
             device, writer, epoch: int):
    model.eval()
    metrics.reset()
    total_loss = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc=f"[val]   epoch {epoch}", leave=False, ncols=100)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item()
        n_batches += 1
        metrics.update(logits, masks)

    avg_loss = total_loss / max(n_batches, 1)
    results = metrics.compute()

    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/mIoU", results["mIoU"], epoch)
    writer.add_scalar("val/F1", results["F1"], epoch)
    writer.add_scalar("val/accuracy", results["accuracy"], epoch)
    for cls, iou in results["per_class_iou"].items():
        writer.add_scalar(f"val/iou_{cls}", iou, epoch)

    return avg_loss, results


# ------------------------------------------------------------------ main


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device(args.device)
    print(f"[INFO] Device: {device}")

    # Run directory
    if args.run_name is None:
        args.run_name = f"unet_{args.encoder}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Run dir: {run_dir}")

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- Datasets + loaders ---
    train_ds = VDDDataset(args.data_root, "train",
                          transform=get_train_transform(args.image_size))
    val_ds = VDDDataset(args.data_root, "val",
                        transform=get_eval_transform(args.image_size))

    if args.subset > 0:
        k_train = min(args.subset, len(train_ds))
        k_val = min(max(args.subset // 2, 4), len(val_ds))
        train_ds = Subset(train_ds, list(range(k_train)))
        val_ds = Subset(val_ds, list(range(k_val)))
        print(f"[INFO] SUBSET MODE: train={len(train_ds)}, val={len(val_ds)}")
    else:
        print(f"[INFO] Dataset: train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    print(f"[INFO] Batches: train={len(train_loader)}, val={len(val_loader)}")

    # --- Model, loss, optimizer, scheduler ---
    model = build_unet(
        num_classes=args.num_classes,
        encoder_name=args.encoder,
        encoder_weights=None if args.no_pretrained else "imagenet",
    ).to(device)
    _, total_params = count_parameters(model)
    print(f"[INFO] Model: {args.encoder} U-Net | params: {human_readable(total_params)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    class_names = [VDD_CLASSES[i] for i in range(args.num_classes)]
    metrics = SegmentationMetrics(args.num_classes, device, class_names)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    # --- Training loop ---
    best_miou = -1.0
    best_epoch = -1
    history = []

    print(f"\n[INFO] Starting training for {args.epochs} epochs\n")
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     device, writer, epoch, args)
        val_loss, val_results = validate(model, val_loader, criterion, metrics,
                                         device, writer, epoch)
        scheduler.step()

        dt = time.perf_counter() - t0
        miou = val_results["mIoU"]
        print(f"[epoch {epoch:3d}/{args.epochs}] "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"mIoU={miou:.4f}  F1={val_results['F1']:.4f}  "
              f"acc={val_results['accuracy']:.4f}  ({dt:.1f}s)")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mIoU": miou,
            "val_F1": val_results["F1"],
            "val_accuracy": val_results["accuracy"],
            "val_per_class_iou": val_results["per_class_iou"],
            "time_s": dt,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if miou > best_miou:
            best_miou = miou
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mIoU": best_miou,
                "args": vars(args),
            }, run_dir / "best.pth")
            print(f"              -> new best mIoU, saved best.pth")

        # Save history every epoch (so if we Ctrl+C we don't lose it)
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    # --- Final save ---
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "mIoU": val_results["mIoU"],
        "args": vars(args),
    }, run_dir / "last.pth")
    writer.close()

    print("\n" + "=" * 60)
    print(f"TRAINING DONE")
    print("=" * 60)
    print(f"Best val mIoU : {best_miou:.4f}  (epoch {best_epoch})")
    print(f"Run dir       : {run_dir}")
    print(f"\nTo inspect training curves in TensorBoard, run:")
    print(f"  tensorboard --logdir {args.output_dir}")
    print("Then open the URL it prints (usually http://localhost:6006) in your browser.")


if __name__ == "__main__":
    main()
