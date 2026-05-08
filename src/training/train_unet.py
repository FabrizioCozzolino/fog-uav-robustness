"""Train a U-Net for semantic segmentation on VDD.

Phase 5 update: now supports multiple training data roots via --data-roots,
which makes it possible to train on a mix of clean and foggy VDD variants.
The validation root is still a single one (--data-root, used for selecting the
best checkpoint).

Examples:
    # Phase 1 baseline (clean only)
    python src/training/train_unet.py \\
        --data-root data/raw/VDD/VDD \\
        --run-name unet_resnet34_clean_v2_weighted \\
        --image-size 768 --epochs 30 --batch-size 4 \\
        --class-weights inverse_sqrt

    # Phase 5 mixed training: clean + foggy_medium_768 + foggy_dense_768
    python src/training/train_unet.py \\
        --data-root data/raw/VDD/VDD \\
        --data-roots data/raw/VDD/VDD \\
                     data/processed/VDD_foggy_medium_768 \\
                     data/processed/VDD_foggy_dense_768 \\
        --run-name unet_resnet34_mixed_v3 \\
        --image-size 768 --epochs 30 --batch-size 4 \\
        --class-weights inverse_sqrt
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.datasets.vdd import VDD_CLASSES, VDDDataset
from src.evaluation.metrics import SegmentationMetrics
from src.models.unet import build_unet, count_parameters, human_readable
from src.utils.transforms import get_eval_transform, get_train_transform


def parse_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument(
        "--data-root",
        default="data/raw/VDD/VDD",
        help="Root used for VALIDATION (single split). The training data is "
             "either this same root (default) or the list passed to --data-roots.",
    )
    p.add_argument(
        "--data-roots",
        nargs="+",
        default=None,
        help="One or more roots used for TRAINING. If given, the training "
             "dataset is the concatenation of the train splits of each root. "
             "If omitted, training uses --data-root only.",
    )
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--subset", type=int, default=0,
                   help="If >0, use only first N samples from each root (smoke test)")

    # Model
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--num-classes", type=int, default=7)
    p.add_argument("--no-pretrained", action="store_true")

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--val-batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--class-weights",
        choices=["none", "inverse", "inverse_sqrt"],
        default="none",
        help="Per-class weighting in CrossEntropyLoss.",
    )

    # Device / seed
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)

    # I/O
    p.add_argument("--output-dir", default="outputs/runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--log-every", type=int, default=10)

    return p.parse_args()


def pick_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_train_dataset(data_roots, image_size, subset, transform):
    """Build a (possibly concatenated) training dataset from one or more VDD-shaped roots."""
    individual = []
    for root in data_roots:
        ds = VDDDataset(root, "train", transform=transform)
        if subset > 0:
            k = min(subset, len(ds))
            ds = Subset(ds, list(range(k)))
        print(f"[INFO]   train root: {root}  ->  {len(ds)} samples")
        individual.append(ds)
    if len(individual) == 1:
        return individual[0]
    return ConcatDataset(individual)


def train_one_epoch(model, loader, optimizer, criterion, device, writer, epoch, args):
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
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if (i + 1) % args.log_every == 0:
            step = (epoch - 1) * len(loader) + i
            writer.add_scalar("train/loss_batch", loss.item(), step)

    avg_loss = total_loss / max(n_batches, 1)
    writer.add_scalar("train/loss_epoch", avg_loss, epoch)
    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
    return avg_loss


@torch.no_grad()
def validate(model, loader, criterion, metrics, device, writer, epoch):
    model.eval()
    metrics.reset()
    total_loss = 0.0
    n_batches = 0
    for images, masks in tqdm(loader, desc=f"[val]   epoch {epoch}", leave=False, ncols=100):
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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device(args.device)
    print(f"[INFO] Device: {device}")

    if args.run_name is None:
        args.run_name = f"unet_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Run dir: {run_dir}")
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- Datasets ---
    train_transform = get_train_transform(args.image_size)
    val_transform = get_eval_transform(args.image_size)

    train_roots = args.data_roots if args.data_roots else [args.data_root]
    print(f"[INFO] Training data roots: {train_roots}")
    train_ds = build_train_dataset(train_roots, args.image_size, args.subset, train_transform)
    val_ds = VDDDataset(args.data_root, "val", transform=val_transform)
    if args.subset > 0:
        k_val = min(max(args.subset // 2, 4), len(val_ds))
        val_ds = Subset(val_ds, list(range(k_val)))
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

    # --- Model ---
    model = build_unet(
        num_classes=args.num_classes,
        encoder_name=args.encoder,
        encoder_weights=None if args.no_pretrained else "imagenet",
    ).to(device)
    trainable, total = count_parameters(model)
    print(f"[INFO] Model: {args.encoder} U-Net | params: {human_readable(total)}")

    # --- Loss with optional class weights ---
    if args.class_weights != "none":
        print(f"[INFO] Computing per-class weights ({args.class_weights}) "
              f"from train masks of FIRST root only ({train_roots[0]}) ...")
        # Use the first root's distribution (typically the clean one).
        # For mixed training, the clean dist is a reasonable baseline; we don't
        # need to scan the foggy variants too because masks are identical.
        base_ds = VDDDataset(train_roots[0], "train", transform=None)
        dist = base_ds.get_class_distribution()
        freqs = np.array(
            [dist.get(VDD_CLASSES[i], 1e-9) for i in range(args.num_classes)],
            dtype=np.float64,
        )
        freqs = np.clip(freqs, 1e-9, None)
        if args.class_weights == "inverse":
            weights = 1.0 / freqs
        else:  # inverse_sqrt
            weights = 1.0 / np.sqrt(freqs)
        weights = weights / weights.mean()
        print(f"[INFO] Class frequencies (%): "
              + ", ".join(f"{VDD_CLASSES[i]}={freqs[i]*100:.3f}" for i in range(args.num_classes)))
        print(f"[INFO] Class weights         : "
              + ", ".join(f"{VDD_CLASSES[i]}={weights[i]:.2f}" for i in range(args.num_classes)))
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    # --- Optimizer / scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Metrics + TB ---
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
        print(f"[epoch {epoch:3d}/{args.epochs}]  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  mIoU={miou:.4f}  F1={val_results['F1']:.4f}  "
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
        })

        if miou > best_miou:
            best_miou = miou
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mIoU": miou,
                "args": vars(args),
            }, run_dir / "best.pth")
            print(f"             -> new best mIoU, saved best.pth")

        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    # --- Final saves ---
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "mIoU": miou,
        "args": vars(args),
    }, run_dir / "last.pth")
    writer.close()

    print("\n" + "=" * 60)
    print("TRAINING DONE")
    print("=" * 60)
    print(f"Best val mIoU : {best_miou:.4f}  (epoch {best_epoch})")
    print(f"Run dir       : {run_dir}")
    print(f"\nTo inspect training curves in TensorBoard, run:")
    print(f"  tensorboard --logdir {args.output_dir}")


if __name__ == "__main__":
    main()
