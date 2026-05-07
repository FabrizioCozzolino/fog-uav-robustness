"""Train Pix2Pix on Foggy Cityscapes (clean -> foggy translation).

Logs everything to TensorBoard:
    - scalars: losses (D, G_adv, G_l1, G_total) and D's prediction means
    - images: every N epochs, save a grid (clean | generated | target) on a
      fixed batch from the val set, so you can SEE how generation improves.

Saves under outputs/runs/<run_name>/:
    config.json   parameters used for this run
    G_best.pth    Generator weights at lowest val L1 loss
    G_last.pth    Generator weights at the end
    D_last.pth    Discriminator at the end (rarely useful, but kept)
    history.json  per-epoch metrics
    tb/           TensorBoard event files
    samples/      occasional sample grids (.png)

Usage:
    # Smoke test on CPU (5 images, 2 epochs):
    python src/training/train_pix2pix.py \\
        --fog-level medium --epochs 2 --subset 5 --batch-size 2

    # Real training on Colab T4:
    python src/training/train_pix2pix.py \\
        --fog-level medium --epochs 50 --batch-size 8 --num-workers 2
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
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from src.datasets.foggy_cityscapes import (
    FoggyCityscapesPairedDataset,
    denormalize_tanh,
)
from src.models.gan.pix2pix import build_pix2pix, count_params


# ------------------------------------------------------------------- CLI


def parse_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--data-root", default="data/raw/foggy_cityscapes/Foggy_Cityscapes")
    p.add_argument("--fog-level", default="medium", choices=["medium", "dense"])
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--subset", type=int, default=0,
                   help="If >0, use only first N samples of train AND val (smoke test)")

    # Model
    p.add_argument("--encoder", default="resnet34",
                   help="Generator encoder backbone (smp-supported)")
    p.add_argument("--ndf", type=int, default=64,
                   help="Number of filters in the first discriminator layer")

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Learning rate (paper default: 2e-4)")
    p.add_argument("--beta1", type=float, default=0.5,
                   help="Adam beta1 (paper default for GANs: 0.5)")
    p.add_argument("--lambda-l1", type=float, default=100.0,
                   help="Weight of L1 in G's loss (paper default: 100)")
    p.add_argument("--num-workers", type=int, default=0)

    # Device / seed
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)

    # I/O
    p.add_argument("--output-dir", default="outputs/runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--sample-every", type=int, default=5,
                   help="Save image samples every N epochs.")
    p.add_argument("--log-every", type=int, default=20,
                   help="Log per-batch losses to TB every N batches.")

    return p.parse_args()


def pick_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------- one training epoch


def train_one_epoch(
    G, D, loader, opt_G, opt_D, bce, l1, device, writer, epoch, args
):
    G.train()
    D.train()

    sums = {
        "loss_D": 0.0, "loss_D_real": 0.0, "loss_D_fake": 0.0,
        "loss_G_adv": 0.0, "loss_G_l1": 0.0, "loss_G_total": 0.0,
        "D_real_mean": 0.0, "D_fake_mean": 0.0,
    }
    n_batches = 0

    pbar = tqdm(loader, desc=f"[train] epoch {epoch}", leave=False, ncols=110)
    for i, (clean, real_foggy) in enumerate(pbar):
        clean = clean.to(device, non_blocking=True)
        real_foggy = real_foggy.to(device, non_blocking=True)

        # ============ Update D ============
        # Generate fake (no grad through G here -> .detach())
        with torch.no_grad():
            fake_foggy = G(clean)
        d_real = D(clean, real_foggy)
        d_fake = D(clean, fake_foggy)

        real_label = torch.ones_like(d_real)
        fake_label = torch.zeros_like(d_fake)

        loss_D_real = bce(d_real, real_label)
        loss_D_fake = bce(d_fake, fake_label)
        loss_D = 0.5 * (loss_D_real + loss_D_fake)

        opt_D.zero_grad(set_to_none=True)
        loss_D.backward()
        opt_D.step()

        # ============ Update G ============
        fake_foggy = G(clean)  # recompute with grad
        d_fake_for_G = D(clean, fake_foggy)
        loss_G_adv = bce(d_fake_for_G, torch.ones_like(d_fake_for_G))
        loss_G_l1 = l1(fake_foggy, real_foggy)
        loss_G = loss_G_adv + args.lambda_l1 * loss_G_l1

        opt_G.zero_grad(set_to_none=True)
        loss_G.backward()
        opt_G.step()

        # --- Track metrics ---
        with torch.no_grad():
            sums["loss_D"]       += loss_D.item()
            sums["loss_D_real"]  += loss_D_real.item()
            sums["loss_D_fake"]  += loss_D_fake.item()
            sums["loss_G_adv"]   += loss_G_adv.item()
            sums["loss_G_l1"]    += loss_G_l1.item()
            sums["loss_G_total"] += loss_G.item()
            sums["D_real_mean"]  += torch.sigmoid(d_real).mean().item()
            sums["D_fake_mean"]  += torch.sigmoid(d_fake_for_G).mean().item()
        n_batches += 1

        if (i + 1) % args.log_every == 0:
            step = (epoch - 1) * len(loader) + i
            writer.add_scalar("train_batch/loss_D",     loss_D.item(),     step)
            writer.add_scalar("train_batch/loss_G_adv", loss_G_adv.item(), step)
            writer.add_scalar("train_batch/loss_G_l1",  loss_G_l1.item(),  step)

        pbar.set_postfix(
            D=f"{loss_D.item():.3f}",
            G_adv=f"{loss_G_adv.item():.3f}",
            G_l1=f"{loss_G_l1.item():.3f}",
        )

    avgs = {k: v / max(n_batches, 1) for k, v in sums.items()}
    for k, v in avgs.items():
        writer.add_scalar(f"train/{k}", v, epoch)
    return avgs


# ------------------------------------------------------------ validation


@torch.no_grad()
def validate(G, loader, l1, device, writer, epoch):
    """Validation = average L1 reconstruction error of G on the val set.

    No discriminator involved. We track this as our 'best model' criterion
    because L1 on val is the most reliable signal for paired translation
    (low L1 => generated foggy is close to ground-truth foggy).
    """
    G.eval()
    total_l1 = 0.0
    n_batches = 0
    for clean, real_foggy in loader:
        clean = clean.to(device, non_blocking=True)
        real_foggy = real_foggy.to(device, non_blocking=True)
        fake_foggy = G(clean)
        total_l1 += l1(fake_foggy, real_foggy).item()
        n_batches += 1
    avg_l1 = total_l1 / max(n_batches, 1)
    writer.add_scalar("val/L1", avg_l1, epoch)
    return avg_l1


@torch.no_grad()
def save_sample_grid(G, fixed_batch, device, writer, epoch, save_dir):
    """Run G on a fixed val batch and write a grid (clean | fake | real)
    to TensorBoard and to outputs/runs/<run>/samples/epoch_NNN.png."""
    G.eval()
    clean, real_foggy = fixed_batch
    clean = clean.to(device)
    real_foggy = real_foggy.to(device)
    fake_foggy = G(clean)

    # Bring back to [0, 1] for display
    clean_d = denormalize_tanh(clean)
    fake_d  = denormalize_tanh(fake_foggy)
    real_d  = denormalize_tanh(real_foggy)

    # Stack rows: clean / fake / real, each row has B images
    n = clean_d.shape[0]
    row1 = make_grid(clean_d, nrow=n, padding=2)
    row2 = make_grid(fake_d,  nrow=n, padding=2)
    row3 = make_grid(real_d,  nrow=n, padding=2)
    grid = torch.cat([row1, row2, row3], dim=1)  # along height

    writer.add_image("samples/clean__fake__real", grid, epoch)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_image(grid, save_dir / f"epoch_{epoch:03d}.png")


# ------------------------------------------------------------------ main


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device(args.device)
    print(f"[INFO] Device: {device}")

    if args.run_name is None:
        args.run_name = (
            f"pix2pix_{args.fog_level}_{args.encoder}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Run dir: {run_dir}")

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- Datasets + loaders ---
    train_ds = FoggyCityscapesPairedDataset(
        root=args.data_root, fog_level=args.fog_level,
        image_size=args.image_size, split="train",
    )
    val_ds = FoggyCityscapesPairedDataset(
        root=args.data_root, fog_level=args.fog_level,
        image_size=args.image_size, split="val",
    )

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

    # --- Models, losses, optimizers ---
    G, D = build_pix2pix(
        encoder_name=args.encoder, encoder_weights="imagenet", ndf=args.ndf,
    )
    G = G.to(device)
    D = D.to(device)
    print(f"[INFO] Generator     params: {count_params(G)/1e6:.2f}M")
    print(f"[INFO] Discriminator params: {count_params(D)/1e6:.2f}M")

    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    samples_dir = run_dir / "samples"

    # Fix one batch from val for monitoring (always show the same images)
    fixed_iter = iter(val_loader)
    fixed_batch = next(fixed_iter)
    # Limit to first 4 images for grid clarity
    fixed_batch = (fixed_batch[0][:4], fixed_batch[1][:4])

    # --- Training loop ---
    best_val_l1 = float("inf")
    best_epoch = -1
    history = []

    print(f"\n[INFO] Starting Pix2Pix training for {args.epochs} epochs "
          f"(fog_level={args.fog_level}, lambda_l1={args.lambda_l1})\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_metrics = train_one_epoch(
            G, D, train_loader, opt_G, opt_D, bce, l1, device, writer, epoch, args,
        )
        val_l1 = validate(G, val_loader, l1, device, writer, epoch)
        dt = time.perf_counter() - t0

        print(f"[epoch {epoch:3d}/{args.epochs}]  "
              f"loss_D={train_metrics['loss_D']:.3f}  "
              f"G_adv={train_metrics['loss_G_adv']:.3f}  "
              f"G_l1={train_metrics['loss_G_l1']:.3f}  "
              f"D(real)={train_metrics['D_real_mean']:.2f} "
              f"D(fake)={train_metrics['D_fake_mean']:.2f}  "
              f"val_L1={val_l1:.4f}  ({dt:.1f}s)")

        # Periodic visual samples
        if epoch == 1 or epoch % args.sample_every == 0 or epoch == args.epochs:
            save_sample_grid(G, fixed_batch, device, writer, epoch, samples_dir)

        # Track best
        history.append({
            "epoch": epoch,
            **train_metrics,
            "val_L1": val_l1,
            "time_s": dt,
        })
        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "G_state_dict": G.state_dict(),
                "val_L1": val_l1,
                "args": vars(args),
            }, run_dir / "G_best.pth")
            print(f"             -> new best val_L1, saved G_best.pth")

        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    # --- Final saves ---
    torch.save({
        "epoch": args.epochs,
        "G_state_dict": G.state_dict(),
        "val_L1": val_l1,
        "args": vars(args),
    }, run_dir / "G_last.pth")
    torch.save({
        "epoch": args.epochs,
        "D_state_dict": D.state_dict(),
        "args": vars(args),
    }, run_dir / "D_last.pth")
    writer.close()

    print("\n" + "=" * 60)
    print("TRAINING DONE")
    print("=" * 60)
    print(f"Best val L1     : {best_val_l1:.4f}  (epoch {best_epoch})")
    print(f"Run dir         : {run_dir}")
    print(f"\nTo inspect: tensorboard --logdir {args.output_dir}")
    print(f"Sample grids saved to: {samples_dir}")


if __name__ == "__main__":
    main()
