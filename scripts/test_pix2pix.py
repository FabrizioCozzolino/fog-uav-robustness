"""Smoke test for the Pix2Pix Generator and Discriminator.

Verifies:
  1. Both networks can be built (downloads pretrained encoder on first run).
  2. Forward shapes are correct on a fake batch.
  3. A single full training step (G + D update) runs without error and the
     gradients actually flow.

Usage:
    python scripts/test_pix2pix.py
    python scripts/test_pix2pix.py --image-size 256 --batch-size 2
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from src.models.gan.pix2pix import build_pix2pix, count_params


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lambda-l1", type=float, default=100.0,
                   help="Weight of L1 loss in the G objective (paper default: 100)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
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

    # --- 1. Build models ---
    print(f"\n[1/3] Building Pix2Pix models (encoder={args.encoder}) ...")
    t0 = time.perf_counter()
    G, D = build_pix2pix(encoder_name=args.encoder, encoder_weights="imagenet")
    G = G.to(device)
    D = D.to(device)
    print(f"      Built in {time.perf_counter() - t0:.1f}s")
    print(f"      Generator      params: {count_params(G):>12,}  ({count_params(G)/1e6:.2f}M)")
    print(f"      Discriminator  params: {count_params(D):>12,}  ({count_params(D)/1e6:.2f}M)")

    # --- 2. Forward pass on a fake batch ---
    print(f"\n[2/3] Forward pass: batch={args.batch_size}, image={args.image_size}x{args.image_size}")
    clean = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    real_foggy = torch.randn_like(clean)

    G.eval()
    D.eval()
    with torch.no_grad():
        fake_foggy = G(clean)
        d_real = D(clean, real_foggy)
        d_fake = D(clean, fake_foggy)
    print(f"      G(clean)            -> {tuple(fake_foggy.shape)}   range [{fake_foggy.min():.2f}, {fake_foggy.max():.2f}]")
    print(f"      D(clean, real_fog)  -> {tuple(d_real.shape)}")
    print(f"      D(clean, fake_fog)  -> {tuple(d_fake.shape)}")

    # G output must be (B, 3, H, W) in [-1, +1] (tanh)
    assert fake_foggy.shape == clean.shape, f"G output shape mismatch: {fake_foggy.shape}"
    assert fake_foggy.min() >= -1.0001 and fake_foggy.max() <= 1.0001, \
        f"G output not in [-1, +1]: [{fake_foggy.min()}, {fake_foggy.max()}]"
    # D output must be a (B, 1, ~30, ~30) score map for 256x256 inputs
    assert d_real.shape == d_fake.shape, "D output shapes inconsistent between real and fake"
    expected_h = args.image_size // 8 - 2  # PatchGAN with 256 -> 30
    print(f"      D output spatial size = {d_real.shape[2]}x{d_real.shape[3]}  (expected ~{expected_h})")

    # --- 3. One full training step ---
    print(f"\n[3/3] One full training step (G + D update):")
    G.train()
    D.train()
    bce = nn.BCEWithLogitsLoss()
    l1  = nn.L1Loss()
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # ----- Update D -----
    fake_foggy = G(clean).detach()  # detach so G doesn't get D's grad here
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

    # ----- Update G -----
    fake_foggy = G(clean)               # recompute with grad to G
    d_fake_for_G = D(clean, fake_foggy)
    loss_G_adv = bce(d_fake_for_G, real_label)        # G wants D to think fake is real
    loss_G_l1  = l1(fake_foggy, real_foggy)            # plus L1 to the actual target
    loss_G = loss_G_adv + args.lambda_l1 * loss_G_l1

    opt_G.zero_grad(set_to_none=True)
    loss_G.backward()
    opt_G.step()

    # Sanity check: gradients flowed through both networks
    g_grads = sum(1 for p in G.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    d_grads = sum(1 for p in D.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"      loss_D       = {loss_D.item():.4f}   (expect ~ln(2) = 0.69 at init)")
    print(f"      loss_G_adv   = {loss_G_adv.item():.4f}")
    print(f"      loss_G_l1    = {loss_G_l1.item():.4f}")
    print(f"      loss_G total = {loss_G.item():.4f}")
    print(f"      G params with grad : {g_grads}")
    print(f"      D params with grad : {d_grads}")
    assert g_grads > 0, "No gradients flowed through G!"
    assert d_grads > 0, "No gradients flowed through D!"

    print("\n" + "=" * 60)
    print(" ALL TESTS PASSED ")
    print("=" * 60)
    if device.type == "cpu":
        print("Note: on CPU, this single step takes a few seconds.")
        print("Real training will be much faster on Colab (~100ms/step on T4).")


if __name__ == "__main__":
    main()
