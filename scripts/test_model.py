"""Smoke test for the U-Net model.

Runs four checks:
  1. Build the model (downloads pretrained weights on first run).
  2. Count parameters and estimate size.
  3. Forward pass on a single random batch and verify output shape.
  4. Backward pass (loss + gradient step) to ensure the graph is correct.

Usage:
    python scripts/test_model.py
    python scripts/test_model.py --encoder resnet50 --image-size 512 --batch-size 2
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from src.models.unet import build_unet, count_parameters, human_readable


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-classes", type=int, default=7)
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def pick_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"[INFO] Using device: {device}")

    # --- 1. Build model ---
    print(f"\n[1/4] Building U-Net (encoder={args.encoder}, classes={args.num_classes}) ...")
    t0 = time.perf_counter()
    model = build_unet(
        num_classes=args.num_classes,
        encoder_name=args.encoder,
        encoder_weights="imagenet",
    ).to(device)
    print(f"      Built in {time.perf_counter() - t0:.1f}s")
    print(f"      (First run may download pretrained weights from the internet.)")

    # --- 2. Parameters ---
    trainable, total = count_parameters(model)
    # Approximate memory for fp32 parameters: 4 bytes each
    size_mb = total * 4 / 1e6
    print(f"\n[2/4] Parameters:")
    print(f"      Trainable : {human_readable(trainable)} ({trainable:,})")
    print(f"      Total     : {human_readable(total)} ({total:,})")
    print(f"      Size (fp32): ~{size_mb:.1f} MB")

    # --- 3. Forward pass ---
    print(f"\n[3/4] Forward pass: batch={args.batch_size}, image={args.image_size}x{args.image_size}")
    x = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        logits = model(x)
        fwd_time = time.perf_counter() - t0
    print(f"      Input shape  : {tuple(x.shape)}")
    print(f"      Output shape : {tuple(logits.shape)}   <-- (B, C, H, W) with C = num_classes")
    expected = (args.batch_size, args.num_classes, args.image_size, args.image_size)
    assert tuple(logits.shape) == expected, f"Unexpected output shape!"
    print(f"      Forward time : {fwd_time * 1000:.0f} ms  ({fwd_time / args.batch_size * 1000:.0f} ms/img)")

    # --- 4. Backward pass ---
    print(f"\n[4/4] Backward pass (loss + gradient step):")
    model.train()
    # Fake target: random class IDs for each pixel
    targets = torch.randint(0, args.num_classes,
                            (args.batch_size, args.image_size, args.image_size),
                            device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    t0 = time.perf_counter()
    logits = model(x)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    bwd_time = time.perf_counter() - t0

    # Verify gradients actually flowed
    grads_present = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    n_params = sum(1 for _ in model.parameters())

    print(f"      Loss value        : {loss.item():.4f}")
    print(f"      Fwd+Bwd time      : {bwd_time * 1000:.0f} ms")
    print(f"      Params with grad  : {grads_present}/{n_params}")
    assert grads_present > 0, "No gradients flowed! Something is wrong with the graph."

    # --- Summary ---
    print("\n" + "=" * 60)
    print(" ALL TESTS PASSED ")
    print("=" * 60)
    if device.type == "cpu":
        print("Note: on CPU, forward+backward is slow. Expected: 5-30 s per batch.")
        print("For real training, use Colab/Kaggle GPU (~100 ms per batch on T4).")


if __name__ == "__main__":
    main()
