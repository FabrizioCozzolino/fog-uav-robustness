"""
Environment sanity check.
Verifies that PyTorch is installed correctly and reports available devices.
Run this FIRST to make sure everything works.

Usage:
    python scripts/check_env.py
"""
import sys
import platform


def main():
    print("=" * 60)
    print("FOG-UAV-ROBUSTNESS :: Environment Check")
    print("=" * 60)

    # Python
    print(f"\n[Python]")
    print(f"  Version : {sys.version.split()[0]}")
    print(f"  Platform: {platform.system()} {platform.release()} ({platform.machine()})")

    # PyTorch
    try:
        import torch
        print(f"\n[PyTorch]")
        print(f"  Version       : {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device   : {torch.cuda.get_device_name(0)}")
            print(f"  CUDA VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except ImportError:
        print("\n[PyTorch] NOT installed. Run: pip install torch torchvision "
              "--index-url https://download.pytorch.org/whl/cpu")
        return

    # Intel XPU (Intel Arc via IPEX) - optional
    try:
        import intel_extension_for_pytorch as ipex  # noqa
        xpu_ok = torch.xpu.is_available() if hasattr(torch, "xpu") else False
        print(f"\n[Intel Extension for PyTorch]")
        print(f"  IPEX version  : {ipex.__version__}")
        print(f"  XPU available : {xpu_ok}")
        if xpu_ok:
            print(f"  XPU device    : {torch.xpu.get_device_name(0)}")
    except ImportError:
        print("\n[Intel Extension for PyTorch] not installed (optional; only if you want to try Intel Arc training)")

    # Core libraries
    print("\n[Core libraries]")
    for name in ["torchvision", "segmentation_models_pytorch", "albumentations",
                 "torchmetrics", "cv2", "numpy", "matplotlib"]:
        try:
            mod = __import__(name)
            version = getattr(mod, "__version__", "?")
            print(f"  OK   {name:32s} {version}")
        except ImportError:
            print(f"  MISS {name}")

    # Recommended device
    print("\n[Recommended device for this run]")
    if torch.cuda.is_available():
        print("  --> CUDA GPU detected. Use device='cuda'.")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        print("  --> Intel XPU detected. Use device='xpu' (via IPEX).")
    else:
        print("  --> Only CPU available. Use device='cpu'.")
        print("      This is fine for local development/debugging on small subsets.")
        print("      For real training, use Google Colab or Kaggle (free GPU).")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
