"""
BonsAI - GPU Check Utility
==========================
Verifies AMD GPU and ROCm are properly detected by PyTorch.

Usage:
    python scripts/check_gpu.py
"""

import sys


def check_gpu():
    print("\n🔍 BonsAI — GPU Environment Check\n" + "=" * 40)

    # Check Python
    print(f"Python:  {sys.version.split()[0]}")

    # Check PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not installed. Run: pip install -r requirements.txt")
        return

    # Check CUDA/ROCm
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"\n✅ GPU(s) detected: {count}")
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            vram  = props.total_memory / 1024**3
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"          VRAM: {vram:.1f} GB")
            print(f"          Compute: {props.major}.{props.minor}")

        # Quick tensor test
        print("\n⚡ Running GPU tensor test...")
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.randn(1000, 1000, device="cuda")
        z = torch.matmul(x, y)
        print(f"   Matrix multiply (1000x1000): ✅  Result shape: {z.shape}")

        # Estimate training speed
        print("\n📊 Estimated training time (50 epochs, 2000 images, batch=32):")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if vram_gb >= 16:
            print("   ~20–35 minutes (excellent!)")
        elif vram_gb >= 8:
            print("   ~45–70 minutes (good)")
        else:
            print("   ~90–120 minutes (reduce batch size if OOM)")

    else:
        print("\n⚠️  No GPU detected by PyTorch.")
        print("   If you have an AMD GPU, make sure ROCm is installed:")
        print("   https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html")
        print("\n   You can still train on CPU, but it will be much slower.")

    print("\n" + "=" * 40)


if __name__ == "__main__":
    check_gpu()
