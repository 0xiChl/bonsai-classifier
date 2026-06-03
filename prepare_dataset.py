"""
BonsAI - Dataset Preparation Script
=====================================
Organizes raw bonsai images into train/val/test splits.

Usage:
    python scripts/prepare_dataset.py --input /path/to/raw_images --output data/
"""

import argparse
import shutil
import random
from pathlib import Path


TRAIN_RATIO = 0.75
VAL_RATIO   = 0.15
TEST_RATIO  = 0.10

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def prepare_dataset(input_dir: str, output_dir: str, seed: int = 42):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return

    # Discover class folders
    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    if not class_dirs:
        print("❌ No class subdirectories found.")
        print("   Expected structure: input_dir/species_name/image.jpg")
        return

    print(f"📂 Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")

    random.seed(seed)
    total_images = 0

    for class_dir in sorted(class_dirs):
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        if not images:
            print(f"   ⚠️  No images found in: {class_dir.name}")
            continue

        random.shuffle(images)
        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        splits = {
            "train": images[:n_train],
            "val":   images[n_train:n_train + n_val],
            "test":  images[n_train + n_val:],
        }

        for split_name, split_images in splits.items():
            dest = output_path / split_name / class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for img in split_images:
                shutil.copy2(img, dest / img.name)

        print(f"   ✅ {class_dir.name:20s} → train:{len(splits['train'])} | val:{len(splits['val'])} | test:{len(splits['test'])}")
        total_images += n

    print(f"\n✅ Done! {total_images} images split into train/val/test.")
    print(f"   Output: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Prepare bonsai dataset splits")
    parser.add_argument("--input",  required=True, help="Directory with class subfolders")
    parser.add_argument("--output", default="data", help="Output directory (default: data/)")
    parser.add_argument("--seed",   type=int, default=42, help="Random seed")
    args = parser.parse_args()

    prepare_dataset(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
