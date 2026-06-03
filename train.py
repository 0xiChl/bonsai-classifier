"""
BonsAI - Training Script
========================
Trains a multi-task CNN for bonsai species identification
and health assessment using AMD GPU (ROCm) or CPU fallback.

Usage:
    python src/train.py --epochs 50 --batch-size 32
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import json
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

SPECIES_CLASSES = [
    "juniper", "ficus", "maple", "black_pine",
    "azalea", "fukien_tea", "snow_rose", "chinese_elm",
    "jade_bonsai", "pemphis"
]

HEALTH_CLASSES = [
    "healthy", "overwatered", "underwatered", "pest_detected"
]

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ── Data Transforms ────────────────────────────────────────────────────────────

def get_transforms(mode="train"):
    """Return image transforms for train or val/test mode."""
    if mode == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])


# ── Model ──────────────────────────────────────────────────────────────────────

class BonsAIClassifier(nn.Module):
    """
    Multi-task ResNet-50 classifier.
    Single forward pass → species label + health label.
    """

    def __init__(self, num_species=10, num_health=4):
        super().__init__()

        # Pretrained backbone (ImageNet weights)
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # remove final FC

        feature_dim = backbone.fc.in_features  # 2048

        # Shared intermediate layer
        self.shared_head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
        )

        # Task-specific output branches
        self.species_head = nn.Linear(512, num_species)
        self.health_head  = nn.Linear(512, num_health)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.shared_head(x)
        return self.species_head(x), self.health_head(x)


# ── Training Loop ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    species_correct = health_correct = total = 0

    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images = images.to(device)
        # For demo: use same label for both heads (replace with dual-label dataset)
        species_labels = labels.to(device)
        health_labels  = (labels % len(HEALTH_CLASSES)).to(device)

        optimizer.zero_grad()
        sp_logits, hl_logits = model(images)

        loss = criterion(sp_logits, species_labels) + criterion(hl_logits, health_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        species_correct += (sp_logits.argmax(1) == species_labels).sum().item()
        health_correct  += (hl_logits.argmax(1) == health_labels).sum().item()
        total += images.size(0)

    return {
        "loss":           total_loss / total,
        "species_acc":    species_correct / total,
        "health_acc":     health_correct / total,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    species_correct = health_correct = total = 0

    for images, labels in tqdm(loader, desc="  Validating", leave=False):
        images = images.to(device)
        species_labels = labels.to(device)
        health_labels  = (labels % len(HEALTH_CLASSES)).to(device)

        sp_logits, hl_logits = model(images)
        loss = criterion(sp_logits, species_labels) + criterion(hl_logits, health_labels)

        total_loss += loss.item() * images.size(0)
        species_correct += (sp_logits.argmax(1) == species_labels).sum().item()
        health_correct  += (hl_logits.argmax(1) == health_labels).sum().item()
        total += images.size(0)

    return {
        "loss":           total_loss / total,
        "species_acc":    species_correct / total,
        "health_acc":     health_correct / total,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train BonsAI Classifier")
    p.add_argument("--data-dir",   default="data",   help="Root data directory")
    p.add_argument("--save-dir",   default="models", help="Where to save checkpoints")
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--workers",    type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Device Detection ──────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU detected — training on CPU (will be slow)")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_dir = Path(args.data_dir) / "train"
    val_dir   = Path(args.data_dir) / "val"

    if not train_dir.exists():
        print(f"\n❌ Data directory not found: {train_dir}")
        print("   Please prepare your dataset first:")
        print("   python scripts/prepare_dataset.py --input /your/images --output data/")
        return

    train_ds = datasets.ImageFolder(train_dir, transform=get_transforms("train"))
    val_ds   = datasets.ImageFolder(val_dir,   transform=get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=True)

    print(f"\n📂 Dataset loaded:")
    print(f"   Train: {len(train_ds):,} images")
    print(f"   Val:   {len(val_ds):,} images")
    print(f"   Species classes: {train_ds.classes}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BonsAIClassifier(
        num_species=len(SPECIES_CLASSES),
        num_health=len(HEALTH_CLASSES)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model: BonsAI-ResNet50 ({total_params:,} parameters)")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Training ──────────────────────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    history = []

    print(f"\n🚀 Starting training for {args.epochs} epochs ...\n")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch:3d}/{args.epochs}")

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        avg_val_acc = (val_metrics["species_acc"] + val_metrics["health_acc"]) / 2

        print(f"  Train → loss: {train_metrics['loss']:.4f} | "
              f"species_acc: {train_metrics['species_acc']:.3f} | "
              f"health_acc: {train_metrics['health_acc']:.3f}")
        print(f"  Val   → loss: {val_metrics['loss']:.4f} | "
              f"species_acc: {val_metrics['species_acc']:.3f} | "
              f"health_acc: {val_metrics['health_acc']:.3f}")

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            ckpt_path = save_dir / "bonsai_best.pt"
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_metrics,
            }, ckpt_path)
            print(f"  💾 Saved best model → {ckpt_path}")

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print()

    # Save training history
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"✅ Training complete! Best val accuracy: {best_val_acc:.3f}")
    print(f"   Model saved to: {save_dir / 'bonsai_best.pt'}")


if __name__ == "__main__":
    main()
