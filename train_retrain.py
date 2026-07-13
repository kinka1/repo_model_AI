"""
Unified Retrain Script for Gram Bacteria Classification
=======================================================
Called by the backend via subprocess.Popen.

Supports: resnet50, resnet101, efficientnet_b0, efficientnet_b3, densenet121
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from app.model_architectures import (
    GramResNet50Classifier,
    GramResNet101Classifier,
    GramEfficientNetB0Classifier,
    GramEfficientNetB3Classifier,
    GramDenseNet121Classifier,
)

ARCH_MAP = {
    "resnet50": GramResNet50Classifier,
    "resnet101": GramResNet101Classifier,
    "efficientnet_b0": GramEfficientNetB0Classifier,
    "efficientnet_b3": GramEfficientNetB3Classifier,
    "densenet121": GramDenseNet121Classifier,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = (224, 224)


def parse_args():
    parser = argparse.ArgumentParser(description="Retrain Gram Classification Model")
    parser.add_argument("--data", required=True, help="Path to training data directory")
    parser.add_argument("--epochs-head", type=int, default=10, help="Epochs for head training")
    parser.add_argument("--epochs-ft", type=int, default=30, help="Epochs for fine-tuning")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--save-model", required=True, help="Path to save trained model")
    parser.add_argument("--save-metrics", required=True, help="Path to save metrics JSON")
    parser.add_argument("--name", default="retrain", help="Run name")
    parser.add_argument("--arch", default="resnet50", help="Model architecture key")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Learning rate for head training")
    parser.add_argument("--lr-ft", type=float, default=5e-5, help="Learning rate for fine-tuning")
    return parser.parse_args()


def load_checkpoint(model, arch_key, device):
    """Load existing model weights if available."""
    model_paths = {
        "resnet50": ROOT_DIR / "models" / "best_model_resnet50.pth",
        "resnet101": ROOT_DIR / "models" / "best_model_resnet101.pth",
        "efficientnet_b0": ROOT_DIR / "models" / "best_model_efficientnet_b0.pth",
        "efficientnet_b3": ROOT_DIR / "models" / "best_model_efficientnet_b3.pth",
        "densenet121": ROOT_DIR / "models" / "best_model_densenet121.pth",
    }
    ckpt_path = model_paths.get(arch_key)
    if not ckpt_path or not ckpt_path.exists():
        print(f"[WARN] No existing checkpoint for {arch_key}, starting from scratch")
        return

    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=True)
        print(f"[OK] Loaded existing weights: {ckpt_path.name}")
    except Exception as e:
        print(f"[WARN] Could not load checkpoint: {e}, starting from scratch")


def create_dataloaders(data_dir, batch_size, val_split=0.15):
    """Create train/val data loaders from the retrain dataset."""
    data_path = Path(data_dir)
    if not (data_path / "train").exists():
        # Try flat structure with subdirectories per class
        dataset = datasets.ImageFolder(data_path, transform=_get_train_transform())
    else:
        train_transform = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_transform = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = datasets.ImageFolder(data_path / "train", transform=train_transform)
        val_dataset = datasets.ImageFolder(data_path / "val", transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return train_loader, val_loader, len(train_dataset), len(val_dataset)


def _get_train_transform():
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _get_val_transform():
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def freeze_backbone(model, arch_key):
    """Freeze all layers except the final classifier head."""
    if arch_key.startswith("resnet"):
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.backbone.fc.parameters():
            param.requires_grad = True
    elif arch_key.startswith("efficientnet"):
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.backbone._fc.parameters():
            param.requires_grad = True
    elif arch_key.startswith("densenet"):
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.backbone.classifier.parameters():
            param.requires_grad = True


def unfreeze_last_layers(model, arch_key):
    """Unfreeze the last few layers for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def save_checkpoint(model, metrics, epoch, save_path):
    """Save model state dict and metrics."""
    save_path = Path(save_path)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "arch": args.arch,
    }
    torch.save(checkpoint, save_path)
    print(f"[OK] Model saved: {save_path.name}")


def save_metrics(metrics, metrics_path):
    """Save metrics to JSON."""
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Metrics saved: {Path(metrics_path).name}")


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print(f"Retrain: {args.name}")
    print(f"Architecture: {args.arch}")
    print(f"Device: {DEVICE}")
    print(f"Data: {args.data}")
    print(f"Head epochs: {args.epochs_head}, FT epochs: {args.epochs_ft}")
    print("=" * 60)

    # Create data loaders
    train_loader, val_loader, n_train, n_val = create_dataloaders(args.data, args.batch_size)
    print(f"Train samples: {n_train}, Val samples: {n_val}")

    # Initialize model
    ModelClass = ARCH_MAP.get(args.arch)
    if not ModelClass:
        print(f"[ERROR] Unknown architecture: {args.arch}")
        sys.exit(1)

    model = ModelClass().to(DEVICE)
    load_checkpoint(model, args.arch, DEVICE)

    criterion = nn.CrossEntropyLoss()

    # === PHASE 1: Head Training ===
    if args.epochs_head > 0:
        print("\n--- Phase 1: Head Training ---")
        freeze_backbone(model, args.arch)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_head)

        for epoch in range(args.epochs_head):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = validate(model, val_loader, criterion)
            print(f"  Head Epoch {epoch+1}/{args.epochs_head}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
            print(f"  PROGRESS:epoch:{epoch + 1}")

    # === PHASE 2: Fine-tuning ===
    if args.epochs_ft > 0:
        print("\n--- Phase 2: Fine-tuning ---")
        unfreeze_last_layers(model, args.arch)
        optimizer = optim.Adam(model.parameters(), lr=args.lr_ft)

        best_val_acc = 0.0
        patience = 8
        patience_counter = 0
        total_ft_epochs = args.epochs_head + args.epochs_ft

        for epoch in range(args.epochs_ft):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = validate(model, val_loader, criterion)

            overall_epoch = args.epochs_head + epoch + 1
            print(f"  FT Epoch {epoch+1}/{args.epochs_ft}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
            print(f"  PROGRESS:epoch:{overall_epoch}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                save_checkpoint(model, {"val_acc": val_acc, "epoch": overall_epoch}, overall_epoch, args.save_model)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  [Early stopping] No improvement for {patience} epochs")
                    break

    # === Final evaluation ===
    train_loss, train_acc = validate(model, train_loader, criterion)
    val_loss, val_acc = validate(model, val_loader, criterion)
    print(f"\nFinal: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    metrics = {
        "accuracy": val_acc,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "arch": args.arch,
        "epochs_head": args.epochs_head,
        "epochs_ft": args.epochs_ft,
        "batch_size": args.batch_size,
        "n_train": n_train,
        "n_val": n_val,
    }
    save_metrics(metrics, args.save_metrics)
    save_checkpoint(model, metrics, args.epochs_head + args.epochs_ft, args.save_model)

    print("\n[DONE] Retrain complete")