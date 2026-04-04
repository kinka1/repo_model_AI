"""
VGG16 Training Script for Gram Bacteria Classification
=======================================================

This script implements transfer learning using VGG16 pre-trained on ImageNet
for binary classification of Gram-positive and Gram-negative bacteria.

Features:
- Transfer learning with VGG16 backbone
- Custom classifier head with Batch Normalization
- Class imbalance handling with weighted CrossEntropyLoss
- Early stopping (patience=10)
- Model checkpointing (save best model)
- SGD optimizer with ReduceLROnPlateau scheduler
- Mixed precision training (FP16) for memory optimization
- Comprehensive evaluation metrics
- Visualization (confusion matrix, ROC curve, training curves)

Author: TA KITA Project
Date: April 2026
"""

import os
import json
import time
from pathlib import Path
from typing import Tuple, Dict, List

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import models, transforms, datasets

# Data science
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Random seed for reproducibility
SEED = 42

# Training hyperparameters
BATCH_SIZE = 8              # Small batch size for 4GB VRAM
EPOCHS = 80                 # Maximum epochs (early stopping will trigger earlier)
INITIAL_LR = 5e-5           # Conservative learning rate for fine-tuning
WEIGHT_DECAY = 1e-4         # L2 regularization
MOMENTUM = 0.9              # SGD momentum
NUM_WORKERS = 4             # DataLoader workers
INPUT_SIZE = (224, 224)     # Standard ImageNet input size
NUM_CLASSES = 2             # Binary classification (G- and G+)

# Early stopping & scheduler
EARLY_STOPPING_PATIENCE = 10
LR_SCHEDULER_PATIENCE = 5
LR_FACTOR = 0.5
MIN_LR = 1e-7

# Dropout
DROPOUT_P = 0.5

# Mixed precision training
USE_MIXED_PRECISION = True

# Paths
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / 'data' / 'processed'
EXP_DIR = ROOT_DIR / 'experiments' / 'scenario_5a_vgg16'
EXP_DIR.mkdir(parents=True, exist_ok=True)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print("=" * 70)
print("🔬 VGG16 Training for Gram Bacteria Classification")
print("=" * 70)
print(f"📁 Experiment directory: {EXP_DIR}")
print(f"🔧 Device: {DEVICE}")
print(f"🎲 Random seed: {SEED}")
print(f"📊 Batch size: {BATCH_SIZE}")
print(f"🔄 Max epochs: {EPOCHS}")
print(f"📈 Initial LR: {INITIAL_LR}")
print(f"⚡ Mixed precision: {USE_MIXED_PRECISION}")
print("=" * 70)


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class GramVGG16Classifier(nn.Module):
    """
    VGG16-based classifier for Gram bacteria classification.
    
    Architecture:
    - Backbone: VGG16 pre-trained on ImageNet (13 conv layers + pooling)
    - Custom classifier head with BatchNorm and Dropout
    - Output: 2 classes (Gram Negative, Gram Positive)
    
    Features:
    - Transfer learning from ImageNet
    - Batch Normalization for training stability
    - Dropout for regularization
    - Medium-depth classifier (3 FC layers: 25088 -> 4096 -> 512 -> 2)
    """
    
    def __init__(self, num_classes: int = 2, dropout_p: float = 0.5):
        """
        Initialize VGG16 classifier.
        
        Args:
            num_classes (int): Number of output classes (default: 2)
            dropout_p (float): Dropout probability (default: 0.5)
        """
        super(GramVGG16Classifier, self).__init__()
        
        # Load pre-trained VGG16
        print("📥 Loading VGG16 pre-trained weights from ImageNet...")
        self.backbone = models.vgg16(weights='IMAGENET1K_V1')
        
        # Replace classifier head
        # VGG16 features output: 7×7×512 = 25,088 features
        in_features = 25088
        
        self.backbone.classifier = nn.Sequential(
            # First FC layer: 25088 -> 4096
            nn.Linear(in_features, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            
            # Second FC layer: 4096 -> 512
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            
            # Output layer: 512 -> num_classes
            nn.Linear(512, num_classes)
        )
        
        print("✅ VGG16 model initialized with custom classifier head")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, 3, 224, 224)
        
        Returns:
            torch.Tensor: Output logits of shape (batch, num_classes)
        """
        return self.backbone(x)


# ============================================================================
# DATA TRANSFORMS & LOADERS
# ============================================================================

print("\n📂 Setting up data transforms...")

# Training transforms (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Validation transforms (without augmentation)
val_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Load datasets
print("📂 Loading datasets...")
train_dataset = datasets.ImageFolder(DATA_DIR / 'train', transform=train_transform)
val_dataset = datasets.ImageFolder(DATA_DIR / 'val', transform=val_transform)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True if torch.cuda.is_available() else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True if torch.cuda.is_available() else False
)

print(f"✅ Training samples: {len(train_dataset)}")
print(f"✅ Validation samples: {len(val_dataset)}")
print(f"✅ Class mapping: {train_dataset.class_to_idx}")


# ============================================================================
# CLASS IMBALANCE HANDLING
# ============================================================================

print("\n⚖️  Computing class weights for imbalance handling...")

# Extract labels from dataset
train_labels = [label for _, label in train_dataset.samples]

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

print(f"📊 Class distribution in training set:")
print(f"   - Class 0 (Gram Negative): {train_labels.count(0)} samples")
print(f"   - Class 1 (Gram Positive): {train_labels.count(1)} samples")
print(f"⚖️  Class weights:")
print(f"   - Gram Negative weight: {class_weights[0]:.4f}")
print(f"   - Gram Positive weight: {class_weights[1]:.4f}")


# ============================================================================
# MODEL, OPTIMIZER, SCHEDULER, LOSS
# ============================================================================

print("\n🧠 Initializing model, optimizer, and scheduler...")

# Initialize model
model = GramVGG16Classifier(num_classes=NUM_CLASSES, dropout_p=DROPOUT_P).to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"📊 Model parameters:")
print(f"   - Total: {total_params:,}")
print(f"   - Trainable: {trainable_params:,}")

# Loss function with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Optimizer - SGD works better for VGG
optimizer = optim.SGD(
    model.parameters(),
    lr=INITIAL_LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',          # Maximize validation accuracy
    factor=LR_FACTOR,
    patience=LR_SCHEDULER_PATIENCE,
    min_lr=MIN_LR,
    verbose=True
)

# Mixed precision scaler
scaler = GradScaler() if USE_MIXED_PRECISION else None

print(f"✅ Optimizer: SGD (lr={INITIAL_LR}, momentum={MOMENTUM})")
print(f"✅ Scheduler: ReduceLROnPlateau (patience={LR_SCHEDULER_PATIENCE})")
print(f"✅ Loss function: CrossEntropyLoss (weighted)")


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """
    Early stopping to stop training when validation accuracy stops improving.
    
    Args:
        patience (int): How many epochs to wait after last improvement
        min_delta (float): Minimum change to qualify as improvement
        verbose (bool): Print messages when stopping
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_acc: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_acc (float): Current validation accuracy
            epoch (int): Current epoch number
        
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = val_acc
            self.best_epoch = epoch
            return False
        
        if val_acc > self.best_score + self.min_delta:
            self.best_score = val_acc
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"   ⏳ EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n🛑 Early stopping triggered! Best epoch: {self.best_epoch}")
                return True
        
        return False


# Initialize early stopping
early_stopping = EarlyStopping(
    patience=EARLY_STOPPING_PATIENCE,
    min_delta=0.001,
    verbose=True
)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler = None
) -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Neural network model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use (cuda/cpu)
        scaler: GradScaler for mixed precision (optional)
    
    Returns:
        Tuple[float, float]: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100.0 * correct / total:.2f}%"
        })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Validate model.
    
    Args:
        model: Neural network model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to use (cuda/cpu)
    
    Returns:
        Tuple[float, float, float]: (loss, accuracy, f1_score)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(all_labels)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return val_loss, val_acc, val_f1


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_model():
    """Main training loop with early stopping and checkpointing."""
    
    print("\n" + "=" * 70)
    print("🚀 Starting Training")
    print("=" * 70)
    
    training_log = []
    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        print(f"\n📅 Epoch {epoch+1}/{EPOCHS}")
        print("-" * 70)
        
        # Training phase
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler
        )
        
        # Validation phase
        val_loss, val_acc, val_f1 = validate(
            model, val_loader, criterion, DEVICE
        )
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Epoch time
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1: {val_f1:.4f}")
        print(f"   LR: {current_lr:.2e} | Time: {epoch_time:.2f}s")
        
        # Log metrics
        training_log.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'lr': current_lr,
            'epoch_time': epoch_time
        })
        
        # Save checkpoint if best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_loss': val_loss,
                'class_weights': class_weights.tolist()
            }
            
            checkpoint_path = EXP_DIR / 'best_model.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"   ✅ Saved new best model (val_acc: {val_acc:.4f})")
        
        # Early stopping check
        if early_stopping(val_acc, epoch + 1):
            print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
            break
        
        # Save training log periodically
        if (epoch + 1) % 5 == 0:
            pd.DataFrame(training_log).to_csv(EXP_DIR / 'training_log.csv', index=False)
    
    # Training complete
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print(f"   Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"   Total Time: {total_time/3600:.2f} hours")
    print("=" * 70)
    
    # Save final training log
    pd.DataFrame(training_log).to_csv(EXP_DIR / 'training_log.csv', index=False)
    
    return training_log, best_epoch, best_val_acc


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict:
    """
    Comprehensive evaluation of model on test/validation set.
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to use
    
    Returns:
        Dict containing all metrics and predictions
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\n🔍 Evaluating model on validation set...")
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluation"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])  # Prob for class 1 (G+)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'kappa': cohen_kappa_score(all_labels, all_preds),
        'auc_roc': roc_auc_score(all_labels, all_probs)
    }
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    class_names = ['Gram Negative (G-)', 'Gram Positive (G+)']
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    return {
        'metrics': metrics,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_curves(training_log: List[Dict], save_path: Path):
    """Plot training and validation curves."""
    df = pd.DataFrame(training_log)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('VGG16 - Training Progress', fontsize=16, fontweight='bold')
    
    # Loss curves
    axes[0, 0].plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
    axes[0, 0].plot(df['epoch'], df['val_loss'], 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Loss Curves', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(df['epoch'], df['train_acc'], 'b-', label='Train Acc', linewidth=2, marker='o', markersize=4)
    axes[0, 1].plot(df['epoch'], df['val_acc'], 'r-', label='Val Acc', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy', fontsize=11)
    axes[0, 1].set_title('Accuracy Curves', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(df['epoch'], df['val_f1'], 'g-', linewidth=2, marker='D', markersize=4)
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('F1 Score', fontsize=11)
    axes[1, 0].set_title('Validation F1 Score', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 1].plot(df['epoch'], df['lr'], 'm-', linewidth=2, marker='^', markersize=4)
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=11)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved training curves to {save_path.name}")


def plot_confusion_matrix(cm: np.ndarray, save_path: Path):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['G-', 'G+'],
        yticklabels=['G-', 'G+'],
        cbar_kws={'label': 'Count'},
        square=True,
        linewidths=1,
        linecolor='gray'
    )
    
    plt.title('VGG16 - Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved confusion matrix to {save_path.name}")


def plot_roc_curve(labels: List, probs: List, save_path: Path):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'VGG16 (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier (AUC = 0.5000)')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('VGG16 - ROC Curve', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved ROC curve to {save_path.name}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Train model
    training_log, best_epoch, best_val_acc = train_model()
    
    # Load best model for evaluation
    print("\n📂 Loading best model for final evaluation...")
    checkpoint = torch.load(EXP_DIR / 'best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on validation set
    eval_results = evaluate_model(model, val_loader, DEVICE)
    
    # Print metrics
    print("\n" + "=" * 70)
    print("📊 Final Validation Metrics")
    print("=" * 70)
    for metric, value in eval_results['metrics'].items():
        print(f"   {metric.capitalize():.<25} {value:.4f}")
    print("=" * 70)
    
    # Save classification report
    with open(EXP_DIR / 'classification_report.txt', 'w', encoding='utf-8') as f:
        f.write("VGG16 - Gram Bacteria Classification\n")
        f.write("=" * 70 + "\n\n")
        f.write("Final Validation Metrics:\n")
        f.write("-" * 70 + "\n")
        for metric, value in eval_results['metrics'].items():
            f.write(f"{metric.capitalize():.<25} {value:.4f}\n")
        f.write("\n" + "=" * 70 + "\n\n")
        f.write("Classification Report:\n")
        f.write("-" * 70 + "\n")
        f.write(eval_results['classification_report'])
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("Confusion Matrix:\n")
        f.write("-" * 70 + "\n")
        f.write(str(eval_results['confusion_matrix']))
    print(f"✅ Saved classification report")
    
    # Save metrics summary
    metrics_summary = {
        'scenario': 'Scenario 5a: Transfer Learning (VGG16)',
        'model': 'VGG16',
        'augmentation': True,
        'transfer_learning': True,
        'fine_tuning': 'Full model',
        'batch_size': BATCH_SIZE,
        'initial_lr': INITIAL_LR,
        'optimizer': 'SGD',
        'momentum': MOMENTUM,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'final_metrics': eval_results['metrics'],
        'confusion_matrix': eval_results['confusion_matrix'].tolist(),
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'class_weights': class_weights.tolist(),
        'training_config': {
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'initial_lr': INITIAL_LR,
            'weight_decay': WEIGHT_DECAY,
            'momentum': MOMENTUM,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE,
            'lr_scheduler_patience': LR_SCHEDULER_PATIENCE,
            'input_size': INPUT_SIZE,
            'seed': SEED,
            'mixed_precision': USE_MIXED_PRECISION
        }
    }
    
    with open(EXP_DIR / 'metrics_summary.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"✅ Saved metrics summary")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    plot_training_curves(training_log, EXP_DIR / 'training_curves.png')
    plot_confusion_matrix(eval_results['confusion_matrix'], EXP_DIR / 'confusion_matrix.png')
    plot_roc_curve(eval_results['labels'], eval_results['probabilities'], EXP_DIR / 'roc_curve.png')
    
    print("\n" + "=" * 70)
    print("🎉 Experiment Complete!")
    print(f"📁 All results saved to: {EXP_DIR}")
    print("=" * 70)
    
    # Print file summary
    print("\n📄 Generated Files:")
    files = [
        'best_model.pth',
        'training_log.csv',
        'metrics_summary.json',
        'classification_report.txt',
        'confusion_matrix.png',
        'roc_curve.png',
        'training_curves.png'
    ]
    for file in files:
        file_path = EXP_DIR / file
        if file_path.exists():
            size = file_path.stat().st_size / (1024**2)  # MB
            print(f"   ✅ {file:<30} ({size:.2f} MB)")
    
    print("\n🚀 VGG16 training script finished successfully!")
    print("\n" + "=" * 70)
