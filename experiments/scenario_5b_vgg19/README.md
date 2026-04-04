# VGG19 Training Script for Gram Bacteria Classification

## 📋 Overview

This directory contains the complete training script for VGG19 model to classify Gram-positive and Gram-negative bacteria using transfer learning.

**Scenario:** 5b - Transfer Learning with VGG19  
**Model:** VGG19 pre-trained on ImageNet  
**Task:** Binary classification (Gram Negative vs Gram Positive)

---

## 🎯 Features Implemented

### ✅ Model Architecture
- **Backbone:** VGG19 (16 conv layers) pre-trained on ImageNet
  - **3 more conv layers** than VGG16 for deeper feature extraction
  - Better at capturing complex patterns
- **Custom Classifier Head:**
  - FC Layer 1: 25088 → 4096 + BatchNorm + ReLU + Dropout(0.5)
  - FC Layer 2: 4096 → 512 + BatchNorm + ReLU + Dropout(0.3)
  - Output Layer: 512 → 2

### ✅ Training Features
- **Transfer Learning:** ImageNet pre-trained weights
- **Class Imbalance Handling:** Weighted CrossEntropyLoss
  - Automatic class weight calculation based on training data distribution
  - G- weight: ~0.56, G+ weight: ~1.20
- **Early Stopping:** Patience = 10 epochs
- **Model Checkpointing:** Save best model based on validation accuracy
- **Learning Rate Scheduler:** ReduceLROnPlateau (patience=5, factor=0.5)
- **Mixed Precision Training:** FP16 for memory optimization (4GB VRAM)
- **Batch Normalization:** In classifier head for training stability

### ✅ Data Augmentation
- Random Rotation (±30°)
- Random Horizontal Flip (p=0.5)
- Random Vertical Flip (p=0.5)
- Color Jitter (brightness, contrast, saturation)
- Random Resized Crop (scale=0.8-1.0)

### ✅ Evaluation & Visualization
- **Metrics:** Accuracy, Precision, Recall, F1-Score, Cohen's Kappa, AUC-ROC
- **Visualizations:**
  - Training curves (loss & accuracy)
  - Confusion matrix
  - ROC curve
  - Learning rate schedule

---

## 🚀 How to Run

### Prerequisites
```bash
# Ensure you're in the project root directory
cd D:\kerja\dataset

# Verify GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Run Training
```bash
# Navigate to experiment directory
cd experiments/scenario_5b_vgg19

# Run training script
python train_vgg19.py
```

### Expected Runtime
- **GPU (RTX 3050 4GB):** 3-4 hours (with early stopping)
- **CPU:** 4-5 days (not recommended)

**Note:** VGG19 trains ~20% slower than VGG16 due to additional layers.

---

## 📊 Hyperparameters

```python
BATCH_SIZE = 8              # Optimized for 4GB VRAM
EPOCHS = 80                 # Max (early stopping active)
INITIAL_LR = 5e-5           # Conservative for fine-tuning
WEIGHT_DECAY = 1e-4         # L2 regularization
MOMENTUM = 0.9              # SGD momentum
OPTIMIZER = SGD             # Best for VGG architectures
SCHEDULER = ReduceLROnPlateau
EARLY_STOPPING_PATIENCE = 10
LR_SCHEDULER_PATIENCE = 5
MIXED_PRECISION = True      # FP16 training
```

---

## 📁 Output Files

After training completes, the following files will be generated:

| File | Size | Description |
|------|------|-------------|
| `best_model.pth` | ~548 MB | Model checkpoint (best validation accuracy) |
| `training_log.csv` | ~10 KB | Per-epoch training history |
| `metrics_summary.json` | ~2 KB | Final metrics & configuration |
| `classification_report.txt` | ~1 KB | Per-class precision/recall/F1 |
| `confusion_matrix.png` | ~100 KB | Visual confusion matrix |
| `roc_curve.png` | ~80 KB | ROC curve with AUC score |
| `training_curves.png` | ~150 KB | Loss & accuracy plots |

---

## 📈 Expected Performance

Based on project documentation and VGG19 characteristics:

| Metric | Expected Range |
|--------|----------------|
| **Validation Accuracy** | 83-88% |
| **Precision** | 82-87% |
| **Recall** | 83-88% |
| **F1-Score** | 82-87% |
| **AUC-ROC** | 0.89-0.93 |

**Note:** VGG19 expected to perform slightly better than VGG16 (82-87%) due to deeper architecture, but still lower than ResNet50 (93.86%).

---

## 🔍 Model Details

### Architecture Summary
```
VGG19 Backbone (Deeper than VGG16):
├── Conv Block 1: 2 × Conv(64) + MaxPool
├── Conv Block 2: 2 × Conv(128) + MaxPool
├── Conv Block 3: 4 × Conv(256) + MaxPool  ← +1 conv vs VGG16
├── Conv Block 4: 4 × Conv(512) + MaxPool  ← +1 conv vs VGG16
└── Conv Block 5: 4 × Conv(512) + MaxPool  ← +1 conv vs VGG16
    └── Output: 7×7×512 = 25,088 features

Custom Classifier:
├── FC1: 25088 → 4096 (BatchNorm + ReLU + Dropout 0.5)
├── FC2: 4096 → 512 (BatchNorm + ReLU + Dropout 0.3)
└── Output: 512 → 2 (Logits)
```

### Parameter Count
- **Total Parameters:** ~144,000,000 (~5M more than VGG16)
- **Trainable Parameters:** ~144,000,000
- **Model Size:** ~548 MB

---

## 🔬 VGG19 vs VGG16 Comparison

| Aspect | VGG16 | VGG19 |
|--------|-------|-------|
| **Conv Layers** | 13 | 16 (+3) |
| **Parameters** | 139M | 144M (+5M) |
| **Model Size** | 528 MB | 548 MB |
| **Training Speed** | Faster | Slower (~20%) |
| **Feature Depth** | Good | Better |
| **Expected Accuracy** | 82-87% | 83-88% |

**Trade-off:** VGG19 provides slightly better accuracy at the cost of longer training time.

---

## 🛠️ Troubleshooting

### Out of Memory (OOM) Error
```bash
# Reduce batch size in train_vgg19.py
BATCH_SIZE = 4  # Instead of 8
```

### Slow Training
```bash
# Verify GPU is being used
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Check CUDA compatibility
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

### Import Errors
```bash
# Install missing dependencies
pip install -r ../../requirements_api.txt
```

---

## 📝 Script Structure

```python
# 1. IMPORTS & CONFIGURATION (Lines 1-120)
# 2. MODEL DEFINITION (Lines 121-185)
# 3. DATA TRANSFORMS & LOADERS (Lines 186-265)
# 4. CLASS IMBALANCE HANDLING (Lines 266-295)
# 5. MODEL, OPTIMIZER, SCHEDULER (Lines 296-345)
# 6. EARLY STOPPING CLASS (Lines 346-385)
# 7. TRAINING FUNCTIONS (Lines 386-485)
# 8. MAIN TRAINING LOOP (Lines 486-585)
# 9. EVALUATION FUNCTIONS (Lines 586-655)
# 10. VISUALIZATION FUNCTIONS (Lines 656-755)
# 11. MAIN EXECUTION (Lines 756-873)
```

---

## 🔗 Related Experiments

| Experiment | Model | Status | Accuracy |
|------------|-------|--------|----------|
| Scenario 1 | Simple CNN | ✅ Complete | 87.49% |
| Scenario 4a | ResNet50 | ✅ Complete | 93.86% |
| Scenario 4b | ResNet101 | ✅ Complete | ~92% |
| Scenario 5a | VGG16 | 🚧 Ready to run | TBD |
| **Scenario 5b** | **VGG19** | 🚧 **Ready to run** | TBD |

---

## 📊 Dataset Information

```
Training Set: 4,478 images
├── Gram Negative: 2,620 images (58.5%)
└── Gram Positive: 1,858 images (41.5%)

Validation Set: 1,774 images
├── Gram Negative: 1,376 images (77.6%)
└── Gram Positive: 398 images (22.4%)

Image Size: 224×224×3 (RGB)
Normalization: ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

---

## 💡 When to Use VGG19 over VGG16

**Use VGG19 if:**
- ✅ You have sufficient training time (3-4 hours vs 2-3 hours)
- ✅ You want slightly better accuracy (~1% improvement)
- ✅ Dataset has complex patterns requiring deeper features
- ✅ GPU memory is not a constraint

**Use VGG16 if:**
- ✅ Training time is limited
- ✅ Model size is a concern
- ✅ Quick baseline comparison needed
- ✅ Dataset patterns are relatively simple

---

## 📞 Support

For issues or questions:
1. Check `../../doc/SKENARIO_UJICOBA_ARSITEKTUR.md` for experiment details
2. Review `../../doc/API_GUIDE.md` for API integration
3. Contact: TA KITA Project Team

---

## 📜 License

Part of TA KITA Project - Gram Bacteria Classification System  
© 2026 TA KITA Team

---

**Last Updated:** April 4, 2026  
**Script Version:** 1.0.0  
**PyTorch Version:** 2.5.1+cu121
