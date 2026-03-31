# 🧪 SKENARIO UJICOBA ARSITEKTUR CNN
## Klasifikasi Bakteri Gram - Comprehensive Testing Protocol

**Tanggal**: November 25, 2025  
**Proyek**: Sistem Klasifikasi Bakteri Gram Berbasis Deep Learning  
**Tujuan**: Membandingkan performa 7 arsitektur CNN untuk menentukan model optimal

---

## 📋 DAFTAR ISI

1. [Overview Eksperimen](#overview-eksperimen)
2. [Arsitektur yang Diuji](#arsitektur-yang-diuji)
3. [Metodologi Ujicoba](#metodologi-ujicoba)
4. [Skenario Per Eksperimen](#skenario-per-eksperimen)
5. [Metrik Evaluasi](#metrik-evaluasi)
6. [Timeline Eksekusi](#timeline-eksekusi)
7. [Expected Results](#expected-results)
8. [Analisis & Kesimpulan](#analisis--kesimpulan)

---

## 1️⃣ OVERVIEW EKSPERIMEN

### **Tujuan Penelitian**
Menentukan arsitektur CNN optimal untuk klasifikasi bakteri Gram dengan mempertimbangkan:
- ✅ Akurasi klasifikasi
- ✅ Kecepatan inferensi
- ✅ Ukuran model (deployment)
- ✅ Kemudahan training
- ✅ Generalisasi pada data baru

### **Research Questions**
1. Arsitektur mana yang memberikan akurasi tertinggi?
2. Apakah model besar selalu lebih baik dari model kecil?
3. Bagaimana trade-off antara akurasi dan efisiensi komputasi?
4. Model mana yang paling cocok untuk production deployment?

### **Dataset Split**
```
Total Images: ~8,000 citra bakteri
├── Training Set:   70% (~5,600 images)
├── Validation Set: 15% (~1,200 images)
└── Test Set:       15% (~1,200 images)

Class Distribution:
├── Gram Positif: ~4,000 images (50%)
└── Gram Negatif: ~4,000 images (50%)
```

---

## 2️⃣ ARSITEKTUR YANG DIUJI

### **A. Baseline Model (Custom CNN)**

#### **Spesifikasi:**
```python
Architecture: Simple 3-Layer CNN
Parameters: ~500K
Input Size: 224x224x3
Layers:
  - Conv2D(32) → BatchNorm → ReLU → MaxPool
  - Conv2D(64) → BatchNorm → ReLU → MaxPool
  - Conv2D(128) → BatchNorm → ReLU → MaxPool
  - Flatten → FC(256) → Dropout(0.5) → FC(2)
```

#### **Kelebihan:**
- ✅ Cepat untuk training
- ✅ Ukuran model kecil (~2 MB)
- ✅ Mudah di-debug
- ✅ Good baseline untuk comparison

#### **Kekurangan:**
- ❌ Akurasi terbatas (70-80%)
- ❌ Feature extraction sederhana
- ❌ Tidak pre-trained

---

### **B. ResNet50**

#### **Spesifikasi:**
```python
Architecture: Residual Network 50 layers
Parameters: ~25.6M (pre-trained ImageNet)
Input Size: 224x224x3
Key Features:
  - Residual connections (skip connections)
  - Batch Normalization
  - Global Average Pooling
  - Custom classifier head
```

#### **Kelebihan:**
- ✅ Standard di medical imaging
- ✅ Deep network tanpa vanishing gradient
- ✅ Pre-trained ImageNet (transfer learning)
- ✅ Proven track record

#### **Kekurangan:**
- ❌ Model besar (~98 MB)
- ❌ Inference lebih lambat
- ❌ Butuh GPU untuk training

---

### **C. ResNet101**

#### **Spesifikasi:**
```python
Architecture: Residual Network 101 layers
Parameters: ~44.5M (pre-trained ImageNet)
Input Size: 224x224x3
Difference from ResNet50:
  - Lebih deep (101 vs 50 layers)
  - Lebih banyak residual blocks
```

#### **Kelebihan:**
- ✅ Akurasi potensial lebih tinggi
- ✅ Feature extraction lebih kompleks
- ✅ Pre-trained ImageNet

#### **Kekurangan:**
- ❌ Model sangat besar (~170 MB)
- ❌ Training & inference lambat
- ❌ Risk overfitting pada dataset kecil

---

### **D. EfficientNet-B0**

#### **Spesifikasi:**
```python
Architecture: EfficientNet Baseline
Parameters: ~5.3M (pre-trained ImageNet)
Input Size: 224x224x3
Key Features:
  - Compound scaling (width, depth, resolution)
  - Mobile Inverted Bottleneck (MBConv)
  - Squeeze-and-Excitation blocks
```

#### **Kelebihan:**
- ✅ Best accuracy/parameter ratio
- ✅ Efficient computation
- ✅ Cocok untuk mobile deployment
- ✅ State-of-the-art architecture

#### **Kekurangan:**
- ❌ Lebih kompleks dari ResNet
- ❌ Training time moderate

---

### **E. EfficientNet-B3**

#### **Spesifikasi:**
```python
Architecture: EfficientNet Scaled
Parameters: ~12M (pre-trained ImageNet)
Input Size: 300x300x3 (higher resolution)
Scaling:
  - Width: 1.2x
  - Depth: 1.4x
  - Resolution: 1.25x
```

#### **Kelebihan:**
- ✅ Higher accuracy than B0
- ✅ Better feature extraction
- ✅ Still efficient

#### **Kekurangan:**
- ❌ Butuh lebih banyak memory
- ❌ Training time lebih lama

---

### **F. VGG16**

#### **Spesifikasi:**
```python
Architecture: Visual Geometry Group 16 layers
Parameters: ~138M (pre-trained ImageNet)
Input Size: 224x224x3
Key Features:
  - Simple sequential architecture
  - Only 3x3 convolutions
  - 5 Conv blocks + 3 FC layers
```

#### **Kelebihan:**
- ✅ Simple & interpretable
- ✅ Strong feature extraction
- ✅ Good for baseline comparison

#### **Kekurangan:**
- ❌ SANGAT banyak parameter
- ❌ Slow training & inference
- ❌ Outdated architecture

---

### **G. MobileNetV2**

#### **Spesifikasi:**
```python
Architecture: Mobile-optimized CNN
Parameters: ~3.5M (pre-trained ImageNet)
Input Size: 224x224x3
Key Features:
  - Depthwise separable convolutions
  - Inverted residual blocks
  - Linear bottlenecks
```

#### **Kelebihan:**
- ✅ SANGAT ringan (~14 MB)
- ✅ Fast inference (mobile-ready)
- ✅ Low memory footprint
- ✅ Good for edge devices

#### **Kekurangan:**
- ❌ Akurasi sedikit lebih rendah
- ❌ Feature capacity terbatas

---

## 3️⃣ METODOLOGI UJICOBA

### **A. Environment Setup**

```bash
Hardware Requirements:
├── CPU: Intel i5/i7 atau AMD Ryzen 5/7
├── RAM: Minimum 16GB (32GB recommended)
├── GPU: NVIDIA RTX 3050/3060 (8GB VRAM minimum)
└── Storage: 50GB free space

Software Stack:
├── Python: 3.10+
├── PyTorch: 2.0+
├── CUDA: 11.8+ (untuk GPU)
├── cuDNN: 8.6+
└── Dependencies: requirements.txt
```

### **B. Data Preprocessing**

```python
Preprocessing Pipeline:
1. Load & Parse Annotations
   ├── Dataset 1: COCO JSON → Extract ROI
   └── Dataset 2: LabelMe JSON → Extract labels

2. Image Processing
   ├── Resize to target size (224x224 atau 300x300)
   ├── Normalize: mean=[0.485, 0.456, 0.406]
   │            std=[0.229, 0.224, 0.225]
   └── Convert to tensor

3. Data Augmentation (Training only)
   ├── RandomRotation(30°)
   ├── RandomHorizontalFlip(p=0.5)
   ├── RandomVerticalFlip(p=0.5)
   ├── ColorJitter(brightness=0.2, contrast=0.2)
   ├── RandomResizedCrop(scale=(0.8, 1.0))
   └── GaussianNoise(std=0.01)

4. Label Encoding
   ├── Gram Positive (G+) → 0
   └── Gram Negative (G-) → 1
```

### **C. Training Configuration**

```python
Common Hyperparameters:
├── Batch Size: 16 (adjust based on model size)
├── Epochs: 100 (with early stopping patience=15)
├── Optimizer: AdamW
├── Initial LR: 0.0001
├── Weight Decay: 0.0001
├── Loss Function: CrossEntropyLoss (class-weighted)
└── LR Scheduler: ReduceLROnPlateau
    ├── Factor: 0.5
    ├── Patience: 5
    └── Min LR: 1e-7

Transfer Learning Strategy:
├── Phase 1: Freeze backbone, train classifier (10 epochs)
├── Phase 2: Unfreeze last 20% layers (20 epochs)
└── Phase 3: Fine-tune all layers (70 epochs)
```

### **D. Evaluation Protocol**

```python
Evaluation Steps:
1. Training Performance
   ├── Monitor loss & accuracy per epoch
   ├── Track validation metrics
   └── Save best model (based on val_acc)

2. Test Set Evaluation
   ├── Load best checkpoint
   ├── Predict on test set
   ├── Calculate all metrics
   └── Generate confusion matrix

3. Cross-Validation (Optional)
   ├── 5-fold stratified CV
   └── Report mean ± std

4. Statistical Testing
   ├── McNemar's test (model comparison)
   └── Confidence intervals (95%)
```

---

## 4️⃣ SKENARIO PER EKSPERIMEN

### **EKSPERIMEN 1: Baseline CNN**

#### **Tujuan:**
Establish baseline performance dengan custom simple CNN

#### **Skenario Ujicoba:**

```python
# Script: 03_train_baseline.py

Configuration:
├── Architecture: BaselineCNN (3 conv layers)
├── Input Size: 224x224
├── Batch Size: 32
├── Epochs: 50
├── Learning Rate: 0.001
├── Optimizer: Adam (betas=(0.9, 0.999))
├── Loss: CrossEntropyLoss
└── Augmentation: Basic (flip, rotate)

Expected Duration: 2-3 hours (GPU) / 8-12 hours (CPU)
```

#### **Success Criteria:**
- ✅ Accuracy: 70-80%
- ✅ Model converges (loss decreases)
- ✅ No overfitting (val_loss close to train_loss)
- ✅ Baseline untuk comparison

#### **Output:**
```
experiments/exp1_baseline/
├── best_model.pth
├── training_log.csv
├── training_curves.png
├── confusion_matrix.png
└── classification_report.txt
```

---

### **EKSPERIMEN 2A: ResNet50 Transfer Learning**

#### **Tujuan:**
Evaluate ResNet50 dengan transfer learning dari ImageNet

#### **Skenario Ujicoba:**

```python
# Script: 04_train_resnet50.py

Configuration:
├── Architecture: ResNet50 (pre-trained ImageNet)
├── Input Size: 224x224
├── Batch Size: 16
├── Epochs: 100
├── Learning Rate: 0.0001
├── Optimizer: AdamW (weight_decay=0.0001)
├── Loss: CrossEntropyLoss (class-weighted)
├── LR Scheduler: ReduceLROnPlateau
└── Augmentation: Advanced

Training Strategy:
Phase 1 (Epochs 1-10):
  └── Freeze backbone, train only classifier head

Phase 2 (Epochs 11-30):
  └── Unfreeze layer4 (last residual block)

Phase 3 (Epochs 31-100):
  └── Fine-tune all layers with low LR

Expected Duration: 6-8 hours (GPU RTX 3050)
```

#### **Success Criteria:**
- ✅ Accuracy: 85-92%
- ✅ Better than baseline
- ✅ AUC-ROC > 0.90
- ✅ Generalization on test set

#### **Output:**
```
experiments/exp2a_resnet50/
├── best_model.pth (~98 MB)
├── training_log.csv
├── learning_curves.png
├── confusion_matrix.png
├── roc_curve.png
├── feature_maps/ (visualizations)
└── metrics_summary.json
```

---

### **EKSPERIMEN 2B: ResNet101 Transfer Learning**

#### **Tujuan:**
Compare deeper network (ResNet101) vs ResNet50

#### **Skenario Ujicoba:**

```python
# Script: 04_train_resnet101.py

Configuration:
├── Architecture: ResNet101 (pre-trained ImageNet)
├── Input Size: 224x224
├── Batch Size: 8 (smaller due to memory)
├── Epochs: 100
├── Learning Rate: 0.00005 (lower for deeper net)
├── Optimizer: AdamW (weight_decay=0.0001)
├── Gradient Accumulation: 2 steps (effective batch=16)
└── Mixed Precision: FP16 (to save memory)

Hypothesis:
"Deeper network akan memberikan akurasi lebih tinggi,
 tapi dengan risiko overfitting pada dataset kecil"

Expected Duration: 10-12 hours (GPU)
```

#### **Success Criteria:**
- ✅ Accuracy >= ResNet50
- ✅ No severe overfitting (val_acc close to train_acc)
- ✅ AUC-ROC > 0.92

#### **Analysis Points:**
- Compare with ResNet50
- Check overfitting indicators
- Evaluate if depth helps with microscopic details

---

### **EKSPERIMEN 3A: EfficientNet-B0**

#### **Tujuan:**
Test efficient architecture dengan parameter minimal

#### **Skenario Ujicoba:**

```python
# Script: 05_train_efficientnet_b0.py

Configuration:
├── Architecture: EfficientNet-B0 (pre-trained ImageNet)
├── Input Size: 224x224
├── Batch Size: 24
├── Epochs: 100
├── Learning Rate: 0.0001
├── Optimizer: AdamW
├── Dropout: 0.3 (in classifier head)
└── Stochastic Depth: 0.2

Hypothesis:
"EfficientNet-B0 akan memberikan akurasi mendekati ResNet50
 dengan ukuran model jauh lebih kecil"

Expected Duration: 5-7 hours (GPU)
```

#### **Success Criteria:**
- ✅ Accuracy: 88-93%
- ✅ Model size < 25 MB
- ✅ Inference time < ResNet50
- ✅ Best accuracy/size ratio

#### **Comparison Focus:**
- Accuracy vs ResNet50
- Model size vs ResNet50
- Training time vs ResNet50
- Inference speed vs ResNet50

---

### **EKSPERIMEN 3B: EfficientNet-B3**

#### **Tujuan:**
Test scaled EfficientNet dengan higher resolution

#### **Skenario Ujicoba:**

```python
# Script: 05_train_efficientnet_b3.py

Configuration:
├── Architecture: EfficientNet-B3 (pre-trained ImageNet)
├── Input Size: 300x300 (higher resolution)
├── Batch Size: 12
├── Epochs: 100
├── Learning Rate: 0.00008
├── Optimizer: AdamW
├── Dropout: 0.4
└── Stochastic Depth: 0.3

Hypothesis:
"Higher resolution (300x300) akan capture detail
 mikroskopis lebih baik, meningkatkan akurasi"

Expected Duration: 8-10 hours (GPU)
```

#### **Success Criteria:**
- ✅ Accuracy > EfficientNet-B0
- ✅ Accuracy: 90-95%
- ✅ Better feature extraction

#### **Analysis Points:**
- Impact of higher resolution
- Diminishing returns analysis
- Cost-benefit vs B0

---

### **EKSPERIMEN 4: VGG16**

#### **Tujuan:**
Baseline comparison dengan classical deep architecture

#### **Skenario Ujicoba:**

```python
# Script: 06_train_vgg16.py

Configuration:
├── Architecture: VGG16 (pre-trained ImageNet)
├── Input Size: 224x224
├── Batch Size: 8 (large model)
├── Epochs: 80 (shorter due to slow training)
├── Learning Rate: 0.00005
├── Optimizer: SGD (momentum=0.9) # VGG works better with SGD
├── Dropout: 0.5 (VGG default)
└── Gradient Clipping: 1.0

Warning:
"VGG16 sangat besar dan lambat. Digunakan hanya
 untuk comparison dengan modern architectures"

Expected Duration: 12-15 hours (GPU)
```

#### **Success Criteria:**
- ✅ Model converges
- ✅ Accuracy baseline established
- ⚠️ Expected lower than EfficientNet

#### **Analysis Focus:**
- Old vs new architecture
- Parameter efficiency
- NOT recommended for production

---

### **EKSPERIMEN 5: MobileNetV2**

#### **Tujuan:**
Test mobile-optimized architecture untuk deployment

#### **Skenario Ujicoba:**

```python
# Script: 07_train_mobilenet_v2.py

Configuration:
├── Architecture: MobileNetV2 (pre-trained ImageNet)
├── Input Size: 224x224
├── Batch Size: 32
├── Epochs: 100
├── Learning Rate: 0.0001
├── Optimizer: RMSprop (alpha=0.9) # Better for MobileNet
├── Dropout: 0.2
└── Width Multiplier: 1.0

Hypothesis:
"MobileNetV2 akan memberikan akurasi acceptable
 dengan inference speed terbaik untuk mobile deployment"

Expected Duration: 4-6 hours (GPU)
```

#### **Success Criteria:**
- ✅ Accuracy: 83-88%
- ✅ Model size < 15 MB
- ✅ Fastest inference time
- ✅ Mobile-ready

#### **Deployment Testing:**
```python
Additional Tests:
1. Quantization (INT8)
   └── Test accuracy degradation

2. ONNX Export
   └── Test cross-platform compatibility

3. Mobile Inference Benchmark
   ├── Android (CPU)
   ├── iOS (CPU)
   └── Raspberry Pi 4

4. Latency Testing
   └── Target: < 100ms per image
```

---

### **EKSPERIMEN 6: Ensemble Models**

#### **Tujuan:**
Combine top 3 models untuk maximize accuracy

#### **Skenario Ujicoba:**

```python
# Script: 08_train_ensemble.py

Strategy 1: Hard Voting
├── Models: ResNet50 + EfficientNet-B3 + MobileNetV2
├── Method: Majority vote
└── Prediction: argmax(votes)

Strategy 2: Soft Voting (Weighted Average)
├── Models: ResNet50 + EfficientNet-B3 + MobileNetV2
├── Weights: [0.4, 0.4, 0.2] (based on individual performance)
└── Prediction: argmax(weighted_average(probabilities))

Strategy 3: Stacking
├── Base Models: ResNet50, EfficientNet-B3, MobileNetV2
├── Meta-Learner: Logistic Regression
└── Training: 5-fold CV predictions → train meta-learner

Expected Duration: 2 hours (using pre-trained models)
```

#### **Success Criteria:**
- ✅ Accuracy > best individual model
- ✅ AUC-ROC > 0.95
- ✅ Reduced variance

#### **Analysis:**
- Diversity of predictions
- Improvement margin
- Inference time trade-off

---

### **EKSPERIMEN 7: Hyperparameter Optimization**

#### **Tujuan:**
Optimize best model from Exp 1-6 menggunakan Optuna

#### **Skenario Ujicoba:**

```python
# Script: 09_hyperparameter_tuning.py

Search Space:
├── Learning Rate: [1e-5, 1e-4, 1e-3] (log-uniform)
├── Batch Size: [8, 16, 32]
├── Dropout: [0.2, 0.3, 0.4, 0.5]
├── Weight Decay: [1e-5, 1e-4, 1e-3]
├── Optimizer: [Adam, AdamW, SGD, RMSprop]
├── LR Scheduler: [ReduceLR, CosineAnnealing, StepLR]
└── Data Augmentation Intensity: [low, medium, high]

Optimization:
├── Framework: Optuna
├── Trials: 50
├── Objective: Maximize validation accuracy
├── Pruning: Median pruner (early stop bad trials)
└── Sampler: TPE (Tree-structured Parzen Estimator)

Expected Duration: 3-5 days (parallel trials on GPU)
```

#### **Success Criteria:**
- ✅ Find optimal hyperparameters
- ✅ Accuracy improvement > 1%
- ✅ Statistical significance

#### **Output:**
```
experiments/exp7_hyperparameter_tuning/
├── study.db (Optuna database)
├── optimization_history.png
├── param_importances.png
├── best_params.json
└── best_model.pth
```

---

## 5️⃣ METRIK EVALUASI

### **A. Classification Metrics**

```python
Primary Metrics:
1. Accuracy = (TP + TN) / Total
   └── Target: ≥ 90%

2. Precision = TP / (TP + FP)
   └── Target: ≥ 88%

3. Recall (Sensitivity) = TP / (TP + FN)
   └── Target: ≥ 88%

4. F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
   └── Target: ≥ 88%

5. Specificity = TN / (TN + FP)
   └── Target: ≥ 88%

6. AUC-ROC (Area Under ROC Curve)
   └── Target: ≥ 0.95

7. Cohen's Kappa
   └── Target: ≥ 0.85 (almost perfect agreement)
```

### **B. Confusion Matrix Analysis**

```
                 Predicted
                 G+    G-
Actual   G+     [TP]  [FN]
         G-     [FP]  [TN]

Analysis Points:
- False Positives: G- diprediksi sebagai G+ (risk: wrong antibiotic)
- False Negatives: G+ diprediksi sebagai G- (risk: missed diagnosis)
- Medical context: Minimize FN lebih penting
```

### **C. Model Efficiency Metrics**

```python
1. Model Size (MB)
   ├── Baseline CNN: ~2 MB
   ├── MobileNetV2: ~14 MB
   ├── EfficientNet-B0: ~20 MB
   ├── ResNet50: ~98 MB
   └── VGG16: ~528 MB

2. Inference Time (ms per image)
   ├── Device: NVIDIA RTX 3050
   ├── Batch Size: 1
   └── Target: < 50ms

3. Training Time (hours)
   └── Budget: < 12 hours per experiment

4. GPU Memory Usage (GB)
   └── Constraint: < 8GB VRAM

5. FLOPs (Floating Point Operations)
   └── Compare computational complexity
```

### **D. Generalization Metrics**

```python
1. Train-Val Gap
   └── |train_acc - val_acc| < 5% (no overfitting)

2. Cross-Validation Stability
   └── std(cv_scores) < 2% (consistent performance)

3. Test Set Performance
   └── test_acc ≥ val_acc - 2% (good generalization)

4. Per-Class Performance
   └── Check for class-specific biases
```

---

## 6️⃣ TIMELINE EKSEKUSI

### **Week 1: Baseline & Setup**

```
Day 1-2: Environment Setup
├── ✅ Install dependencies
├── ✅ Prepare datasets
├── ✅ Data preprocessing pipeline
└── ✅ Verify GPU setup

Day 3-5: Experiment 1 (Baseline CNN)
├── ✅ Train baseline model
├── ✅ Evaluate & document
└── ✅ Establish baseline metrics

Day 6-7: Code Review & Documentation
└── ✅ Refactor reusable components
```

### **Week 2: Transfer Learning - Part 1**

```
Day 1-3: Experiment 2A (ResNet50)
├── Train ResNet50
├── Hyperparameter search (mini)
└── Document results

Day 4-6: Experiment 3A (EfficientNet-B0)
├── Train EfficientNet-B0
├── Compare with ResNet50
└── Document results

Day 7: Analysis & Interim Report
└── Compare 3 models so far
```

### **Week 3: Transfer Learning - Part 2**

```
Day 1-3: Experiment 2B (ResNet101)
└── Train & compare with ResNet50

Day 4-6: Experiment 3B (EfficientNet-B3)
└── Train & compare with B0

Day 7: Mid-experiment review
└── Determine top 3 models
```

### **Week 4: Remaining Architectures**

```
Day 1-3: Experiment 4 (VGG16)
└── Train & baseline comparison

Day 4-6: Experiment 5 (MobileNetV2)
└── Train & mobile optimization

Day 7: Compile all results
```

### **Week 5: Ensemble & Optimization**

```
Day 1-2: Experiment 6 (Ensemble)
├── Hard voting
├── Soft voting
└── Stacking

Day 3-7: Experiment 7 (Hyperparameter Tuning)
├── Setup Optuna study
├── Run 50 trials
└── Train final optimized model
```

### **Week 6: Final Analysis**

```
Day 1-3: Comprehensive Testing
├── Test set evaluation
├── Cross-validation
├── Statistical tests
└── Error analysis

Day 4-7: Documentation & Reporting
├── Write final report
├── Create visualizations
├── Prepare presentation
└── Code cleanup
```

**Total Duration: 6 weeks (42 days)**

---

## 7️⃣ EXPECTED RESULTS

### **Performance Ranking (Predicted)**

```
Accuracy Ranking:
1. 🥇 EfficientNet-B3:      91-95% ⭐ BEST ACCURACY
2. 🥈 Ensemble (Top 3):     90-94%
3. 🥉 ResNet50:             88-92%
4.     EfficientNet-B0:     87-91%
5.     ResNet101:           87-91%
6.     MobileNetV2:         83-88%
7.     VGG16:               82-87%
8.     Baseline CNN:        70-80%

Efficiency Ranking (Speed):
1. 🥇 MobileNetV2:          ~15ms/image ⚡ FASTEST
2. 🥈 EfficientNet-B0:      ~25ms/image
3. 🥉 Baseline CNN:         ~10ms/image (but low accuracy)
4.     ResNet50:            ~40ms/image
5.     EfficientNet-B3:     ~60ms/image
6.     ResNet101:           ~80ms/image
7.     VGG16:               ~100ms/image
8.     Ensemble:            ~125ms/image

Model Size Ranking:
1. 🥇 Baseline CNN:         ~2 MB
2. 🥈 MobileNetV2:          ~14 MB ⭐ BEST SIZE/ACC
3. 🥉 EfficientNet-B0:      ~20 MB
4.     EfficientNet-B3:     ~48 MB
5.     ResNet50:            ~98 MB
6.     ResNet101:           ~171 MB
7.     VGG16:               ~528 MB
```

### **Recommended Model by Use Case**

```python
Use Case 1: Production Medical System (Hospital Lab)
Recommendation: EfficientNet-B3
├── Akurasi: 91-95% (highest)
├── Size: 48 MB (acceptable)
├── Speed: 60ms (acceptable untuk lab)
└── Justifikasi: Prioritas akurasi untuk diagnosis

Use Case 2: Mobile Point-of-Care Device
Recommendation: MobileNetV2
├── Akurasi: 83-88% (acceptable)
├── Size: 14 MB (mobile-friendly)
├── Speed: 15ms (real-time capable)
└── Justifikasi: Balance akurasi & efisiensi

Use Case 3: High-Throughput Screening
Recommendation: EfficientNet-B0
├── Akurasi: 87-91% (good)
├── Size: 20 MB (moderate)
├── Speed: 25ms (fast enough)
└── Justifikasi: Best balance untuk batch processing

Use Case 4: Research / Maximum Accuracy
Recommendation: Ensemble (ResNet50 + EfficientNet-B3 + MobileNetV2)
├── Akurasi: 90-94% (ensemble boost)
├── Size: ~160 MB combined
├── Speed: 125ms (acceptable untuk research)
└── Justifikasi: Squeeze every % of accuracy
```

---

## 8️⃣ ANALISIS & KESIMPULAN

### **A. Comparative Analysis Template**

```markdown
## Model Comparison Summary

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | Size | Speed |
|-------|----------|-----------|--------|-----|---------|------|-------|
| Baseline CNN | XX% | XX% | XX% | XX% | 0.XX | 2MB | XXms |
| ResNet50 | XX% | XX% | XX% | XX% | 0.XX | 98MB | XXms |
| ResNet101 | XX% | XX% | XX% | XX% | 0.XX | 171MB | XXms |
| EfficientNet-B0 | XX% | XX% | XX% | XX% | 0.XX | 20MB | XXms |
| EfficientNet-B3 | XX% | XX% | XX% | XX% | 0.XX | 48MB | XXms |
| VGG16 | XX% | XX% | XX% | XX% | 0.XX | 528MB | XXms |
| MobileNetV2 | XX% | XX% | XX% | XX% | 0.XX | 14MB | XXms |
| Ensemble | XX% | XX% | XX% | XX% | 0.XX | 160MB | XXms |

### Key Findings:

1. **Best Overall Model**: [Model Name]
   - Justifikasi: ...

2. **Best Efficiency**: [Model Name]
   - Trade-off analysis: ...

3. **Production Recommendation**: [Model Name]
   - Deployment considerations: ...
```

### **B. Statistical Analysis**

```python
Statistical Tests to Perform:

1. McNemar's Test
   └── Compare paired predictions (model A vs model B)
   └── Determine statistical significance of accuracy difference

2. Confidence Intervals (95%)
   └── Report accuracy as: XX.X% ± Y.Y%

3. ANOVA
   └── Compare multiple models simultaneously

4. Effect Size (Cohen's d)
   └── Measure practical significance
```

### **C. Error Analysis**

```python
Qualitative Analysis:

1. Misclassified Samples Review
   ├── Visualize top 20 false positives
   ├── Visualize top 20 false negatives
   └── Identify patterns in errors

2. Confusion Patterns
   ├── Which class is harder to classify?
   ├── Are errors consistent across models?
   └── Sample quality issues?

3. Feature Visualization
   ├── Grad-CAM heatmaps
   ├── What does model "look at"?
   └── Interpretability analysis
```

### **D. Lessons Learned**

```markdown
Document for Each Experiment:

1. What worked?
   └── [specific findings]

2. What didn't work?
   └── [failures & reasons]

3. Unexpected results?
   └── [surprises]

4. Recommendations for future work?
   └── [improvements]
```

---

## 9️⃣ SCRIPTS & AUTOMATION

### **Master Training Script**

```python
# run_all_experiments.py

import subprocess
import logging
from datetime import datetime

experiments = [
    "03_train_baseline.py",
    "04_train_resnet50.py",
    "04_train_resnet101.py",
    "05_train_efficientnet_b0.py",
    "05_train_efficientnet_b3.py",
    "06_train_vgg16.py",
    "07_train_mobilenet_v2.py",
    "08_train_ensemble.py",
    "09_hyperparameter_tuning.py"
]

for exp in experiments:
    logging.info(f"Starting {exp} at {datetime.now()}")
    subprocess.run(["python", exp])
    logging.info(f"Completed {exp} at {datetime.now()}")
```

### **Results Aggregation Script**

```python
# aggregate_results.py

import pandas as pd
import json

def load_experiment_results(exp_dir):
    """Load metrics from experiment directory"""
    with open(f"{exp_dir}/metrics_summary.json") as f:
        return json.load(f)

def create_comparison_table():
    """Create comprehensive comparison table"""
    results = []
    for exp in experiments:
        metrics = load_experiment_results(exp)
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df.to_csv("experiments/comparison_table.csv")
    df.to_latex("experiments/comparison_table.tex")
    
create_comparison_table()
```

---

## 🔟 DELIVERABLES

### **Per Experiment:**

```
experiments/expX_[model_name]/
├── 📊 training_log.csv
├── 📈 learning_curves.png
├── 🎯 confusion_matrix.png
├── 📉 roc_curve.png
├── 💾 best_model.pth
├── 📝 classification_report.txt
├── 📋 metrics_summary.json
├── 🖼️ sample_predictions.png
└── 📄 experiment_notes.md
```

### **Final Report:**

```
doc/LAPORAN_HASIL_UJICOBA.md
├── 1. Executive Summary
├── 2. Methodology
├── 3. Individual Experiment Results
├── 4. Comparative Analysis
├── 5. Statistical Tests
├── 6. Error Analysis
├── 7. Deployment Recommendations
├── 8. Conclusions
└── 9. Appendices
    ├── A. Hyperparameters per model
    ├── B. Training logs
    └── C. Code repository
```

### **Presentation Materials:**

```
presentations/
├── slides_results.pptx
├── demo_video.mp4
└── poster_scientific.pdf
```

---

## ✅ QUALITY CHECKLIST

### **Before Starting Each Experiment:**

- [ ] Dataset preprocessed & verified
- [ ] Train/val/test split consistent
- [ ] GPU availability confirmed
- [ ] Disk space sufficient
- [ ] Logging configured
- [ ] Checkpoint saving enabled
- [ ] Random seeds set (reproducibility)

### **During Experiment:**

- [ ] Monitor GPU utilization
- [ ] Check training curves (overfitting?)
- [ ] Validate data loading (no corruption)
- [ ] Log hyperparameters
- [ ] Save intermediate checkpoints

### **After Experiment:**

- [ ] Evaluate on test set
- [ ] Generate all visualizations
- [ ] Document findings
- [ ] Compare with previous experiments
- [ ] Archive model & logs
- [ ] Update comparison table

---

## 📚 REFERENCES

### **Papers:**
1. He et al. (2016) - Deep Residual Learning (ResNet)
2. Tan & Le (2019) - EfficientNet: Rethinking Model Scaling
3. Sandler et al. (2018) - MobileNetV2
4. Simonyan & Zisserman (2014) - Very Deep Convolutional Networks (VGG)
5. Smith et al. (2020) - Deep Learning for Gram Stain Interpretation

### **Tools:**
- PyTorch Documentation: https://pytorch.org/docs/
- Optuna: https://optuna.org/
- Weights & Biases: https://wandb.ai/

---

## 📞 SUPPORT

**Troubleshooting:**
- GPU Out of Memory → Reduce batch size
- Slow training → Check data loading (use num_workers)
- Poor accuracy → Check data augmentation, learning rate
- Overfitting → Add dropout, reduce model complexity

**Contact:**
- Email: [your.email@example.com]
- Lab: [Lab name & location]

---

**Document Version**: 1.0  
**Last Updated**: November 25, 2025  
**Author**: [Nama Anda]  
**Status**: Ready for Execution ✅

---

🎯 **GOOD LUCK WITH YOUR EXPERIMENTS!** 🔬
