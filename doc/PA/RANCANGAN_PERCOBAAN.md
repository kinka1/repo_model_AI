# RANCANGAN PERCOBAAN (EXPERIMENTAL DESIGN)
## KLASIFIKASI BAKTERI GRAM POSITIF DAN GRAM NEGATIF

---

## 📊 OVERVIEW

Dokumen ini merinci seluruh eksperimen yang akan dilakukan untuk mencapai model AI terbaik dalam mengklasifikasikan bakteri Gram positif dan Gram negatif.

---

## 🎯 EKSPERIMEN 1: BASELINE MODEL

### Tujuan
Menetapkan baseline performance dengan model CNN sederhana untuk membandingkan dengan model yang lebih kompleks.

### Hipotesis
Model CNN sederhana dengan 3-4 layer konvolusi dapat mencapai akurasi >75% pada data bakteri yang relatif terstruktur.

### Metodologi

**Model Architecture:**
```
Input (640×640×3)
  ↓
Conv2D(32, 3×3) → ReLU → BatchNorm → MaxPool(2×2)
  ↓
Conv2D(64, 3×3) → ReLU → BatchNorm → MaxPool(2×2)
  ↓
Conv2D(128, 3×3) → ReLU → BatchNorm → MaxPool(2×2)
  ↓
Conv2D(256, 3×3) → ReLU → BatchNorm → MaxPool(2×2)
  ↓
GlobalAvgPool → Flatten
  ↓
Dense(128) → ReLU → Dropout(0.5)
  ↓
Dense(2) → Softmax
```

**Hyperparameters:**
```yaml
learning_rate: 0.001
batch_size: 32
epochs: 50
optimizer: Adam
loss: CrossEntropyLoss
weight_decay: 0.0001
dropout: 0.5
```

**Training Strategy:**
- No pretrained weights
- Train from scratch
- Basic augmentation (flip, rotate)
- Early stopping patience: 10 epochs

### Metrics to Track
- Training loss & accuracy
- Validation loss & accuracy
- Inference time per image
- Model size (MB)
- Overfitting gap (train_acc - val_acc)

### Expected Results
- Validation Accuracy: 70-80%
- Training time: ~2-3 hours
- Model size: <50MB
- Overfitting: Expected due to limited model capacity

### Files to Create
```
experiments/exp1_baseline/
├── train.py
├── model.py
├── config.yaml
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── metrics.json
│   └── model_baseline.pth
└── logs/
    └── tensorboard/
```

---

## 🚀 EKSPERIMEN 2: TRANSFER LEARNING COMPARISON

### Tujuan
Membandingkan performa berbagai arsitektur pre-trained (ResNet, EfficientNet, VGG, MobileNet) untuk menemukan yang terbaik.

### Hipotesis
EfficientNet akan memberikan trade-off terbaik antara akurasi dan efisiensi komputasi, dengan akurasi >85%.

### Metodologi

**Models to Compare:**

1. **ResNet50**
   - Parameters: 25.6M
   - Pre-trained: ImageNet
   - Strategy: Fine-tune all layers

2. **ResNet101**
   - Parameters: 44.5M
   - Pre-trained: ImageNet
   - Strategy: Fine-tune all layers

3. **EfficientNet-B0**
   - Parameters: 5.3M
   - Pre-trained: ImageNet
   - Strategy: Fine-tune all layers

4. **EfficientNet-B3**
   - Parameters: 12M
   - Pre-trained: ImageNet
   - Strategy: Fine-tune all layers

5. **VGG16**
   - Parameters: 138M
   - Pre-trained: ImageNet
   - Strategy: Freeze backbone, train classifier

6. **MobileNetV2**
   - Parameters: 3.5M
   - Pre-trained: ImageNet
   - Strategy: Fine-tune all layers

**Hyperparameters (Same for all):**
```yaml
learning_rate: 0.0001
batch_size: 16
epochs: 100
optimizer: AdamW
loss: CrossEntropyLoss(label_smoothing=0.1)
weight_decay: 0.0001
dropout: 0.5
scheduler: CosineAnnealingLR
warmup_epochs: 5
```

**Training Strategy:**
```python
# Phase 1: Warmup (5 epochs)
- Low learning rate: 1e-5
- Train only classifier layers

# Phase 2: Fine-tuning (95 epochs)
- Learning rate: 1e-4
- Train all layers
- Cosine annealing scheduler
```

### Comparison Metrics

| Model | Val Acc | Precision | Recall | F1 | AUC | Params | Size(MB) | Inference(ms) |
|-------|---------|-----------|--------|----|----|--------|----------|---------------|
| ResNet50 | ? | ? | ? | ? | ? | 25.6M | ~100 | ? |
| ResNet101 | ? | ? | ? | ? | ? | 44.5M | ~171 | ? |
| EfficientNet-B0 | ? | ? | ? | ? | ? | 5.3M | ~20 | ? |
| EfficientNet-B3 | ? | ? | ? | ? | ? | 12M | ~47 | ? |
| VGG16 | ? | ? | ? | ? | ? | 138M | ~528 | ? |
| MobileNetV2 | ? | ? | ? | ? | ? | 3.5M | ~14 | ? |

### Expected Results
- Best Accuracy: EfficientNet-B3 (>87%)
- Best Efficiency: MobileNetV2 or EfficientNet-B0
- ResNet models: Good baseline (85-86%)
- VGG16: High memory usage, moderate accuracy

### Decision Criteria
**Select best model based on:**
1. Validation Accuracy (weight: 40%)
2. F1-Score (weight: 30%)
3. Inference Time (weight: 20%)
4. Model Size (weight: 10%)

**Formula:**
```
Score = 0.4×Val_Acc + 0.3×F1 + 0.2×(1/Inference_Time_normalized) + 0.1×(1/Model_Size_normalized)
```

### Files to Create
```
experiments/exp2_transfer_learning/
├── resnet50/
│   ├── train.py
│   ├── config.yaml
│   └── results/
├── efficientnet_b0/
├── efficientnet_b3/
├── vgg16/
├── mobilenet_v2/
└── comparison_report.md
```

---

## 🔧 EKSPERIMEN 3: HYPERPARAMETER TUNING

### Tujuan
Mengoptimalkan hyperparameter model terbaik dari Eksperimen 2 menggunakan Optuna.

### Hipotesis
Hyperparameter tuning dapat meningkatkan akurasi 2-3% dari baseline model terpilih.

### Metodologi

**Base Model:** (Pilih dari Exp 2, misal: EfficientNet-B3)

**Search Space:**
```python
{
    'learning_rate': [1e-5, 1e-4, 5e-4, 1e-3],
    'batch_size': [8, 16, 32],
    'dropout': [0.3, 0.4, 0.5, 0.6, 0.7],
    'weight_decay': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    'optimizer': ['adam', 'adamw', 'sgd'],
    'scheduler': ['cosine', 'step', 'plateau'],
    'label_smoothing': [0.0, 0.05, 0.1, 0.15],
    'mixup_alpha': [0.0, 0.2, 0.4],
    'cutmix_alpha': [0.0, 0.2, 0.4]
}
```

**Optimization Strategy:**
```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    dropout = trial.suggest_uniform('dropout', 0.3, 0.7)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
    
    # Train model with these params
    val_acc = train_and_evaluate(lr, batch_size, dropout, weight_decay)
    
    return val_acc

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, timeout=86400)  # 24 hours

# Best params
print(f"Best hyperparameters: {study.best_params}")
print(f"Best validation accuracy: {study.best_value}")
```

**Trials Configuration:**
- Number of trials: 50
- Timeout: 24 hours
- Pruning: MedianPruner (stop unpromising trials early)
- Sampler: TPE (Tree-structured Parzen Estimator)

### Expected Results
- Improvement: +2-3% validation accuracy
- Best learning rate: ~5e-5 to 1e-4
- Best batch size: 16 or 32
- Optimal dropout: 0.4-0.5

### Visualization
```python
# Optuna visualization
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
optuna.visualization.plot_parallel_coordinate(study)
optuna.visualization.plot_slice(study)
```

### Files to Create
```
experiments/exp3_hyperparameter_tuning/
├── optuna_search.py
├── config.yaml
├── results/
│   ├── study.db
│   ├── best_params.json
│   ├── optimization_history.png
│   ├── param_importances.png
│   └── parallel_coordinates.png
└── best_model.pth
```

---

## 🎨 EKSPERIMEN 4: DATA AUGMENTATION IMPACT

### Tujuan
Mengukur dampak berbagai teknik augmentasi data terhadap performa model.

### Hipotesis
Heavy augmentation dengan mixup/cutmix akan mencegah overfitting dan meningkatkan generalisasi.

### Metodologi

**Augmentation Variants:**

**1. No Augmentation (Baseline)**
```python
transform = A.Compose([
    A.Resize(640, 640),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

**2. Basic Augmentation**
```python
transform = A.Compose([
    A.Resize(640, 640),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

**3. Moderate Augmentation**
```python
transform = A.Compose([
    A.Resize(640, 640),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

**4. Heavy Augmentation**
```python
transform = A.Compose([
    A.Resize(640, 640),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.7),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.7),
    A.GaussianBlur(blur_limit=(3, 9), p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianNoise(var_limit=(10, 50), p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
    A.GridDistortion(p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

**5. Heavy + Mixup**
```python
# Heavy augmentation + Mixup during training
alpha = 0.2
lambda = np.random.beta(alpha, alpha)
mixed_x = lambda * x1 + (1 - lambda) * x2
mixed_y = lambda * y1 + (1 - lambda) * y2
```

**6. Heavy + CutMix**
```python
# Heavy augmentation + CutMix during training
def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y, y[rand_index], lam
```

### Comparison Matrix

| Augmentation Type | Train Acc | Val Acc | Test Acc | Overfitting Gap | Training Time |
|-------------------|-----------|---------|----------|----------------|---------------|
| No Aug | ? | ? | ? | ? | ? |
| Basic | ? | ? | ? | ? | ? |
| Moderate | ? | ? | ? | ? | ? |
| Heavy | ? | ? | ? | ? | ? |
| Heavy + Mixup | ? | ? | ? | ? | ? |
| Heavy + CutMix | ? | ? | ? | ? | ? |

### Expected Results
- No Augmentation: High overfitting (gap >15%)
- Basic: Moderate overfitting (gap ~10%)
- Moderate: Best trade-off (gap ~5%)
- Heavy: May hurt performance on simple patterns
- Mixup/CutMix: Best generalization (gap <5%)

### Files to Create
```
experiments/exp4_augmentation/
├── no_aug/
├── basic_aug/
├── moderate_aug/
├── heavy_aug/
├── mixup/
├── cutmix/
└── comparison_report.md
```

---

## ⚖️ EKSPERIMEN 5: CLASS IMBALANCE HANDLING

### Tujuan
Mengatasi ketidakseimbangan kelas (jika ada) antara Gram+ dan Gram-.

### Hipotesis
Class-weighted loss atau focal loss akan meningkatkan recall untuk kelas minoritas.

### Metodologi

**Check Imbalance:**
```python
gram_pos_count = (df['label'] == 'G+').sum()
gram_neg_count = (df['label'] == 'G-').sum()
imbalance_ratio = max(gram_pos_count, gram_neg_count) / min(gram_pos_count, gram_neg_count)

print(f"Imbalance Ratio: {imbalance_ratio:.2f}")
# If > 1.5, apply class balancing techniques
```

**Techniques to Compare:**

**1. Baseline (No handling)**
```python
criterion = nn.CrossEntropyLoss()
```

**2. Class-Weighted Loss**
```python
# Calculate class weights
class_counts = torch.tensor([gram_pos_count, gram_neg_count])
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()

criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**3. Focal Loss**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

**4. Oversampling (SMOTE-like)**
```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
```

**5. Undersampling**
```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
```

**6. Class-Balanced Sampling**
```python
from torch.utils.data import WeightedRandomSampler

# Calculate sample weights
class_sample_count = torch.tensor([(y_train == t).sum() for t in torch.unique(y_train)])
weight = 1. / class_sample_count.float()
samples_weight = torch.tensor([weight[t] for t in y_train])

sampler = WeightedRandomSampler(
    weights=samples_weight,
    num_samples=len(samples_weight),
    replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
```

### Evaluation Metrics (Focus on minority class)

| Method | Overall Acc | Minority Precision | Minority Recall | Minority F1 | Macro F1 |
|--------|-------------|-------------------|----------------|-------------|----------|
| Baseline | ? | ? | ? | ? | ? |
| Class Weights | ? | ? | ? | ? | ? |
| Focal Loss | ? | ? | ? | ? | ? |
| Oversampling | ? | ? | ? | ? | ? |
| Undersampling | ? | ? | ? | ? | ? |
| Balanced Sampling | ? | ? | ? | ? | ? |

### Expected Results
- Weighted Loss: +3-5% recall on minority class
- Focal Loss: Best for hard samples
- Oversampling: May cause overfitting
- Undersampling: May lose information
- Balanced Sampling: Good compromise

### Files to Create
```
experiments/exp5_class_imbalance/
├── baseline/
├── weighted_loss/
├── focal_loss/
├── oversampling/
├── undersampling/
├── balanced_sampling/
└── comparison_report.md
```

---

## 🤝 EKSPERIMEN 6: ENSEMBLE METHODS

### Tujuan
Meningkatkan akurasi dan robustness dengan menggabungkan prediksi dari multiple models.

### Hipotesis
Ensemble dari 3-5 model terbaik dapat meningkatkan akurasi 1-2% dibanding single best model.

### Metodologi

**Select Top Models from Exp 2:**
```python
# Example: Top 3 models
models = [
    EfficientNet-B3 (val_acc: 0.89),
    ResNet50 (val_acc: 0.87),
    EfficientNet-B0 (val_acc: 0.86)
]
```

**Ensemble Strategies:**

**1. Hard Voting (Majority Vote)**
```python
def hard_voting(models, x):
    predictions = []
    for model in models:
        with torch.no_grad():
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            predictions.append(pred)
    
    predictions = torch.stack(predictions)
    final_pred = torch.mode(predictions, dim=0)[0]
    return final_pred
```

**2. Soft Voting (Weighted Average)**
```python
def soft_voting(models, x, weights=None):
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    ensemble_probs = torch.zeros(x.size(0), num_classes)
    
    for model, weight in zip(models, weights):
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            ensemble_probs += weight * probs
    
    final_pred = torch.argmax(ensemble_probs, dim=1)
    return final_pred, ensemble_probs

# Weights based on validation accuracy
weights = [0.40, 0.35, 0.25]  # EfficientNet-B3, ResNet50, EfficientNet-B0
```

**3. Stacking with Meta-Learner**
```python
# Level 0: Base models predictions
base_predictions = []
for model in models:
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        base_predictions.append(probs)

# Stack predictions
stacked_features = torch.cat(base_predictions, dim=1)

# Level 1: Meta-learner (Logistic Regression)
meta_learner = LogisticRegression()
meta_learner.fit(stacked_features_train, y_train)
final_pred = meta_learner.predict(stacked_features_test)
```

**4. Weighted Stacking**
```python
# Optimize weights using validation set
from scipy.optimize import minimize

def ensemble_loss(weights, predictions, targets):
    weights = weights / weights.sum()  # Normalize
    ensemble_pred = (predictions * weights[:, None, None]).sum(axis=0)
    loss = F.cross_entropy(torch.tensor(ensemble_pred), torch.tensor(targets))
    return loss.item()

# Optimize
initial_weights = np.array([1.0] * len(models))
result = minimize(
    ensemble_loss,
    initial_weights,
    args=(val_predictions, val_targets),
    method='SLSQP',
    bounds=[(0, 1)] * len(models)
)

optimal_weights = result.x / result.x.sum()
```

### Comparison

| Ensemble Method | Val Acc | Test Acc | Inference Time | Memory Usage |
|-----------------|---------|----------|----------------|--------------|
| Single Best | ? | ? | ? | ? |
| Hard Voting | ? | ? | ? | ? |
| Soft Voting (Equal) | ? | ? | ? | ? |
| Soft Voting (Weighted) | ? | ? | ? | ? |
| Stacking (LR) | ? | ? | ? | ? |
| Weighted Stacking | ? | ? | ? | ? |

### Expected Results
- Hard Voting: +0.5-1% accuracy
- Soft Voting (Weighted): +1-1.5% accuracy
- Stacking: +1.5-2% accuracy
- Trade-off: 3x inference time

### Files to Create
```
experiments/exp6_ensemble/
├── hard_voting/
├── soft_voting/
├── stacking/
├── weighted_stacking/
├── ensemble_model.py
└── comparison_report.md
```

---

## 📈 EKSPERIMEN 7: MODEL INTERPRETABILITY

### Tujuan
Memahami bagaimana model membuat keputusan menggunakan teknik visualisasi dan interpretability.

### Metodologi

**Techniques:**

**1. Grad-CAM (Gradient-weighted Class Activation Mapping)**
```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Initialize Grad-CAM
target_layer = model.layer4[-1]  # For ResNet
cam = GradCAM(model=model, target_layers=[target_layer])

# Generate heatmap
grayscale_cam = cam(input_tensor=image, targets=None)

# Visualize
visualization = show_cam_on_image(original_image, grayscale_cam[0, :], use_rgb=True)
plt.imshow(visualization)
plt.title(f"Grad-CAM: Predicted as {predicted_class}")
plt.show()
```

**2. SHAP (SHapley Additive exPlanations)**
```python
import shap

# Create explainer
explainer = shap.DeepExplainer(model, background_data)

# Explain predictions
shap_values = explainer.shap_values(test_images)

# Visualize
shap.image_plot(shap_values, test_images)
```

**3. Integrated Gradients**
```python
from captum.attr import IntegratedGradients

ig = IntegratedGradients(model)
attributions = ig.attribute(image, target=predicted_class, n_steps=50)

# Visualize
visualize_image_attr(attributions, original_image, method='heat_map')
```

**4. Attention Maps (if using attention-based model)**
```python
# Extract attention weights
attention_weights = model.get_attention_weights(image)

# Visualize
plt.imshow(attention_weights, cmap='hot')
plt.colorbar()
plt.title('Attention Map')
plt.show()
```

### Analysis Questions
1. Apakah model fokus pada bakterinya atau background?
2. Fitur apa yang membedakan Gram+ vs Gram-?
3. Apakah ada bias sistematis dalam prediksi?
4. Kasus apa yang paling sulit diidentifikasi?

### Expected Insights
- Model should focus on bacterial cell walls (where Gram staining differs)
- Gram+ bacteria: Thicker peptidoglycan layer → different appearance
- Gram- bacteria: Thinner layer, outer membrane
- Misclassifications: Overlapping bacteria, poor image quality

### Files to Create
```
experiments/exp7_interpretability/
├── gradcam/
│   ├── gram_positive_samples/
│   └── gram_negative_samples/
├── shap/
├── integrated_gradients/
├── attention_maps/
└── analysis_report.md
```

---

## 🎯 EKSPERIMEN FINAL: MODEL OPTIMIZATION & DEPLOYMENT

### Tujuan
Mengoptimalkan model terbaik untuk deployment dengan fokus pada inference speed dan model size.

### Techniques

**1. Model Quantization**
```python
import torch.quantization

# Post-training quantization
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Measure improvement
original_size = get_model_size(model)
quantized_size = get_model_size(model_quantized)
print(f"Size reduction: {100 * (1 - quantized_size/original_size):.1f}%")
```

**2. Pruning**
```python
import torch.nn.utils.prune as prune

# Prune 30% of connections
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.3)

# Remove pruning reparametrization
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.remove(module, 'weight')
```

**3. Knowledge Distillation**
```python
# Teacher: Large model (EfficientNet-B7)
teacher_model = GramEfficientNet('efficientnet_b7')

# Student: Small model (MobileNetV2)
student_model = GramMobileNet('mobilenet_v2')

# Distillation loss
def distillation_loss(student_logits, teacher_logits, labels, T=3.0, alpha=0.5):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    
    hard_loss = F.cross_entropy(student_logits, labels)
    
    return alpha * soft_loss + (1 - alpha) * hard_loss

# Train student
for x, y in train_loader:
    with torch.no_grad():
        teacher_logits = teacher_model(x)
    
    student_logits = student_model(x)
    loss = distillation_loss(student_logits, teacher_logits, y)
    loss.backward()
    optimizer.step()
```

**4. ONNX Export**
```python
# Export to ONNX
dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    model,
    dummy_input,
    "gram_classifier.onnx",
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# Load and run with ONNX Runtime
import onnxruntime as ort

session = ort.InferenceSession("gram_classifier.onnx")
output = session.run(None, {'input': input_data.numpy()})
```

**5. TensorRT Optimization** (For NVIDIA GPUs)
```python
import torch_tensorrt

# Compile with TensorRT
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input((1, 3, 640, 640))],
    enabled_precisions={torch.float16}
)

# Benchmark
trt_inference_time = benchmark(trt_model, test_images)
pytorch_inference_time = benchmark(model, test_images)

print(f"Speedup: {pytorch_inference_time / trt_inference_time:.2f}x")
```

### Optimization Targets

| Metric | Original | Target | Achieved |
|--------|----------|--------|----------|
| Accuracy | 89% | ≥88% | ? |
| Model Size | 50MB | <20MB | ? |
| Inference Time (CPU) | 200ms | <100ms | ? |
| Inference Time (GPU) | 10ms | <5ms | ? |
| Memory Usage | 500MB | <200MB | ? |

### Deployment Options

**1. FastAPI Backend**
```python
from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load image
    image = Image.open(file.file)
    
    # Preprocess
    image_tensor = preprocess(image)
    
    # Predict
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0))
        prob = F.softmax(logits, dim=1)
        pred_class = torch.argmax(prob, dim=1).item()
    
    return {
        "prediction": "Gram Positive" if pred_class == 0 else "Gram Negative",
        "confidence": float(prob[0, pred_class])
    }
```

**2. Streamlit Web App**
```python
import streamlit as st

st.title("Gram Bacteria Classifier")

uploaded_file = st.file_uploader("Upload bacteria image", type=['jpg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')
    
    if st.button('Classify'):
        with st.spinner('Analyzing...'):
            prediction, confidence = predict(image)
        
        st.success(f"Prediction: {prediction}")
        st.info(f"Confidence: {confidence:.2%}")
```

**3. Docker Container**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📊 SUMMARY OF ALL EXPERIMENTS

### Experiment Execution Order

```
Week 1-2:  Exp 1 - Baseline Model
Week 3-4:  Exp 2 - Transfer Learning Comparison
Week 5:    Exp 3 - Hyperparameter Tuning
Week 6:    Exp 4 - Data Augmentation
Week 7:    Exp 5 - Class Imbalance (if needed)
Week 8:    Exp 6 - Ensemble Methods
Week 9:    Exp 7 - Model Interpretability
Week 10-11: Final Optimization & Deployment
Week 12:   Testing, Documentation, Report
```

### Expected Final Results

**Best Single Model:**
- Architecture: EfficientNet-B3 (optimized)
- Validation Accuracy: 89-91%
- Test Accuracy: 88-90%
- Precision: 88-90%
- Recall: 88-90%
- F1-Score: 88-90%
- AUC-ROC: 0.95+

**Best Ensemble:**
- Models: EfficientNet-B3 + ResNet50 + EfficientNet-B0
- Test Accuracy: 90-92%
- Inference Time: 30-50ms (3x single model)

**Deployed Model:**
- Model: EfficientNet-B0 (quantized)
- Test Accuracy: 87-89%
- Model Size: <15MB
- Inference Time (CPU): <100ms
- Inference Time (GPU): <5ms

---

## 📝 REPORTING TEMPLATE

### For Each Experiment, Create:

**1. README.md**
```markdown
# Experiment X: [Name]

## Objective
[What are we trying to achieve?]

## Hypothesis
[What do we expect to happen?]

## Methodology
[How did we conduct the experiment?]

## Results
[What actually happened?]

## Analysis
[Why did we get these results?]

## Conclusion
[What did we learn?]

## Next Steps
[What should we do next?]
```

**2. metrics.json**
```json
{
  "experiment_name": "exp2_efficientnet_b3",
  "model": "EfficientNet-B3",
  "hyperparameters": {
    "learning_rate": 0.0001,
    "batch_size": 16,
    "epochs": 100
  },
  "results": {
    "train_acc": 0.95,
    "val_acc": 0.89,
    "test_acc": 0.88,
    "precision": 0.87,
    "recall": 0.89,
    "f1_score": 0.88,
    "auc_roc": 0.94
  },
  "training_time_hours": 6.5,
  "inference_time_ms": 15,
  "model_size_mb": 47
}
```

**3. Visualizations**
- Training curves (loss & accuracy)
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Sample predictions
- Error analysis

---

**🎓 Good Luck with Your Thesis!**

