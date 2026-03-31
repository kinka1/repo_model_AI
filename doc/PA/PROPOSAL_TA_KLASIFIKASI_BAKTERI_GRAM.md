# PROPOSAL TUGAS AKHIR
## KLASIFIKASI CITRA MIKROSKOPIS BAKTERI GRAM POSITIF DAN GRAM NEGATIF MENGGUNAKAN DEEP LEARNING

---

## 1. LATAR BELAKANG

Pewarnaan Gram merupakan metode fundamental dalam mikrobiologi untuk mengklasifikasikan bakteri menjadi dua kategori utama: Gram positif (G+) dan Gram negatif (G-). Klasifikasi ini sangat penting dalam diagnosis medis, pemilihan antibiotik, dan penelitian mikrobiologi. Namun, identifikasi manual melalui mikroskop memerlukan keahlian khusus, waktu yang lama, dan rentan terhadap kesalahan manusia.

Perkembangan teknologi deep learning, khususnya Convolutional Neural Networks (CNN), telah menunjukkan performa yang sangat baik dalam klasifikasi citra medis. Penelitian ini bertujuan untuk mengembangkan sistem otomatis yang dapat mengklasifikasikan bakteri Gram positif dan negatif dari citra mikroskopis dengan akurasi tinggi.

---

## 2. RUMUSAN MASALAH

1. Bagaimana membangun model deep learning yang efektif untuk klasifikasi bakteri Gram positif dan Gram negatif?
2. Bagaimana performa model dalam hal akurasi, presisi, recall, dan F1-score?
3. Arsitektur model apa yang paling optimal untuk kasus klasifikasi bakteri ini?
4. Bagaimana cara menangani ketidakseimbangan data dan variasi citra mikroskopis?

---

## 3. TUJUAN PENELITIAN

1. Mengembangkan model AI berbasis deep learning untuk klasifikasi bakteri Gram positif dan Gram negatif
2. Mencapai akurasi klasifikasi minimal 90%
3. Membandingkan performa berbagai arsitektur CNN (ResNet, EfficientNet, VGG, dll)
4. Mengimplementasikan sistem yang dapat digunakan untuk membantu tenaga medis dalam diagnosis

---

## 4. BATASAN MASALAH

1. Dataset yang digunakan berasal dari dua sumber:
   - Dataset 1: PBCs Microorganism Annotation (format JSON COCO)
   - Dataset 2: 640 DataSet (format LabelMe JSON dengan label G+/G-)
2. Fokus pada klasifikasi binary: Gram positif vs Gram negatif
3. Input berupa citra mikroskopis dengan resolusi 640x640 pixels
4. Implementasi menggunakan Python dengan framework PyTorch/TensorFlow
5. Evaluasi dilakukan menggunakan train-validation-test split (70:15:15)

---

## 5. METODOLOGI PENELITIAN

### 5.1 Alur Penelitian

```
┌─────────────────────────┐
│  Pengumpulan Dataset    │
│  - Dataset 1            │
│  - Dataset 2            │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Preprocessing Data     │
│  - Parsing Annotations  │
│  - Label Extraction     │
│  - Data Cleaning        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Eksplorasi Data (EDA)  │
│  - Distribusi Kelas     │
│  - Visualisasi          │
│  - Statistik Data       │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Data Augmentation      │
│  - Rotation             │
│  - Flip                 │
│  - Color Jittering      │
│  - Zoom                 │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Pembuatan Model        │
│  - ResNet50/101         │
│  - EfficientNet B0-B7   │
│  - VGG16/19             │
│  - Custom CNN           │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Training & Validation  │
│  - Train: 70%           │
│  - Validation: 15%      │
│  - Test: 15%            │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Evaluasi Model         │
│  - Accuracy             │
│  - Precision/Recall     │
│  - F1-Score             │
│  - Confusion Matrix     │
│  - ROC-AUC              │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Model Deployment       │
│  - Web Application      │
│  - REST API             │
└─────────────────────────┘
```

### 5.2 Tools dan Library

- **Programming Language**: Python 3.8+
- **Deep Learning Framework**: PyTorch / TensorFlow 2.x
- **Data Processing**: NumPy, Pandas, OpenCV
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Model Training**: PyTorch Lightning / Keras
- **Hyperparameter Tuning**: Optuna / Ray Tune
- **Deployment**: Flask/FastAPI, Streamlit
- **Version Control**: Git, DVC (Data Version Control)

---

## 6. SPESIFIKASI DATASET

### Dataset 1: PBCs Microorganism Annotation
- **Format**: COCO JSON
- **Jumlah gambar**: ~5000+ images
- **Resolusi**: 5472 x 3648 pixels
- **Anotasi**: 3 annotator (annotator1, annotator2, DoubleCheck)
- **Label**: Mikroorganisme dengan bounding box/polygon

### Dataset 2: 640 DataSet
- **Format**: LabelMe JSON
- **Jumlah gambar**: ~3000+ images  
- **Resolusi**: 640 x 640 pixels
- **Label**: "G+" (Gram Positif) dan "G-" (Gram Negatif)
- **Anotasi**: Polygon dengan koordinat titik-titik

**Total Dataset Gabungan**: ~8000+ citra mikroskopis

---

## 7. RANCANGAN ARSITEKTUR MODEL

### 7.1 Baseline Model: Simple CNN
```
Input (640x640x3)
    │
    ▼
Conv2D (32 filters, 3x3) → ReLU → MaxPool (2x2)
    │
    ▼
Conv2D (64 filters, 3x3) → ReLU → MaxPool (2x2)
    │
    ▼
Conv2D (128 filters, 3x3) → ReLU → MaxPool (2x2)
    │
    ▼
Flatten
    │
    ▼
Dense (256) → ReLU → Dropout (0.5)
    │
    ▼
Dense (2) → Softmax
    │
    ▼
Output: [Gram+, Gram-]
```

### 7.2 Transfer Learning Models

**Model Kandidat:**
1. **ResNet50/101**
   - Pre-trained pada ImageNet
   - Fine-tuning pada dataset bakteri
   - Residual connections membantu gradient flow

2. **EfficientNetB0-B7**
   - Skalabilitas optimal
   - Compound scaling
   - Efisien secara komputasi

3. **VGG16/19**
   - Arsitektur sederhana
   - Deep feature extraction
   - Baseline comparison

4. **MobileNetV2**
   - Lightweight model
   - Untuk deployment mobile/edge

### 7.3 Ensemble Model
- Voting dari 3 model terbaik
- Weighted averaging
- Stacking dengan meta-learner

---

## 8. PREPROCESSING DAN AUGMENTASI DATA

### 8.1 Preprocessing Pipeline
```python
1. Load Image & Annotation
   ├── Parse JSON (Dataset 1: COCO, Dataset 2: LabelMe)
   ├── Extract bounding box/polygon
   └── Crop bacterial regions

2. Image Preprocessing
   ├── Resize to 640x640 (if needed)
   ├── Normalize pixel values [0-1]
   ├── Standardize (mean=0, std=1)
   └── Convert to tensor

3. Label Processing
   ├── Map "G+" → 0 (Gram Positive)
   ├── Map "G-" → 1 (Gram Negative)
   └── One-hot encoding (optional)
```

### 8.2 Data Augmentation
```python
Training Augmentation:
- RandomRotation(30°)
- RandomHorizontalFlip(p=0.5)
- RandomVerticalFlip(p=0.5)
- ColorJitter(brightness=0.2, contrast=0.2)
- RandomResizedCrop(640, scale=(0.8, 1.0))
- GaussianBlur(kernel_size=5, p=0.3)
- RandomAffine(degrees=15, translate=(0.1, 0.1))

Validation/Test:
- Resize to 640x640
- Normalization only
```

---

## 9. RANCANGAN EKSPERIMEN

### Eksperimen 1: Baseline CNN
- **Tujuan**: Mengetahui performa model sederhana
- **Model**: Custom CNN 3-layer
- **Hyperparameter**:
  - Learning rate: 0.001
  - Batch size: 32
  - Epochs: 50
  - Optimizer: Adam
  - Loss: CrossEntropyLoss

### Eksperimen 2: Transfer Learning Comparison
- **Tujuan**: Membandingkan pre-trained models
- **Models**: ResNet50, EfficientNetB0, VGG16, MobileNetV2
- **Strategy**: Fine-tuning seluruh layer
- **Hyperparameter**:
  - Learning rate: 0.0001 (dengan scheduler)
  - Batch size: 16
  - Epochs: 100
  - Optimizer: AdamW
  - Loss: CrossEntropyLoss dengan Label Smoothing

### Eksperimen 3: Hyperparameter Tuning
- **Tujuan**: Optimasi model terbaik dari Exp 2
- **Tuning Parameters**:
  - Learning rate: [1e-5, 1e-4, 1e-3]
  - Batch size: [8, 16, 32]
  - Dropout rate: [0.3, 0.5, 0.7]
  - Weight decay: [1e-5, 1e-4]
- **Method**: Optuna dengan 50 trials

### Eksperimen 4: Data Augmentation Impact
- **Tujuan**: Mengukur pengaruh augmentasi
- **Variants**:
  - No augmentation
  - Basic augmentation (flip, rotate)
  - Advanced augmentation (+ color jitter, blur)
  - Heavy augmentation (+ mixup, cutout)

### Eksperimen 5: Class Imbalance Handling
- **Tujuan**: Menangani distribusi kelas yang tidak seimbang
- **Methods**:
  - Weighted Loss Function
  - Oversampling (SMOTE untuk images)
  - Undersampling
  - Focal Loss
  - Class-balanced sampling

### Eksperimen 6: Ensemble Methods
- **Tujuan**: Meningkatkan akurasi dengan ensemble
- **Strategies**:
  - Hard voting (3 best models)
  - Soft voting (weighted average)
  - Stacking dengan Logistic Regression

---

## 10. METRIK EVALUASI

### 10.1 Confusion Matrix
```
                Predicted
                G+    G-
Actual   G+    [TP]  [FN]
         G-    [FP]  [TN]
```

### 10.2 Metrics
1. **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
2. **Precision**: TP / (TP + FP)
3. **Recall (Sensitivity)**: TP / (TP + FN)
4. **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
5. **Specificity**: TN / (TN + FP)
6. **ROC-AUC**: Area Under Receiver Operating Characteristic Curve
7. **Cohen's Kappa**: Inter-rater agreement

### 10.3 Target Performance
- **Accuracy**: ≥ 90%
- **Precision**: ≥ 88%
- **Recall**: ≥ 88%
- **F1-Score**: ≥ 88%
- **ROC-AUC**: ≥ 0.95

---

## 11. RANCANGAN IMPLEMENTASI

### 11.1 Struktur Direktori
```
gram-bacteria-classification/
│
├── data/
│   ├── raw/
│   │   ├── dataset_1/
│   │   └── dataset_2/
│   ├── processed/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── annotations/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_transfer_learning.ipynb
│   └── 05_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── augmentation.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_cnn.py
│   │   ├── resnet.py
│   │   ├── efficientnet.py
│   │   └── ensemble.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── callbacks.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── visualize.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
│
├── configs/
│   ├── config.yaml
│   └── hyperparameters.yaml
│
├── experiments/
│   ├── exp1_baseline/
│   ├── exp2_transfer_learning/
│   └── exp3_hyperparameter_tuning/
│
├── models/
│   ├── checkpoints/
│   └── final_model.pth
│
├── app/
│   ├── api.py
│   ├── streamlit_app.py
│   └── templates/
│
├── tests/
│   └── test_models.py
│
├── requirements.txt
├── setup.py
└── README.md
```

### 11.2 Tech Stack
```yaml
Framework:
  - PyTorch 2.0+
  - PyTorch Lightning
  - TorchVision

Data Processing:
  - NumPy
  - Pandas
  - OpenCV (cv2)
  - Pillow
  - Albumentations

Visualization:
  - Matplotlib
  - Seaborn
  - Plotly
  - TensorBoard

Experiment Tracking:
  - MLflow
  - Weights & Biases

Hyperparameter Tuning:
  - Optuna

Deployment:
  - FastAPI
  - Streamlit
  - Docker

Testing:
  - PyTest
  - unittest
```

---

## 12. TIMELINE PENELITIAN

### Fase 1: Persiapan (Minggu 1-2)
- [x] Pengumpulan dataset
- [ ] Instalasi environment
- [ ] Setup version control
- [ ] Literature review

### Fase 2: Data Preparation (Minggu 3-4)
- [ ] Parsing annotations (COCO & LabelMe)
- [ ] Exploratory Data Analysis (EDA)
- [ ] Data cleaning & preprocessing
- [ ] Train-validation-test split
- [ ] Data augmentation pipeline

### Fase 3: Model Development (Minggu 5-8)
- [ ] Baseline CNN implementation
- [ ] Transfer learning models
- [ ] Training & validation
- [ ] Hyperparameter tuning
- [ ] Ensemble methods

### Fase 4: Evaluation (Minggu 9-10)
- [ ] Model evaluation pada test set
- [ ] Analisis confusion matrix
- [ ] Error analysis
- [ ] Model comparison
- [ ] Statistical significance testing

### Fase 5: Deployment (Minggu 11-12)
- [ ] Model optimization (quantization, pruning)
- [ ] API development (FastAPI)
- [ ] Web interface (Streamlit)
- [ ] Docker containerization
- [ ] Documentation

### Fase 6: Dokumentasi (Minggu 13-14)
- [ ] Penulisan laporan TA
- [ ] Pembuatan presentasi
- [ ] Persiapan demo
- [ ] Sidang TA

---

## 13. EXPECTED OUTCOMES

### 13.1 Deliverables
1. **Model AI**: Pre-trained model untuk klasifikasi bakteri Gram
2. **Source Code**: Repository lengkap di GitHub
3. **Dataset Terorganisir**: Annotated dataset dengan split yang jelas
4. **Web Application**: Interface untuk upload dan prediksi
5. **REST API**: Endpoint untuk integrasi dengan sistem lain
6. **Dokumentasi**: 
   - README lengkap
   - API documentation
   - Model card
7. **Laporan TA**: Dokumen lengkap penelitian
8. **Paper**: Draft paper untuk publikasi

### 13.2 Expected Results
- Model dengan akurasi ≥ 90% pada test set
- Inference time < 100ms per image
- Model size < 200MB (setelah optimization)
- API response time < 200ms

---

## 14. REFERENSI

1. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.
3. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. ICML.
4. Szegedy, C., et al. (2015). Rethinking the inception architecture for computer vision. CVPR.
5. Esteva, A., et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks. Nature.
6. Smith, K. P., et al. (2020). Automated interpretation of blood culture gram stains by use of a deep convolutional neural network. JCM.
7. Xie, S., et al. (2017). Aggregated residual transformations for deep neural networks. CVPR.
8. Howard, A. G., et al. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications.

---

## 15. LAMPIRAN

### A. Analisis Dataset Awal

**Dataset 1 (PBCs):**
- Format: COCO JSON
- Total images: ~5000+
- Resolution: 5472×3648
- Annotators: 3 (perlu consensus)

**Dataset 2 (640DataSet):**
- Format: LabelMe JSON
- Total images: ~3000+
- Resolution: 640×640
- Labels: "G+" dan "G-" sudah tersedia
- Sudah siap untuk training

### B. Hardware Requirements

**Minimum:**
- CPU: Intel i5/AMD Ryzen 5
- RAM: 16GB
- GPU: NVIDIA GTX 1660 Ti (6GB VRAM)
- Storage: 100GB SSD

**Recommended:**
- CPU: Intel i7/AMD Ryzen 7
- RAM: 32GB
- GPU: NVIDIA RTX 3070/3080 (8-10GB VRAM)
- Storage: 500GB NVMe SSD

**Cloud Alternative:**
- Google Colab Pro/Pro+ (T4/V100 GPU)
- Kaggle Notebooks (P100 GPU)
- AWS SageMaker
- Azure ML

---

## AUTHOR & SUPERVISOR

**Mahasiswa:**
- Nama: [Nama Anda]
- NIM: [NIM Anda]
- Program Studi: [Prodi Anda]
- Email: [Email Anda]

**Pembimbing:**
- Pembimbing 1: [Nama Dosen 1]
- Pembimbing 2: [Nama Dosen 2]

---

**Tanggal Proposal:** November 2025
**Status:** Draft v1.0
