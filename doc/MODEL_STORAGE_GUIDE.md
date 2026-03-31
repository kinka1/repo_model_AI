# 📁 LOKASI PENYIMPANAN MODEL

## 🎯 Lokasi Utama

Model disimpan di folder:
```
d:\kerja\dataset\experiments\exp1_baseline\
```

---

## 📦 FILE YANG DISIMPAN

### **1. best_model.pth** ⭐ (PALING PENTING)
**Path**: `experiments/exp1_baseline/best_model.pth`

**Isi**:
```python
{
    'epoch': 42,                    # Epoch terbaik
    'model_state_dict': {...},      # Weights model
    'optimizer_state_dict': {...},  # State optimizer
    'val_acc': 0.9345,             # Akurasi validasi
    'val_f1': 0.8756               # F1-score
}
```

**Kapan disimpan**: Setiap kali validation accuracy meningkat  
**Ukuran**: ~100 MB  
**Gunakan ini untuk**: Deployment, testing, inference

---

### **2. final_model.pth**
**Path**: `experiments/exp1_baseline/final_model.pth`

**Isi**: Model weights setelah 50 epoch selesai  
**Ukuran**: ~100 MB  
**Gunakan ini untuk**: Jika ingin coba model di epoch terakhir

---

### **3. training_history.json** 📊
**Path**: `experiments/exp1_baseline/training_history.json`

**Isi**:
```json
{
    "train_loss": [1.1062, 0.3042, 0.2957, ...],
    "val_loss": [0.2038, 0.1882, 0.2228, ...],
    "train_acc": [0.8552, 0.8770, 0.8667, ...],
    "val_acc": [0.9115, 0.9200, 0.8884, ...],
    "train_f1": [0.7274, 0.7679, 0.7547, ...],
    "val_f1": [0.8230, 0.8229, 0.7911, ...],
    "val_precision": [0.7464, 0.8168, 0.6818, ...],
    "val_recall": [0.9171, 0.8291, 0.9422, ...]
}
```

**Ukuran**: ~5 KB  
**Gunakan ini untuk**: Analisis performa, plot grafik

---

### **4. training_history.png** 📈
**Path**: `experiments/exp1_baseline/training_history.png`

**Isi**: 4 grafik training curves
- Loss curve
- Accuracy curve  
- F1-score curve
- Precision & Recall

**Ukuran**: ~200 KB  
**Gunakan ini untuk**: Thesis, presentasi, laporan

---

## 🔍 CARA MENGECEK FILE

### **Cek apakah file sudah ada:**
```powershell
Get-ChildItem experiments\exp1_baseline
```

**Output yang diharapkan:**
```
Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----         11/5/2025  11:30 AM      103847392 best_model.pth
-a----         11/5/2025  11:55 AM      103847392 final_model.pth
-a----         11/5/2025  11:55 AM           4892 training_history.json
-a----         11/5/2025  11:55 AM         187456 training_history.png
```

### **Cek ukuran folder:**
```powershell
Get-ChildItem experiments\exp1_baseline -Recurse | Measure-Object -Property Length -Sum
```

### **Buka folder di Explorer:**
```powershell
explorer experiments\exp1_baseline
```

---

## 💾 CARA LOAD MODEL

### **1. Load Best Model** (Recommended)

```python
import torch
from pathlib import Path

# Path to model
model_path = Path('experiments/exp1_baseline/best_model.pth')

# Load checkpoint
checkpoint = torch.load(model_path)

# Check info
print(f"Best epoch: {checkpoint['epoch']}")
print(f"Val Accuracy: {checkpoint['val_acc']:.4f}")
print(f"Val F1-Score: {checkpoint['val_f1']:.4f}")

# Create model
from your_model import BaselineCNN
model = BaselineCNN(num_classes=2)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set to evaluation mode

print("✅ Model loaded successfully!")
```

### **2. Load Final Model** (Alternative)

```python
import torch

# Load just the weights
model_path = 'experiments/exp1_baseline/final_model.pth'
model.load_state_dict(torch.load(model_path))
model.eval()
```

---

## 🚀 CARA MENGGUNAKAN MODEL UNTUK PREDIKSI

### **Script Lengkap:**

```python
import torch
from torchvision import transforms
from PIL import Image
import sys
sys.path.append('.')
from pathlib import Path

# Define model architecture (same as training)
class BaselineCNN(torch.nn.Module):
    # ... (copy from 03_train_baseline.py)
    pass

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BaselineCNN(num_classes=2).to(device)

checkpoint = torch.load('experiments/exp1_baseline/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Load and predict
def predict_image(image_path):
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # Map to label
    class_names = ['Gram Negative', 'Gram Positive']
    confidence = probabilities[0][predicted_class].item()
    
    print(f"Prediction: {class_names[predicted_class]}")
    print(f"Confidence: {confidence*100:.2f}%")
    
    return predicted_class, confidence

# Example usage
image_path = 'data/processed/test/gram_positive/000001_2_3_100.jpg'
predict_image(image_path)
```

---

## 📊 CARA MELIHAT TRAINING HISTORY

### **1. View JSON:**
```powershell
Get-Content experiments\exp1_baseline\training_history.json | ConvertFrom-Json | Format-List
```

### **2. View Image:**
```powershell
start experiments\exp1_baseline\training_history.png
```

### **3. Programmatically:**
```python
import json

with open('experiments/exp1_baseline/training_history.json', 'r') as f:
    history = json.load(f)

# Print summary
print(f"Total epochs: {len(history['train_loss'])}")
print(f"Final train acc: {history['train_acc'][-1]:.4f}")
print(f"Final val acc: {history['val_acc'][-1]:.4f}")
print(f"Best val acc: {max(history['val_acc']):.4f}")
```

---

## 💾 BACKUP MODEL

### **Copy ke folder backup:**
```powershell
# Create backup folder
New-Item -ItemType Directory -Path "backup_models" -Force

# Copy best model
Copy-Item "experiments\exp1_baseline\best_model.pth" "backup_models\exp1_baseline_best.pth"

# Copy history
Copy-Item "experiments\exp1_baseline\training_history.json" "backup_models\exp1_baseline_history.json"
```

### **Compress untuk archive:**
```powershell
Compress-Archive -Path "experiments\exp1_baseline" -DestinationPath "exp1_baseline_backup.zip"
```

---

## 🔄 STRUKTUR FOLDER LENGKAP

```
d:\kerja\dataset\
│
├── experiments/                    # ← MODEL DISIMPAN DI SINI
│   └── exp1_baseline/             # ← EXPERIMENT 1
│       ├── best_model.pth         # ← MODEL TERBAIK ⭐
│       ├── final_model.pth        # ← MODEL AKHIR
│       ├── training_history.json  # ← METRICS
│       └── training_history.png   # ← GRAFIK
│
├── data/
│   ├── processed/                 # Data untuk training
│   └── annotations/               # CSV annotations
│
├── reports/                       # Visualisasi data exploration
│
├── 03_train_baseline.py           # Script training
└── ... (other files)
```

---

## 📏 UKURAN FILE

| File | Ukuran | Deskripsi |
|------|--------|-----------|
| `best_model.pth` | ~100 MB | Full checkpoint (model + optimizer) |
| `final_model.pth` | ~100 MB | Model weights only |
| `training_history.json` | ~5 KB | All metrics |
| `training_history.png` | ~200 KB | Plots |
| **Total** | **~200 MB** | Per experiment |

**Note**: Dengan 7 experiments, total storage ~1.4 GB

---

## 🎓 UNTUK THESIS

### **File yang Harus Disimpan:**
1. ✅ `best_model.pth` - untuk deployment
2. ✅ `training_history.json` - untuk tabel hasil
3. ✅ `training_history.png` - untuk gambar di laporan
4. ✅ `confusion_matrix.png` - untuk evaluasi (dibuat nanti)

### **Cara Export untuk Laporan:**
```powershell
# Copy ke folder thesis
New-Item -ItemType Directory -Path "thesis_materials\exp1_baseline" -Force
Copy-Item "experiments\exp1_baseline\*.png" "thesis_materials\exp1_baseline\"
Copy-Item "experiments\exp1_baseline\*.json" "thesis_materials\exp1_baseline\"
```

---

## 🔐 KEAMANAN & VERSION CONTROL

### **Git (Jangan commit model!):**
Tambahkan ke `.gitignore`:
```
# Model files (too large)
experiments/*/best_model.pth
experiments/*/final_model.pth
*.pth

# But keep history
!experiments/*/training_history.json
!experiments/*/training_history.png
```

### **Cloud Backup (Recommended):**
- Google Drive / OneDrive
- Only upload `best_model.pth` + `training_history.json`
- ~100 MB per experiment

---

## 📝 MODEL INFO LENGKAP

Setiap model `best_model.pth` berisi:

```python
checkpoint = torch.load('experiments/exp1_baseline/best_model.pth')

# Available keys:
print(checkpoint.keys())
# dict_keys(['epoch', 'model_state_dict', 'optimizer_state_dict', 'val_acc', 'val_f1'])

# Model architecture info:
print(f"Best epoch: {checkpoint['epoch']}")
print(f"Validation Accuracy: {checkpoint['val_acc']}")
print(f"Validation F1-Score: {checkpoint['val_f1']}")

# Model weights:
for name, param in checkpoint['model_state_dict'].items():
    print(f"{name}: {param.shape}")
```

---

## 🚨 TROUBLESHOOTING

### **Problem**: File tidak ada
```powershell
# Check if training created the folder
Test-Path experiments\exp1_baseline

# If False, training hasn't started or crashed
```

### **Problem**: File corrupt / incomplete
```python
# Test if model can be loaded
import torch
try:
    checkpoint = torch.load('experiments/exp1_baseline/best_model.pth')
    print("✅ Model file is valid")
except Exception as e:
    print(f"❌ Error: {e}")
```

### **Problem**: Out of disk space
```powershell
# Check available space
Get-PSDrive C | Select-Object Used, Free

# Clean up if needed
Remove-Item experiments\exp1_baseline\final_model.pth  # Keep only best_model.pth
```

---

## ✅ QUICK REFERENCE

### **Model Location:**
```
experiments/exp1_baseline/best_model.pth
```

### **Load Model:**
```python
checkpoint = torch.load('experiments/exp1_baseline/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### **View Results:**
```powershell
start experiments\exp1_baseline\training_history.png
```

### **Check Files:**
```powershell
explorer experiments\exp1_baseline
```

---

## 🎉 SETELAH TRAINING SELESAI

1. **Verify files exist:**
   ```powershell
   Get-ChildItem experiments\exp1_baseline
   ```

2. **Check best accuracy:**
   ```python
   import torch
   checkpoint = torch.load('experiments/exp1_baseline/best_model.pth')
   print(f"Best accuracy: {checkpoint['val_acc']:.4f}")
   ```

3. **Backup important files:**
   ```powershell
   Copy-Item experiments\exp1_baseline\best_model.pth backup_models\
   ```

4. **Proceed to testing or next experiment!** 🚀

---

**Model Anda akan aman tersimpan di `experiments/exp1_baseline/`!** 💾
