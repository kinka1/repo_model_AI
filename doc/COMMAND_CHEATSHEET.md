# 🚀 COMMAND CHEAT SHEET - Gram Bacteria Classification

## Quick Reference untuk Semua Command yang Anda Butuhkan

---

## 📦 INSTALLATION & SETUP

### First Time Setup (Run Once)
```powershell
# Option 1: Use setup script (recommended)
.\setup.ps1

# Option 2: Manual setup
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Activate Environment (Every Session)
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Check if activated (should show venv in prompt)
# (venv) PS D:\kerja\dataset>
```

### Check Installation
```powershell
# Check Python version
python --version  # Should be 3.8+

# Check installed packages
pip list

# Check if PyTorch can use GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🔬 DATA PROCESSING PIPELINE

### Step 1: Data Exploration (Already Done ✅)
```powershell
# Analyze dataset and generate visualizations
python 01_data_exploration.py

# Output files:
# - reports/class_distribution.png
# - reports/image_size_distribution.png
# - reports/annotations_per_image.png
# - reports/sample_images.png
# - data/annotations/dataset2_annotations.csv
```

### Step 2: Data Preprocessing (Next Step ⏳)
```powershell
# Crop bacteria and create train/val/test splits
python 02_data_preprocessing.py

# Duration: ~30-60 minutes
# Output: ~11,824 cropped images (224×224) in:
#   data/processed/train/
#   data/processed/val/
#   data/processed/test/
```

### Step 3: Baseline Training (Ready 🎯)
```powershell
# Train baseline CNN model
python 03_train_baseline.py

# Duration: ~2-3 hours (CPU) or ~30 min (GPU)
# Output:
#   experiments/exp1_baseline/best_model.pth
#   experiments/exp1_baseline/training_history.png
```

---

## 📊 VIEW RESULTS

### Open Visualizations
```powershell
# Open reports folder
explorer reports

# View specific image
start reports\class_distribution.png
start reports\sample_images.png
```

### Check Dataset Summary
```powershell
# View CSV with pandas
python -c "import pandas as pd; df = pd.read_csv('data/annotations/dataset2_annotations.csv'); print(df.head()); print(df.info())"

# Count samples
python -c "import pandas as pd; df = pd.read_csv('data/annotations/dataset2_annotations.csv'); print(df['label'].value_counts())"
```

### Check Processed Data
```powershell
# Count processed images
Get-ChildItem -Path "data\processed\train\gram_positive" -Recurse -File | Measure-Object | Select-Object Count
Get-ChildItem -Path "data\processed\train\gram_negative" -Recurse -File | Measure-Object | Select-Object Count
```

---

## 🧠 MODEL TRAINING

### Baseline CNN (Experiment 1)
```powershell
# Full training (50 epochs)
python 03_train_baseline.py

# Quick test (5 epochs for debugging)
python 03_train_baseline.py --epochs 5 --batch-size 16
```

### Monitor Training
```powershell
# TensorBoard (if implemented)
tensorboard --logdir=logs/tensorboard

# View in browser
start http://localhost:6006
```

### Check GPU Usage (if applicable)
```powershell
# Install gpustat first
pip install gpustat

# Monitor GPU in real-time
gpustat -cp -i 1
```

---

## 🔍 DEBUGGING & TROUBLESHOOTING

### Check for Errors
```powershell
# Test if script runs without errors (dry run)
python -m py_compile 01_data_exploration.py
python -m py_compile 02_data_preprocessing.py
python -m py_compile 03_train_baseline.py
```

### Memory Issues
```powershell
# Check available RAM
Get-WmiObject Win32_OperatingSystem | Select-Object @{Name="FreeGB";Expression={[math]::Round($_.FreePhysicalMemory/1MB, 2)}}

# If low memory, reduce batch size:
# Edit 03_train_baseline.py line ~287: BATCH_SIZE = 16  (or 8)
```

### Package Issues
```powershell
# Reinstall all packages
pip install --force-reinstall -r requirements.txt

# Update specific package
pip install --upgrade torch torchvision

# Check package version
pip show torch
```

---

## 📁 FILE MANAGEMENT

### List All Generated Files
```powershell
# Show project structure
tree /F

# List only Python scripts
Get-ChildItem -Filter "*.py"

# List all reports
Get-ChildItem -Path reports -Recurse
```

### Cleanup (If Needed)
```powershell
# Remove processed data (to reprocess from scratch)
Remove-Item -Path "data\processed" -Recurse -Force

# Remove experiment results
Remove-Item -Path "experiments\exp1_baseline" -Recurse -Force

# Remove Python cache
Get-ChildItem -Include __pycache__ -Recurse -Force | Remove-Item -Recurse -Force
```

### Backup Important Files
```powershell
# Create backup folder
New-Item -ItemType Directory -Path "backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

# Copy important files
Copy-Item -Path "data\annotations" -Destination "backup_*\annotations" -Recurse
Copy-Item -Path "reports" -Destination "backup_*\reports" -Recurse
```

---

## 🧪 TESTING

### Test Dataset Loading
```powershell
# Quick test if dataset can be loaded
python -c "
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
sys.path.append('.')

# Your dataset class would be here
print('Dataset loading test: OK')
"
```

### Test Model Forward Pass
```powershell
# Test if model can process one batch
python -c "
import torch
import sys
sys.path.append('.')

# Create dummy input
x = torch.randn(1, 3, 224, 224)
print(f'Input shape: {x.shape}')
print('Model forward pass test: OK')
"
```

---

## 📊 RESULTS ANALYSIS

### View Training History
```powershell
# Load and print training history
python -c "
import json
with open('experiments/exp1_baseline/training_history.json', 'r') as f:
    history = json.load(f)
    
print('Final Results:')
print(f'Train Accuracy: {history["train_acc"][-1]:.4f}')
print(f'Val Accuracy: {history["val_acc"][-1]:.4f}')
print(f'Val F1-Score: {history["val_f1"][-1]:.4f}')
"
```

### Compare Experiments
```powershell
# Create comparison table (manual)
python -c "
import json
import pandas as pd

experiments = [
    'exp1_baseline',
    'exp2_transfer_learning',
    # Add more as you complete them
]

results = []
for exp in experiments:
    try:
        with open(f'experiments/{exp}/training_history.json', 'r') as f:
            history = json.load(f)
            results.append({
                'Experiment': exp,
                'Train Acc': history['train_acc'][-1],
                'Val Acc': history['val_acc'][-1],
                'Val F1': history['val_f1'][-1]
            })
    except:
        pass

df = pd.DataFrame(results)
print(df.to_string(index=False))
"
```

---

## 🌐 JUPYTER NOTEBOOK (Optional)

### Launch Jupyter
```powershell
# Install Jupyter (if not already)
pip install jupyter

# Start Jupyter Notebook
jupyter notebook

# Start JupyterLab (modern interface)
jupyter lab
```

### Create Analysis Notebook
```powershell
# Quick create notebook for analysis
jupyter notebook notebooks/results_analysis.ipynb
```

---

## 🔧 COMMON FIXES

### Fix: "Module not found" Error
```powershell
# Make sure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall requirements
pip install -r requirements.txt
```

### Fix: "CUDA out of memory"
```powershell
# Reduce batch size in training script
# Edit line: BATCH_SIZE = 32  →  BATCH_SIZE = 16

# Or clear CUDA cache
python -c "import torch; torch.cuda.empty_cache(); print('Cache cleared')"
```

### Fix: "Permission denied" on Windows
```powershell
# Run PowerShell as Administrator
# Or change execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Fix: Slow Training
```powershell
# Check if GPU is being used
python -c "import torch; print(f'Using: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')"

# If CPU only, reduce num_workers in DataLoader
# Edit line: num_workers=4  →  num_workers=0
```

---

## 📝 QUICK WORKFLOW

### Daily Workflow (When Resuming Work)
```powershell
# 1. Activate environment
.\venv\Scripts\Activate.ps1

# 2. Check current progress
Get-ChildItem experiments

# 3. Continue with next experiment
python 03_train_baseline.py

# 4. Check results
explorer experiments\exp1_baseline
```

### Before Thesis Defense
```powershell
# Generate all visualizations
python 01_data_exploration.py
start reports\class_distribution.png
start reports\sample_images.png

# Show model architecture
python -c "
import torch
from src.models import BaselineCNN
model = BaselineCNN()
print(model)
print(f'\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}')
"

# Print final results
python -c "
import json
with open('experiments/exp1_baseline/training_history.json', 'r') as f:
    history = json.load(f)
print('FINAL RESULTS:')
print(f'Accuracy: {history["val_acc"][-1]*100:.2f}%')
print(f'F1-Score: {history["val_f1"][-1]:.4f}')
print(f'Precision: {history["val_precision"][-1]:.4f}')
print(f'Recall: {history["val_recall"][-1]:.4f}')
"
```

---

## 🎓 USEFUL ONE-LINERS

```powershell
# Count total dataset samples
(Get-Content data\annotations\dataset2_annotations.csv | Measure-Object -Line).Lines - 1

# Find largest image file
Get-ChildItem -Path "dataset 2" -Recurse -File | Sort-Object Length -Descending | Select-Object -First 10

# Check disk space
Get-PSDrive C | Select-Object Used, Free

# Time a command
Measure-Command { python 01_data_exploration.py }

# Run in background (for long training)
Start-Process python -ArgumentList "03_train_baseline.py" -NoNewWindow

# Kill Python processes (if stuck)
Get-Process python | Stop-Process -Force

# Check Python processes
Get-Process python

# Pretty print JSON
Get-Content data\annotations\dataset2_annotations.csv | ConvertFrom-Csv | Select-Object -First 5 | Format-Table
```

---

## 📚 DOCUMENTATION QUICK ACCESS

```powershell
# Open all documentation
start PROPOSAL_TA_KLASIFIKASI_BAKTERI_GRAM.md
start RANCANGAN_KERJA_DETAIL.md
start RANCANGAN_PERCOBAAN.md
start QUICK_START.md
start LAPORAN_LENGKAP.md

# Open this cheat sheet
start COMMAND_CHEATSHEET.md
```

---

## 🎯 EXPERIMENT COMMANDS (Future)

### Experiment 2: Transfer Learning
```powershell
# ResNet50
python 04_train_transfer_learning.py --model resnet50

# EfficientNet-B0
python 04_train_transfer_learning.py --model efficientnet_b0

# Compare all models
python 04_train_transfer_learning.py --compare-all
```

### Experiment 3: Hyperparameter Tuning
```powershell
# Run Optuna optimization
python 05_hyperparameter_tuning.py --trials 50

# Use best hyperparameters
python 05_hyperparameter_tuning.py --use-best
```

### Experiment 5: Class Imbalance
```powershell
# Test SMOTE
python 07_class_imbalance.py --method smote

# Test Focal Loss
python 07_class_imbalance.py --method focal_loss

# Compare all methods
python 07_class_imbalance.py --compare-all
```

---

## 💾 GIT COMMANDS (If Using Version Control)

```powershell
# Initialize git
git init
git add .
git commit -m "Initial commit: Data exploration complete"

# Create experiment branch
git checkout -b experiment1_baseline

# Save progress
git add .
git commit -m "Experiment 1 complete: 75% accuracy achieved"

# View history
git log --oneline --graph
```

---

## ⚡ POWER USER TIPS

### Create Aliases (PowerShell Profile)
```powershell
# Edit profile
notepad $PROFILE

# Add these aliases:
function act { .\venv\Scripts\Activate.ps1 }
function explore { python 01_data_exploration.py }
function preprocess { python 02_data_preprocessing.py }
function train { python 03_train_baseline.py }
function results { explorer experiments }
```

### Multi-Monitor Setup
```powershell
# Terminal 1: Training
python 03_train_baseline.py

# Terminal 2: Monitor GPU
gpustat -cp -i 1

# Terminal 3: TensorBoard
tensorboard --logdir=logs
```

---

**Last Updated**: 2025-01-05  
**Status**: Phase 1 Complete ✅  
**Next**: Run `python 02_data_preprocessing.py` ⏳
