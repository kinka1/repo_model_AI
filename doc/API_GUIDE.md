# 🔬 Gram Bacteria Classification API Guide

API berbasis FastAPI untuk klasifikasi bakteri Gram-positif (G+) dan Gram-negatif (G-) dari gambar mikroskopis.

## 📋 Daftar Isi

- [Quick Start](#-quick-start)
- [Instalasi](#-instalasi)
- [Menjalankan Server](#-menjalankan-server)
- [Endpoints](#-endpoints)
- [Contoh Penggunaan](#-contoh-penggunaan)
- [Testing dengan cURL](#-testing-dengan-curl)
- [Testing dengan Python](#-testing-dengan-python)
- [Deployment](#-deployment)

---

## 🚀 Quick Start

```powershell
# 1. Install dependencies
pip install -r requirements_api.txt

# 2. Jalankan server
uvicorn app.main:app --reload

# 3. Buka browser
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

---

## 📦 Instalasi

### Prerequisites

Pastikan sudah terinstall:
- Python 3.8+
- PyTorch dengan CUDA support (jika menggunakan GPU)
- Model sudah trained (default): `experiments/scenario_4a_resnet50/best_model.pth`

### Install Dependencies

```powershell
# Install API dependencies
pip install -r requirements_api.txt

# Atau install manual
pip install fastapi uvicorn python-multipart pillow pydantic
```

### Verifikasi Instalasi

```powershell
# Check FastAPI version
python -c "import fastapi; print(fastapi.__version__)"

# Check if model exists
python -c "from pathlib import Path; print('Model found' if Path('experiments/scenario_4a_resnet50/best_model.pth').exists() else 'Model NOT found')"
```

---

## 🎯 Menjalankan Server

### Development Mode (dengan auto-reload)

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Optional: Gunakan checkpoint lain

```powershell
$env:GRAM_MODEL_PATH="experiments/exp3a_efficientnet_b0/best_model.pth"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Catatan: implementasi API saat ini memuat arsitektur ResNet50 fine-tuning. Jika checkpoint berbeda arsitektur, model tidak akan ter-load (`/health` akan menunjukkan error).

### Production Mode

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Dengan Custom Port

```powershell
uvicorn app.main:app --reload --port 5000
```

### Server akan berjalan di:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📡 Endpoints

### 1. Root - API Info
```
GET /
```

**Response:**
```json
{
  "message": "🔬 Gram Bacteria Classification API",
  "version": "1.0.0",
  "docs": "/docs",
  "endpoints": {
    "predict": "POST /predict",
    "batch_predict": "POST /predict/batch",
    "model_info": "GET /model/info",
    "health_check": "GET /health"
  }
}
```

---

### 2. Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "version": "1.0.0"
}
```

---

### 3. Predict Single Image
```
POST /predict
```

**Parameters:**
- `file`: Image file (JPG, PNG, BMP, TIFF) - **Required**

**Response:**
```json
{
  "filename": "bacteria_sample.jpg",
  "prediction": "Gram Negative (G-)",
  "class_id": 0,
  "confidence": 0.9234,
  "probabilities": {
    "Gram Negative (G-)": 0.9234,
    "Gram Positive (G+)": 0.0766
  }
}
```

---

### 4. Predict Multiple Images (Batch)
```
POST /predict/batch
```

**Parameters:**
- `files`: List of image files - **Required**

**Response:**
```json
{
  "total_images": 3,
  "predictions": [
    {
      "filename": "sample1.jpg",
      "prediction": "Gram Negative (G-)",
      "class_id": 0,
      "confidence": 0.95,
      "probabilities": {
        "Gram Negative (G-)": 0.95,
        "Gram Positive (G+)": 0.05
      }
    },
    {
      "filename": "sample2.jpg",
      "prediction": "Gram Positive (G+)",
      "class_id": 1,
      "confidence": 0.88,
      "probabilities": {
        "Gram Negative (G-)": 0.12,
        "Gram Positive (G+)": 0.88
      }
    }
  ],
  "summary": {
    "Gram Negative (G-)": 2,
    "Gram Positive (G+)": 1,
    "average_confidence": 0.89
  }
}
```

---

### 5. Model Information
```
GET /model/info
```

**Response:**
```json
{
  "model_name": "BaselineCNN",
  "version": "1.0",
  "parameters": "25,784,578",
  "architecture": {
    "type": "Convolutional Neural Network",
    "layers": "3 conv blocks + 2 FC layers",
    "input_size": "224x224x3",
    "output_classes": 2,
    "class_labels": ["Gram Negative (G-)", "Gram Positive (G+)"]
  },
  "dataset": {
    "total_samples": 11824,
    "classes": 2,
    "class_distribution": {
      "Gram Negative (G-)": "77.5%",
      "Gram Positive (G+)": "22.5%"
    },
    "split": "70% train, 15% val, 15% test",
    "image_size": "224x224x3"
  },
  "performance": {
    "final_val_acc": 0.95,
    "final_val_loss": 0.15,
    "total_epochs": 50
  }
}
```

---

## 💡 Contoh Penggunaan

### 1. Testing dengan Swagger UI

1. Jalankan server: `uvicorn app.main:app --reload`
2. Buka browser: http://localhost:8000/docs
3. Klik endpoint `/predict`
4. Klik **"Try it out"**
5. Upload gambar bakteri
6. Klik **"Execute"**
7. Lihat hasil prediksi

---

### 2. Testing dengan cURL

#### Health Check
```powershell
curl http://localhost:8000/health
```

#### Single Prediction
```powershell
curl -X POST "http://localhost:8000/predict" `
  -H "accept: application/json" `
  -H "Content-Type: multipart/form-data" `
  -F "file=@path/to/bacteria_image.jpg"
```

#### Batch Prediction
```powershell
curl -X POST "http://localhost:8000/predict/batch" `
  -H "accept: application/json" `
  -H "Content-Type: multipart/form-data" `
  -F "files=@image1.jpg" `
  -F "files=@image2.jpg" `
  -F "files=@image3.jpg"
```

#### Model Info
```powershell
curl http://localhost:8000/model/info
```

---

### 3. Testing dengan Python (requests)

#### Install requests
```powershell
pip install requests
```

#### Single Image Prediction
```python
import requests

url = "http://localhost:8000/predict"
files = {'file': open('bacteria_sample.jpg', 'rb')}

response = requests.post(url, files=files)
print(response.json())

# Output:
# {
#   'filename': 'bacteria_sample.jpg',
#   'prediction': 'Gram Negative (G-)',
#   'confidence': 0.9234,
#   'probabilities': {...}
# }
```

#### Batch Prediction
```python
import requests

url = "http://localhost:8000/predict/batch"
files = [
    ('files', open('image1.jpg', 'rb')),
    ('files', open('image2.jpg', 'rb')),
    ('files', open('image3.jpg', 'rb'))
]

response = requests.post(url, files=files)
result = response.json()

print(f"Total images: {result['total_images']}")
print(f"Summary: {result['summary']}")
for pred in result['predictions']:
    print(f"{pred['filename']}: {pred['prediction']} ({pred['confidence']:.2%})")
```

#### Get Model Info
```python
import requests

response = requests.get("http://localhost:8000/model/info")
info = response.json()

print(f"Model: {info['model_name']} v{info['version']}")
print(f"Parameters: {info['parameters']}")
print(f"Architecture: {info['architecture']['layers']}")
print(f"Dataset size: {info['dataset']['total_samples']}")
```

---

### 4. Testing Script (Complete Example)

Buat file `test_api.py`:

```python
"""
Test script for Gram Bacteria Classification API
"""
import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(json.dumps(response.json(), indent=2))
    
def test_model_info():
    """Test model info endpoint"""
    print("\n=== Testing Model Info ===")
    response = requests.get(f"{BASE_URL}/model/info")
    print(json.dumps(response.json(), indent=2))

def test_predict_single(image_path):
    """Test single prediction"""
    print(f"\n=== Testing Single Prediction: {image_path} ===")
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/predict", files=files)
    
    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Probabilities: {json.dumps(result['probabilities'], indent=2)}")

def test_predict_batch(image_paths):
    """Test batch prediction"""
    print(f"\n=== Testing Batch Prediction: {len(image_paths)} images ===")
    
    files = [('files', open(path, 'rb')) for path in image_paths]
    response = requests.post(f"{BASE_URL}/predict/batch", files=files)
    
    result = response.json()
    print(f"Total images processed: {result['total_images']}")
    print(f"Summary: {json.dumps(result['summary'], indent=2)}")
    
    for pred in result['predictions']:
        print(f"\n{pred['filename']}:")
        print(f"  -> {pred['prediction']} ({pred['confidence']:.2%})")

if __name__ == "__main__":
    # Test endpoints
    test_health()
    test_model_info()
    
    # Test predictions (ganti dengan path gambar Anda)
    # test_predict_single("path/to/bacteria_image.jpg")
    # test_predict_batch(["image1.jpg", "image2.jpg", "image3.jpg"])
```

Jalankan:
```powershell
python test_api.py
```

---

## 🐳 Deployment

### Option 1: Docker (Recommended)

Buat `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements_api.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_api.txt

# Install PyTorch with CUDA
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Copy application
COPY app/ ./app/
COPY experiments/ ./experiments/

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build & Run:
```powershell
docker build -t bacteria-api .
docker run -p 8000:8000 bacteria-api
```

---

### Option 2: Production Server dengan Gunicorn

```powershell
pip install gunicorn

gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

### Option 3: Windows Service

Gunakan **NSSM** (Non-Sucking Service Manager):

1. Download NSSM: https://nssm.cc/download
2. Install service:
```powershell
nssm install BacteriaAPI "C:\Python310\python.exe" "-m uvicorn app.main:app --host 0.0.0.0 --port 8000"
nssm start BacteriaAPI
```

---

## 🔒 Security Best Practices

### 1. Update CORS Settings

Edit `app/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domain
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

### 2. Add API Key Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.post("/predict")
async def predict_single(
    file: UploadFile = File(...),
    api_key: str = Security(verify_api_key)
):
    # ... existing code
```

### 3. Rate Limiting

```powershell
pip install slowapi

# Add to main.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")
async def predict_single(...):
    # ... existing code
```

---

## 📊 Monitoring

### Add Prometheus Metrics

```powershell
pip install prometheus-fastapi-instrumentator
```

```python
# Add to main.py
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

Metrics akan tersedia di: `http://localhost:8000/metrics`

---

## 🐛 Troubleshooting

### Model Not Found Error
```
FileNotFoundError: Model file not found at experiments/exp1_baseline/best_model.pth
```

**Solusi:**
- Pastikan model sudah di-train dengan `python 03_train_baseline.py`
- Verifikasi path model di `app/config.py`

---

### CUDA Out of Memory (GPU)
```
RuntimeError: CUDA out of memory
```

**Solusi:**
- Set `USE_GPU = False` di `app/config.py` untuk menggunakan CPU
- Atau gunakan batch size lebih kecil

---

### Port Already in Use
```
ERROR: [Errno 10048] error while attempting to bind on address
```

**Solusi:**
```powershell
# Gunakan port lain
uvicorn app.main:app --reload --port 5000
```

---

## 📝 Changelog

### Version 1.0.0 (2025-11-20)
- ✅ Initial release
- ✅ Single image prediction endpoint
- ✅ Batch prediction endpoint
- ✅ Model info endpoint
- ✅ Health check endpoint
- ✅ Swagger UI documentation
- ✅ CORS support
- ✅ GPU/CPU auto-detection

---

## 📧 Support

Untuk pertanyaan atau issues, silakan hubungi tim development atau buka issue di repository GitHub.

---

## 📄 License

Project ini dibuat untuk keperluan Tugas Akhir (TA) - Klasifikasi Bakteri Gram menggunakan AI.
