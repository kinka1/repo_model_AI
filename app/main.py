from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from typing import Optional
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import io
import json
import os
import time

import torch
import torch.nn as nn
from torchvision import models, transforms
import uvicorn

from dotenv import load_dotenv
load_dotenv()

from app.database import engine, get_db
import app.models as db_models
from app.routers import patients, analysis

app = FastAPI(
    title="🔬 Gram Bacteria Classification API",
    description="API untuk klasifikasi bakteri Gram-positif (G+) dan Gram-negatif (G-) dari gambar mikroskopis.",
    version="1.0.0",
)

app.include_router(patients.router)
app.include_router(analysis.router)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT_DIR / "experiments" / "scenario_4a_resnet50" / "best_model.pth"
DEFAULT_METRICS_PATH = ROOT_DIR / "experiments" / "scenario_4a_resnet50" / "metrics_summary.json"
MODEL_PATH = Path(os.environ.get("GRAM_MODEL_PATH", str(DEFAULT_MODEL_PATH)))

CLASS_LABELS = ["Gram Negative (G-)", "Gram Positive (G+)"]
INPUT_SIZE = (224, 224)


class GramResNet50Classifier(nn.Module):
    """Matches the training checkpoint structure with `backbone.*` keys."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREPROCESS = transforms.Compose(
    [
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

MODEL: Optional[nn.Module] = None
MODEL_LOADED = False
MODEL_LOAD_ERROR = ""
METRICS_CACHE = {}


def _load_state_dict(model_path: Path) -> dict:
    # Use weights_only when supported to reduce unsafe pickle surface.
    try:
        state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=DEVICE)

    if not isinstance(state, dict):
        raise ValueError("Checkpoint format tidak valid: expected state_dict.")
    return state


def load_model() -> None:
    global MODEL, MODEL_LOADED, MODEL_LOAD_ERROR, METRICS_CACHE
    MODEL_LOADED = False
    MODEL_LOAD_ERROR = ""

    if not MODEL_PATH.exists():
        MODEL_LOAD_ERROR = f"Model file tidak ditemukan: {MODEL_PATH}"
        return

    model = GramResNet50Classifier().to(DEVICE)

    try:
        state_dict = _load_state_dict(MODEL_PATH)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        MODEL = model
        MODEL_LOADED = True
    except Exception as exc:
        MODEL_LOAD_ERROR = f"Gagal load model: {exc}"
        MODEL = None
        return

    if DEFAULT_METRICS_PATH.exists():
        try:
            METRICS_CACHE = json.loads(DEFAULT_METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            METRICS_CACHE = {}


def _predict_tensor(image: Image.Image) -> dict:
    if MODEL is None:
        raise RuntimeError("Model belum siap.")

    image = image.convert("RGB")
    tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)

    start = time.perf_counter()
    with torch.no_grad():
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()
    inference_ms = (time.perf_counter() - start) * 1000.0

    class_id = int(max(range(len(probs)), key=lambda idx: probs[idx]))
    confidence = float(probs[class_id])

    return {
        "prediction": CLASS_LABELS[class_id],
        "class_id": class_id,
        "confidence": round(confidence, 6),
        "probabilities": {
            CLASS_LABELS[0]: round(float(probs[0]), 6),
            CLASS_LABELS[1]: round(float(probs[1]), 6),
        },
        "inference_ms": round(inference_ms, 2),
    }


@app.on_event("startup")
async def startup_event() -> None:
    # Initialize database tables
    db_models.Base.metadata.create_all(bind=engine)
    load_model()
    # Load YOLO Model
    from app.ai_pipeline import load_yolo_model
    load_yolo_model()

from fastapi.staticfiles import StaticFiles

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount uploads directory so images can be accessed via URL
app.mount("/static", StaticFiles(directory="uploads"), name="static")

# Mount frontend folder for demo UI
app.mount("/demo", StaticFiles(directory="frontend", html=True), name="frontend")

@app.get("/")
async def root():
    return {
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

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if MODEL_LOADED else "degraded",
        "model_loaded": MODEL_LOADED,
        "device": str(DEVICE),
        "model_path": str(MODEL_PATH),
        "error": MODEL_LOAD_ERROR or None,
        "version": "1.0.0"
    }

@app.post("/predict")
async def predict_single(file: UploadFile = File(...)):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail=f"Model belum siap: {MODEL_LOAD_ERROR}")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar (image/*).")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        prediction = _predict_tensor(image)
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Format gambar tidak didukung atau file rusak.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gagal melakukan prediksi: {exc}") from exc

    return {"filename": file.filename, **prediction}

@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    predictions = []
    gram_neg_count = 0
    gram_pos_count = 0
    confidence_sum = 0.0

    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail=f"Model belum siap: {MODEL_LOAD_ERROR}")

    for f in files:
        try:
            contents = await f.read()
            image = Image.open(io.BytesIO(contents))
            result = _predict_tensor(image)
            predictions.append({"filename": f.filename, **result})
            confidence_sum += float(result["confidence"])

            if result["class_id"] == 0:
                gram_neg_count += 1
            else:
                gram_pos_count += 1
        except Exception as exc:
            predictions.append(
                {
                    "filename": f.filename,
                    "error": f"Gagal diproses: {exc}",
                }
            )

    valid_predictions = max(1, gram_neg_count + gram_pos_count)
    
    return {
        "total_images": len(files),
        "predictions": predictions,
        "summary": {
            "Gram Negative (G-)": gram_neg_count,
            "Gram Positive (G+)": gram_pos_count,
            "average_confidence": round(confidence_sum / valid_predictions, 6),
        }
    }

@app.get("/model/info")
async def model_info():
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail=f"Model belum siap: {MODEL_LOAD_ERROR}")

    params = sum(p.numel() for p in MODEL.parameters()) if MODEL else 0

    return {
        "model_name": "ResNet50 Fine-tuned (Scenario 4a)",
        "version": "1.0",
        "parameters": params,
        "checkpoint": str(MODEL_PATH),
        "architecture": {
            "type": "Convolutional Neural Network",
            "backbone": "ResNet50",
            "head": "Dropout + Linear(2048->512) + ReLU + Dropout + Linear(512->2)",
            "input_size": "224x224x3",
            "output_classes": 2,
            "class_labels": CLASS_LABELS,
        },
        "dataset": {
            "total_samples": 11824,
            "classes": 2,
            "split": "70% train, 15% val, 15% test",
            "image_size": "224x224x3"
        },
        "performance": METRICS_CACHE.get("test_metrics", {}),
        "best_epoch": METRICS_CACHE.get("best_epoch"),
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
