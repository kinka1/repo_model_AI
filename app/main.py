from fastapi import FastAPI, File, HTTPException, UploadFile, Query, Response, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from typing import List
from typing import Optional
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import io
import json
import os
import shutil
import time
import hashlib
from datetime import datetime
from sqlalchemy.orm import Session

import torch
import torch.nn as nn
from torchvision import transforms
import uvicorn

from dotenv import load_dotenv
load_dotenv()

from app.database import engine, get_db, SessionLocal
import app.models as db_models
from app.routers import patients, analysis, dokter, admin, reports, auth, messages
from app.utils import get_local_now
from app.model_architectures import (
    SimpleCNN, 
    GramEfficientNetB0Classifier,
    GramEfficientNetB3Classifier,
    GramResNet50Classifier, 
    GramResNet101Classifier,
    GramVGG16Classifier,
    GramVGG19Classifier,
    GramDenseNet121Classifier,
)

app = FastAPI(
    title="🔬 Gram Bacteria Classification API",
    description="API untuk klasifikasi bakteri Gram-positif (G+) dan Gram-negatif (G-) dari gambar mikroskopis.",
    version="1.0.0",
)

# Allow frontend origins to load API responses and images.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "uploads")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Serve uploaded specimens/crops. Mount /static for legacy paths.
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/static", StaticFiles(directory=str(UPLOAD_DIR)), name="static")

AUTH_EXEMPT_PATHS = {
    "/",
    "/health",
    "/openapi.json",
    "/docs",
    "/redoc",
    "/api/auth/login",
    "/api/auth/forgot-password",
    "/api/auth/reset-password",
}


def _is_auth_exempt(path: str) -> bool:
    if path in AUTH_EXEMPT_PATHS:
        return True
    if path.startswith("/docs") or path.startswith("/redoc"):
        return True
    if path.startswith("/static/"):
        return True
    if path.startswith("/uploads/"):
        return True
    return False


@app.middleware("http")
async def access_token_middleware(request: Request, call_next):
    path = request.url.path

    if request.method == "OPTIONS":
        return await call_next(request)

    # Hanya proteksi endpoint API; non-API (health/docs) tetap terbuka.
    if not path.startswith("/api/"):
        return await call_next(request)

    if _is_auth_exempt(path):
        return await call_next(request)

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.lower().startswith("bearer "):
        return JSONResponse(status_code=401, content={"detail": "Authorization Bearer token dibutuhkan"})

    raw_token = auth_header[7:].strip()
    if not raw_token:
        return JSONResponse(status_code=401, content={"detail": "Access token kosong"})

    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

    db = SessionLocal()
    try:
        session = (
            db.query(db_models.Session)
            .filter(
                db_models.Session.token_hash == token_hash,
                db_models.Session.is_revoked.is_(False),
            )
            .first()
        )

        if not session:
            return JSONResponse(status_code=401, content={"detail": "Token tidak valid"})

        if session.expires_at < get_local_now():
            session.is_revoked = True
            db.add(session)
            db.commit()
            return JSONResponse(status_code=401, content={"detail": "Token sudah expired"})

        user = db.query(db_models.User).filter(db_models.User.id == session.user_id).first()
        if not user:
            return JSONResponse(status_code=401, content={"detail": "User tidak ditemukan"})
        if not user.is_active:
            return JSONResponse(status_code=403, content={"detail": "User nonaktif"})

        request.state.user_id = user.id
        request.state.username = user.username
        request.state.role = user.role
    finally:
        db.close()

    return await call_next(request)


@app.middleware("http")
async def add_static_corp_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/uploads/") or request.url.path.startswith("/static/"):
        response.headers.setdefault("Access-Control-Allow-Origin", "*")
        response.headers.setdefault("Cross-Origin-Resource-Policy", "cross-origin")
    return response

app.include_router(patients.router)
app.include_router(analysis.router)
app.include_router(dokter.router)
app.include_router(admin.router)
app.include_router(reports.router)
app.include_router(auth.router)
app.include_router(messages.router)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    openapi_schema.setdefault("components", {})
    openapi_schema["components"].setdefault("securitySchemes", {})
    openapi_schema["components"]["securitySchemes"]["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "Masukkan access token dari endpoint /api/auth/login",
    }

    public_api_paths = {
        "/api/auth/login",
        "/api/auth/forgot-password",
        "/api/auth/reset-password",
    }

    for path, methods in openapi_schema.get("paths", {}).items():
        if path in public_api_paths:
            continue

        for method_name, operation in methods.items():
            if method_name.lower() in {"get", "post", "put", "patch", "delete"}:
                operation.setdefault("security", [{"BearerAuth": []}])

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "best_model_cnn.pth"
DEFAULT_METRICS_PATH = ROOT_DIR / "models" / "metrics_summary.json"
MODEL_PATH = Path(os.environ.get("GRAM_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
PRODUCTION_MODEL_KEY = os.environ.get("PRODUCTION_MODEL_KEY", "resnet50")

CLASS_LABELS = ["Gram Negative (G-)", "Gram Positive (G+)"]
INPUT_SIZE = (224, 224)

# Model Registry - Configuration for all available models
MODEL_REGISTRY = {
    "scenario_1": {
        "name": "Simple CNN (From Scratch)",
        "class": SimpleCNN,
        "path": ROOT_DIR / "models" / "best_model_cnn.pth",
        "metrics_path": ROOT_DIR / "models" / "metrics_summary.json",
        "scenario": "Scenario 1: Simple CNN trained from scratch",
        "architecture": "SimpleCNN",
    },
    "scenario_2": {
        "name": "Simple CNN (With Augmentation)",
        "class": SimpleCNN,
        "path": ROOT_DIR / "models" / "best_model_cnn.pth",
        "metrics_path": ROOT_DIR / "models" / "metrics_summary.json",
        "scenario": "Scenario 2: Simple CNN with data augmentation",
        "architecture": "SimpleCNN",
    },
    "resnet50": {
        "name": "ResNet50 (Transfer Learning + Fine-tuning)",
        "class": GramResNet50Classifier,
        "path": ROOT_DIR / "models" / "best_model_resnet50.pth",
        "metrics_path": ROOT_DIR / "models" / "metrics_summary.json",
        "scenario": "Scenario 4a: ResNet50 fine-tuned (BEST MODEL)",
        "architecture": "ResNet50",
        "is_production": True,
    },
    "resnet101": {
        "name": "ResNet101 (Transfer Learning + Fine-tuning)",
        "class": GramResNet101Classifier,
        "path": ROOT_DIR / "models" / "best_model_resnet101.pth",
        "metrics_path": ROOT_DIR / "models" / "metrics_summary.json",
        "scenario": "Scenario 4b: ResNet101 fine-tuned",
        "architecture": "ResNet101",
    },
    "vgg16": {
        "name": "VGG16 (Transfer Learning + Fine-tuning)",
        "class": GramVGG16Classifier,
        "path": ROOT_DIR / "models" / "best_model_vgg16.pth",
        "metrics_path": ROOT_DIR / "models" / "metrics_summary.json",
        "scenario": "Scenario 5a: VGG16 fine-tuned",
        "architecture": "VGG16",
    },
    "vgg19": {
        "name": "VGG19 (Transfer Learning + Fine-tuning)",
        "class": GramVGG19Classifier,
        "path": ROOT_DIR / "models" / "best_model_vgg19.pth",
        "metrics_path": ROOT_DIR / "models" / "metrics_summary.json",
        "scenario": "Scenario 5b: VGG19 fine-tuned",
        "architecture": "VGG19",
    },
    "densenet121": {
        "name": "DenseNet121 (Transfer Learning + Fine-tuning)",
        "class": GramDenseNet121Classifier,
        "path": ROOT_DIR / "models" / "best_model_densenet121.pth",
        "metrics_path": ROOT_DIR / "models" / "metrics_summary.json",
        "scenario": "Scenario 6: DenseNet121 two-phase fine-tuning",
        "architecture": "DenseNet121",
    },
    "efficientnet_b0": {
        "name": "EfficientNet-B0 (Transfer Learning + Fine-tuning)",
        "class": GramEfficientNetB0Classifier,
        "path": ROOT_DIR / "models" / "best_model_efficientnet_b0.pth",
        "metrics_path": ROOT_DIR / "models" / "metrics_summary.json",
        "scenario": "Scenario 3c: EfficientNet-B0",
        "architecture": "EfficientNet-B0",
    },
    "efficientnet_b3": {
        "name": "EfficientNet-B3 (Transfer Learning + Fine-tuning)",
        "class": GramEfficientNetB3Classifier,
        "path": ROOT_DIR / "models" / "best_model_efficientnet_b3.pth",
        "metrics_path": ROOT_DIR / "models" / "metrics_summary.json",
        "scenario": "Scenario 3d: EfficientNet-B3",
        "architecture": "EfficientNet-B3",
    },
}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREPROCESS = transforms.Compose(
    [
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Global state for multiple models
MODELS: dict[str, nn.Module] = {}
MODELS_LOADED: dict[str, bool] = {}
MODEL_LOAD_ERRORS: dict[str, str] = {}
METRICS_CACHE: dict[str, dict] = {}

# Legacy single model support (for backward compatibility with /predict endpoint)
MODEL: Optional[nn.Module] = None
MODEL_LOADED = False
MODEL_LOAD_ERROR = ""


def _load_state_dict(model_path: Path) -> dict:
    """
    Load state dict from checkpoint file.
    Handles two formats:
    1. Direct state_dict (OrderedDict) - used by ResNet models
    2. Training checkpoint dict with 'model_state_dict' key - used by VGG models
    """
    # Use weights_only when supported to reduce unsafe pickle surface.
    try:
        state = torch.load(model_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        state = torch.load(model_path, map_location=DEVICE)

    # If it's a training checkpoint with metadata, extract the model state dict
    if isinstance(state, dict) and 'model_state_dict' in state:
        return state['model_state_dict']
    
    # Otherwise, assume it's already a state dict
    if not isinstance(state, dict):
        raise ValueError("Checkpoint format tidak valid: expected state_dict.")
    return state


def seed_ai_models() -> None:
    """Ensure all models in MODEL_REGISTRY exist in ai_models DB table.
    Only the PRODUCTION_MODEL_KEY model is set active for classification.
    YOLO detection model is seeded if missing. Fixes multiple-active-per-type issues."""
    from app.database import SessionLocal
    from app.models import AIModel

    db = SessionLocal()
    try:
        for model_key, config in MODEL_REGISTRY.items():
            existing = db.query(AIModel).filter(
                AIModel.model_name == model_key,
                AIModel.model_type == "Gram Classification",
            ).first()
            if not existing:
                metrics = METRICS_CACHE.get(model_key, {})
                test_m = metrics.get("test_metrics", {})
                db.add(AIModel(
                    model_name=model_key,
                    model_type="Gram Classification",
                    version=config.get("version", "1.0"),
                    model_file_path=str(config["path"]),
                    accuracy=test_m.get("accuracy"),
                    precision_score=test_m.get("precision"),
                    recall_score=test_m.get("recall"),
                    f1_score=test_m.get("f1") or test_m.get("f1_score"),
                    inference_time_s=metrics.get("inference_time_s"),
                    is_active=(model_key == PRODUCTION_MODEL_KEY),
                ))
                print(f"  [SEED] Added {model_key} to ai_models (active={model_key == PRODUCTION_MODEL_KEY})")

        # Register all YOLO .pt files found in models/ directory
        models_dir = ROOT_DIR / "models"
        yolo_files = sorted(models_dir.glob("*.pt"))
        primary_yolo = os.environ.get("YOLO_MODEL_PATH", "models/bacteria_yolo_model.pt")
        existing_detection = {m.model_name for m in db.query(AIModel).filter(AIModel.model_type == "Detection").all()}

        for yolo_file in yolo_files:
            yolo_name = yolo_file.stem  # e.g. "best_yolo", "best_yolo_new", "bacteria_yolo_model"
            if yolo_name not in existing_detection:
                is_primary = str(yolo_file) == str(ROOT_DIR / primary_yolo) or yolo_name == "bacteria_yolo_model"
                db.add(AIModel(
                    model_name=yolo_name,
                    model_type="Detection",
                    version="1.0",
                    model_file_path=str(yolo_file),
                    is_active=is_primary and yolo_name not in existing_detection,
                ))
                print(f"  [SEED] Added YOLO model: {yolo_name} (active={is_primary})")

        # Ensure at least one detection model is active
        active_detection = db.query(AIModel).filter(AIModel.model_type == "Detection", AIModel.is_active.is_(True)).first()
        if not active_detection:
            first_det = db.query(AIModel).filter(AIModel.model_type == "Detection").order_by(AIModel.id.asc()).first()
            if first_det:
                first_det.is_active = True
                print(f"  [FIX] Activated detection model: {first_det.model_name}")

        for mtype in ["Detection", "Gram Classification"]:
            active_models = (
                db.query(AIModel)
                .filter(AIModel.model_type == mtype, AIModel.is_active.is_(True))
                .order_by(AIModel.id.asc())
                .all()
            )
            if len(active_models) > 1:
                for m in active_models[1:]:
                    print(f"  [FIX] Deactivating extra active {mtype} model: {m.model_name} (id={m.id})")
                    m.is_active = False

        db.commit()
    finally:
        db.close()


def load_all_models() -> None:
    """Load only active models from the DB at startup."""
    global MODELS, MODELS_LOADED, MODEL_LOAD_ERRORS, METRICS_CACHE
    global MODEL, MODEL_LOADED, MODEL_LOAD_ERROR

    from app.database import SessionLocal
    from app.models import AIModel

    print(f"[*] Loading models on device: {DEVICE}")

    # Pre-cache metrics from JSON files
    for model_key, config in MODEL_REGISTRY.items():
        metrics_path = config["metrics_path"]
        if metrics_path.exists():
            try:
                METRICS_CACHE[model_key] = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                METRICS_CACHE[model_key] = {}

    db = SessionLocal()
    try:
        active_models = db.query(AIModel).filter(AIModel.is_active.is_(True)).all()
    finally:
        db.close()

    if not active_models:
        print("[WARN] No active models found in DB. Nothing loaded.")
        return

    needs_yolo = False

    for db_model in active_models:
        model_key = db_model.model_name

        if db_model.model_type == "Detection":
            needs_yolo = True
            continue

        config = MODEL_REGISTRY.get(model_key)
        if not config:
            MODEL_LOAD_ERRORS[model_key] = f"Model key '{model_key}' not in MODEL_REGISTRY -- skipped."
            print(f"    [ERROR] {MODEL_LOAD_ERRORS[model_key]}")
            continue

        model_path = config["path"]
        model_class = config["class"]
        MODELS_LOADED[model_key] = False
        MODEL_LOAD_ERRORS[model_key] = ""

        print(f"\n[*] Loading {config['name']} ({model_key})...")

        if not model_path.exists():
            MODEL_LOAD_ERRORS[model_key] = f"Model file tidak ditemukan: {model_path}"
            print(f"    [ERROR] {MODEL_LOAD_ERRORS[model_key]}")
            continue

        try:
            model = model_class().to(DEVICE)
            state_dict = _load_state_dict(model_path)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            MODELS[model_key] = model
            MODELS_LOADED[model_key] = True
            params = sum(p.numel() for p in model.parameters())
            print(f"    [OK] Loaded successfully ({params:,} parameters)")

            metrics = METRICS_CACHE.get(model_key, {})
            test_acc = metrics.get("test_metrics", {}).get("accuracy", 0)
            if test_acc:
                print(f"    [INFO] Test Accuracy: {test_acc:.2%}")

            if model_key == PRODUCTION_MODEL_KEY:
                MODEL = model
                MODEL_LOADED = True
                print(f"    [INFO] Set as production model for /predict endpoint")

        except Exception as exc:
            MODEL_LOAD_ERRORS[model_key] = f"Gagal load model: {exc}"
            print(f"    [ERROR] {MODEL_LOAD_ERRORS[model_key]}")
            continue

    if needs_yolo:
        print("\n[*] Loading YOLO detection model (active in DB)...")
        from app.ai_pipeline import load_yolo_model as _load_yolo
        _load_yolo()
    else:
        print("[*] No active Detection model -- YOLO not loaded.")

    loaded_count = sum(1 for v in MODELS_LOADED.values() if v)
    print(f"\n[*] Model loading complete: {loaded_count} classification model(s) loaded")

    if not MODEL_LOADED and MODELS:
        first_key = next(iter(MODELS.keys()))
        MODEL = MODELS[first_key]
        MODEL_LOADED = True
        print(f"[WARN] Production model fallback: using {first_key}")


def benchmark_model(model_key: str, test_data_path: str = None, sample_count: int = None) -> dict:
    """Run a classification model against a test dataset and return metrics.

    Args:
        model_key: Model key in MODEL_REGISTRY (e.g. 'resnet50').
        test_data_path: Path to ImageFolder test set. Falls back to TEST_DATA_PATH env var.
        sample_count: Max images to evaluate. Falls back to BENCHMARK_SAMPLE_COUNT env var.

    Returns:
        dict with accuracy, precision, recall, f1, inference_time_s, num_samples.
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets

    if model_key not in MODELS or not MODELS_LOADED.get(model_key):
        config = MODEL_REGISTRY.get(model_key)
        if not config:
            raise RuntimeError(f"Model '{model_key}' not found in MODEL_REGISTRY.")
        try:
            model_instance = config["class"]().to(DEVICE)
            sd = _load_state_dict(config["path"])
            model_instance.load_state_dict(sd, strict=True)
            model_instance.eval()
            MODELS[model_key] = model_instance
            MODELS_LOADED[model_key] = True
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_key}' for benchmark: {e}")

    if test_data_path is None:
        test_data_path = os.environ.get("TEST_DATA_PATH", "models/test")
    test_path = Path(test_data_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Test data directory not found: {test_path}")

    if sample_count is None:
        sample_count = int(os.environ.get("BENCHMARK_SAMPLE_COUNT", "200"))

    transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(str(test_path), transform=transform)
    class_names = dataset.classes

    if sample_count and len(dataset) > sample_count:
        indices = torch.randperm(len(dataset))[:sample_count].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    model = MODELS[model_key]
    model.eval()

    all_preds = []
    all_labels = []
    inference_times = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            start = time.perf_counter()
            outputs = model(images)
            elapsed = time.perf_counter() - start
            inference_times.append(elapsed)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = float(accuracy_score(all_labels, all_preds))
    precision = float(precision_score(all_labels, all_preds, average='macro', zero_division=0))
    recall = float(recall_score(all_labels, all_preds, average='macro', zero_division=0))
    f1 = float(f1_score(all_labels, all_preds, average='macro', zero_division=0))
    avg_inference_time_s = sum(inference_times) / len(inference_times) if inference_times else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "inference_time_s": round(avg_inference_time_s, 6),
        "num_samples": len(all_labels),
        "classes": class_names,
    }


def _prepare_yolo_dataset(base_dir: Path) -> Path:
    """Prepare a merged YOLO dataset from separate gram_positive/gram_negative folders.

    Supports Roboflow YOLOv8 export layout:
        gram_positive/
            train/images/  train/labels/
            valid/images/  valid/labels/
            test/images/   test/labels/
            data.yaml
        gram_negative/
            (same structure)

    If base_dir already has data.yaml, returns base_dir as-is.
    Otherwise merges all subfolder datasets into a combined/ folder.
    """
    if (base_dir / "data.yaml").exists():
        return base_dir

    # Find subfolders that look like YOLO datasets
    subdirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name != "combined"])
    class_dirs = []
    for d in subdirs:
        has_data = (d / "data.yaml").exists()
        has_images = (d / "images").exists()
        has_splits = any((d / s / "images").exists() for s in ("train", "valid", "val"))
        if has_data or has_images or has_splits:
            class_dirs.append(d)

    if not class_dirs:
        raise FileNotFoundError(
            f"Tidak ditemukan dataset YOLO di {base_dir}. "
            "Pastikan ada folder gram_positive/gram_negative dengan struktur Roboflow."
        )

    if len(class_dirs) == 1 and (class_dirs[0] / "data.yaml").exists():
        return class_dirs[0]

    # Merge
    combined = base_dir / "combined"
    if combined.exists():
        shutil.rmtree(combined)
    combined.mkdir()

    all_names = []
    counts = {"train": 0, "valid": 0, "test": 0}

    for cls_dir in class_dirs:
        cls_name = cls_dir.name

        # Read class names from data.yaml if available
        yaml_path = cls_dir / "data.yaml"
        if yaml_path.exists():
            try:
                import yaml as _yaml
                with open(yaml_path, "r", encoding="utf-8") as f:
                    ydata = _yaml.safe_load(f)
                names = ydata.get("names", [cls_name])
                if isinstance(names, dict):
                    names = list(names.values())
                for n in names:
                    if n not in all_names:
                        all_names.append(n)
            except Exception:
                if cls_name not in all_names:
                    all_names.append(cls_name)
        else:
            if cls_name not in all_names:
                all_names.append(cls_name)

        # Copy images and labels from each split
        # Roboflow uses: train/, valid/ (or val/), test/
        for split in ("train", "valid", "val", "test"):
            # Normalize: "val" and "valid" both map to "valid" in output
            out_split = "valid" if split in ("val", "valid") else split
            img_dir = cls_dir / split / "images"
            lbl_dir = cls_dir / split / "labels"

            if not img_dir.exists():
                continue

            out_img = combined / out_split / "images"
            out_lbl = combined / out_split / "labels"
            out_img.mkdir(parents=True, exist_ok=True)
            out_lbl.mkdir(parents=True, exist_ok=True)

            for f in img_dir.iterdir():
                if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                    dst = out_img / f"{cls_name}_{f.name}"
                    shutil.copy2(f, dst)
                    counts[out_split] = counts.get(out_split, 0) + 1

            if lbl_dir.exists():
                for f in lbl_dir.iterdir():
                    if f.is_file() and f.suffix == ".txt":
                        # Match label filename to image filename prefix
                        img_stem = f.stem
                        dst = out_lbl / f"{cls_name}_{img_stem}.txt"
                        shutil.copy2(f, dst)

    # Generate combined data.yaml
    import yaml as _yaml
    nc = len(all_names)

    # Determine which splits have data
    splits_used = []
    for s in ("train", "valid", "test"):
        if (combined / s / "images").exists() and any((combined / s / "images").iterdir()):
            splits_used.append(s)

    yaml_content = {
        "path": str(combined),
        "train": "train/images" if "train" in splits_used else "",
        "val": "valid/images" if "valid" in splits_used else "",
        "nc": nc,
        "names": {i: name for i, name in enumerate(all_names)},
    }
    if "test" in splits_used:
        yaml_content["test"] = "test/images"

    with open(combined / "data.yaml", "w", encoding="utf-8") as f:
        _yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    total = sum(counts.values())
    print(f"  [YOLO MERGE] Digabung: {counts} dari {len(class_dirs)} dataset, {nc} kelas")
    return combined


def benchmark_yolo(test_data_path: str = None, model_path: str = None) -> dict:
    """Run YOLO validation on a detection test dataset and return metrics.

    Supports two layouts:
    1. Single dataset with data.yaml at test_data_path
    2. Separate gram_positive/ and gram_negative/ folders, each with their own
       data.yaml or images/labels structure — auto-merged into a combined dataset.

    Args:
        test_data_path: Path to YOLO dataset directory.
        model_path: Path to YOLO .pt checkpoint. Falls back to YOLO_MODEL_PATH env var.

    Returns:
        dict with map50, map50_95, precision, recall, num_samples.
    """
    from ultralytics import YOLO as _YOLO

    if model_path is None:
        model_path = os.environ.get("YOLO_MODEL_PATH", str(ROOT_DIR / "models" / "best_yolo.pt"))

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"YOLO model file not found: {model_file}")

    if test_data_path is None:
        test_data_path = os.environ.get(
            "YOLO_TEST_DATA_PATH",
            str(ROOT_DIR / "models" / "yolo_test"),
        )

    test_dir = Path(test_data_path)
    if not test_dir.exists():
        raise FileNotFoundError(f"YOLO test data directory not found: {test_dir}")

    # Auto-prepare dataset (merge if needed)
    dataset_dir = _prepare_yolo_dataset(test_dir)
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml tidak ditemukan di {dataset_dir}")

    model = _YOLO(str(model_file))
    results = model.val(data=str(data_yaml), verbose=False)

    num_samples = 0
    if hasattr(results, "seen"):
        num_samples = int(results.seen)
    elif hasattr(results, "nt_per_class"):
        num_samples = int(sum(results.nt_per_class))
    elif hasattr(results, "stats") and results.stats:
        for v in results.stats.values():
            if hasattr(v, "__len__"):
                num_samples = max(num_samples, len(v))
                break

    return {
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
        "precision": float(results.box.p),
        "recall": float(results.box.r),
        "num_samples": num_samples,
    }


def _predict_tensor(image: Image.Image) -> dict:
    """Legacy prediction function for backward compatibility."""
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


def _predict_with_model(image: Image.Image, model_key: str) -> dict:
    """
    Run prediction using specified model from registry.
    
    Args:
        image: PIL Image to classify
        model_key: Model identifier from MODEL_REGISTRY
    
    Returns:
        Dictionary with prediction results and model metadata
    """
    # Validate model key
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model key: {model_key}. Available: {list(MODEL_REGISTRY.keys())}")
    
    # Check if model is loaded
    if not MODELS_LOADED.get(model_key, False):
        error = MODEL_LOAD_ERRORS.get(model_key, "Unknown error")
        raise RuntimeError(f"Model '{model_key}' belum siap: {error}")
    
    # Get model
    model = MODELS[model_key]
    config = MODEL_REGISTRY[model_key]
    
    # Preprocess image
    image = image.convert("RGB")
    tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)
    
    # Run inference
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()
    inference_ms = (time.perf_counter() - start) * 1000.0
    
    # Parse results
    class_id = int(max(range(len(probs)), key=lambda idx: probs[idx]))
    confidence = float(probs[class_id])
    
    # Get model metadata
    metrics = METRICS_CACHE.get(model_key, {})
    test_metrics = metrics.get("test_metrics", {})
    
    model_metadata = {
        "model_key": model_key,
        "model_name": config["name"],
        "scenario": config["scenario"],
        "architecture": config["architecture"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "test_accuracy": round(test_metrics.get("accuracy", 0), 4),
        "test_precision": round(test_metrics.get("precision", 0), 4),
        "test_recall": round(test_metrics.get("recall", 0), 4),
        "test_f1": round(test_metrics.get("f1_score", 0), 4),
        "best_epoch": metrics.get("best_epoch"),
    }
    
    return {
        "prediction": CLASS_LABELS[class_id],
        "class_id": class_id,
        "confidence": round(confidence, 6),
        "probabilities": {
            CLASS_LABELS[0]: round(float(probs[0]), 6),
            CLASS_LABELS[1]: round(float(probs[1]), 6),
        },
        "inference_ms": round(inference_ms, 2),
        "model_metadata": model_metadata,
    }


def _detect_and_classify(image: Image.Image, model_key: str, conf_threshold: float = 0.25, return_annotated_image: bool = False) -> dict:
    """
    Two-stage pipeline: YOLO detection + Gram classification.
    
    1. Use YOLO to detect bacteria regions
    2. Classify each detected region using the specified model
    
    Args:
        image: PIL Image to process
        model_key: Model identifier for classification
        conf_threshold: YOLO confidence threshold (default: 0.25)
        return_annotated_image: If True, includes annotated image with bounding boxes and labels
    
    Returns:
        Dictionary with detection results and classifications (and optional annotated image)
    """
    from app.ai_pipeline import detect_and_crop, YOLO_LOADED, YOLO_LOAD_ERROR
    
    # Check if YOLO is available
    if not YOLO_LOADED:
        raise RuntimeError(f"YOLO model belum siap: {YOLO_LOAD_ERROR}")
    
    # Validate classification model
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model key: {model_key}. Available: {list(MODEL_REGISTRY.keys())}")
    
    if not MODELS_LOADED.get(model_key, False):
        error = MODEL_LOAD_ERRORS.get(model_key, "Unknown error")
        raise RuntimeError(f"Classification model '{model_key}' belum siap: {error}")
    
    # Stage 1: YOLO Detection
    start_detection = time.perf_counter()
    detections = detect_and_crop(image, conf_threshold=conf_threshold)
    detection_ms = (time.perf_counter() - start_detection) * 1000.0
    
    # If no bacteria detected, return early
    if not detections:
        return {
            "bacteria_detected": False,
            "detection_count": 0,
            "detection_time_ms": round(detection_ms, 2),
            "message": "Tidak ada bakteri terdeteksi pada gambar. Coba dengan gambar lain atau turunkan confidence threshold.",
            "detections": []
        }
    
    # Stage 2: Classify each detected region
    classifications = []
    total_classification_time = 0.0
    
    for i, detection in enumerate(detections):
        crop_img = detection["crop"]
        box = detection["box"]
        detection_conf = detection["confidence"]
        
        # Classify the cropped region
        start_classify = time.perf_counter()
        classification = _predict_with_model(crop_img, model_key)
        classify_ms = (time.perf_counter() - start_classify) * 1000.0
        total_classification_time += classify_ms
        
        classifications.append({
            "detection_id": i + 1,
            "bounding_box": box,
            "detection_confidence": round(detection_conf, 4),
            "prediction": classification["prediction"],
            "classification_confidence": classification["confidence"],
            "probabilities": classification["probabilities"],
            "classification_time_ms": round(classify_ms, 2)
        })
    
    # Calculate statistics
    gram_positive_count = sum(1 for c in classifications if c["prediction"] == CLASS_LABELS[1])
    gram_negative_count = sum(1 for c in classifications if c["prediction"] == CLASS_LABELS[0])
    avg_classification_conf = sum(c["classification_confidence"] for c in classifications) / len(classifications)
    
    # Get model metadata
    config = MODEL_REGISTRY[model_key]
    metrics = METRICS_CACHE.get(model_key, {})
    test_metrics = metrics.get("test_metrics", {})
    
    # Prepare result dictionary
    result = {
        "bacteria_detected": True,
        "detection_count": len(detections),
        "detection_time_ms": round(detection_ms, 2),
        "total_classification_time_ms": round(total_classification_time, 2),
        "total_time_ms": round(detection_ms + total_classification_time, 2),
        "summary": {
            "total_bacteria": len(detections),
            "gram_positive": gram_positive_count,
            "gram_negative": gram_negative_count,
            "average_classification_confidence": round(avg_classification_conf, 4)
        },
        "detections": classifications,
        "model_metadata": {
            "model_key": model_key,
            "model_name": config["name"],
            "scenario": config["scenario"],
            "architecture": config["architecture"],
            "test_accuracy": round(test_metrics.get("accuracy", 0), 4),
        }
    }
    
    # Add annotated image if requested
    if return_annotated_image:
        from app.visualization import draw_detections_on_image, image_to_base64
        
        # Prepare detection data for visualization
        detection_data = []
        for classification in classifications:
            box = classification["bounding_box"]
            prediction = classification["prediction"]
            confidence = classification["classification_confidence"]
            
            # Determine class for color coding
            class_label = "G+" if prediction == CLASS_LABELS[1] else "G-"
            
            detection_data.append({
                "box": box,
                "class": class_label,
                "confidence": confidence,
                "detection_id": classification["detection_id"]
            })
        
        try:
            # Draw annotations on image
            annotated_image = draw_detections_on_image(image.copy(), detection_data)
            
            # Convert to base64
            annotated_image_b64 = image_to_base64(annotated_image)
            
            # Add to result
            result["annotated_image_base64"] = annotated_image_b64
            result["image_metadata"] = {
                "width": annotated_image.width,
                "height": annotated_image.height,
                "format": annotated_image.format or "JPEG"
            }
        except Exception as viz_error:
            # If visualization fails, add error but don't fail the whole request
            result["visualization_error"] = f"Failed to generate annotated image: {viz_error}"
    
    return result


@app.on_event("startup")
async def startup_event() -> None:
    # Initialize database tables
    db_models.Base.metadata.create_all(bind=engine)
    # Seed DB model records, then load only active models
    seed_ai_models()
    load_all_models()

# Mount frontend folder for demo UI if it exists
if os.path.exists("frontend"):
    app.mount("/demo", StaticFiles(directory="frontend", html=True), name="frontend")
else:
    print("[WARN] frontend folder not found, /demo endpoint disabled")

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
            "health_check": "GET /health",
            "list_test_models": "GET /api/test/models",
            "test_prediction": "POST /api/test/predict?model=resnet50",
            "test_prediction_with_yolo": "POST /api/test/predict-with-detection?model=resnet50&confidence=0.25",
            "test_prediction_with_visualization": "POST /api/test/predict-with-detection?model=resnet50&confidence=0.25&visualize=true",
            "test_direct_visualization": "POST /api/test/visualize?model=resnet50&confidence=0.25&format=png"
        },
        "visualization": {
            "description": "New visualization endpoints available for bacteria detection results",
            "features": [
                "Color-coded bounding boxes: Green for G+ (Gram Positive), Red for G- (Gram Negative)",
                "Confidence percentages displayed above each detection",
                "Auto-scaling box thickness and font sizes based on image dimensions",
                "Support for both embedded base64 images in JSON and direct image responses"
            ],
            "usage": {
                "json_with_visualization": "Add 'visualize=true' parameter to /api/test/predict-with-detection",
                "direct_image_output": "Use /api/test/visualize endpoint for PNG/JPEG image response"
            }
        }
    }

@app.delete("/api/analysis/cleanup/{specimen_id}")
def cleanup_specimen_classifications(specimen_id: int, request: Request, db: Session = Depends(get_db)):
    from app.models import Specimen, Classification
    import shutil

    specimen = db.query(Specimen).filter(Specimen.id == specimen_id).first()
    if not specimen:
        raise HTTPException(status_code=404, detail="Specimen tidak ditemukan")

    # Immutability check: Analis cannot modify validated specimens
    if specimen.status == "validated" and getattr(request.state, "role", None) == "Analis":
        raise HTTPException(
            status_code=403,
            detail="Data sudah tervalidasi (Selesai). Tidak dapat diubah. Hubungi Dokter untuk membuka kunci."
        )
        
    db.query(Classification).filter(Classification.specimen_id == specimen_id).delete()
    
    crops_dir = os.path.join("uploads", "crops", str(specimen_id))
    if os.path.exists(crops_dir):
        try:
            shutil.rmtree(crops_dir)
        except Exception as e:
            print(f"Failed to delete crops dir: {e}")
            
    specimen.status = "pending"
    specimen.validation_status = "pending"
    specimen.total_detected = 0
    db.commit()
    
    return {"success": True, "message": "Classifications cleaned up successfully"}

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

@app.get("/api/test/models")
async def list_available_models():
    """
    List all available models for testing.
    
    Returns information about each model including:
    - Classification models (CNN, ResNet50, ResNet101)
    - YOLO detection model
    - Model key/identifier
    - Name and scenario description
    - Architecture type
    - Total parameters
    - Test accuracy and other metrics
    - Loading status
    """
    from app.ai_pipeline import YOLO_LOADED, YOLO_LOAD_ERROR, YOLO_MODEL_PATH
    
    models_info = []
    
    # Add classification models
    for model_key, config in MODEL_REGISTRY.items():
        # Get metrics
        metrics = METRICS_CACHE.get(model_key, {})
        test_metrics = metrics.get("test_metrics", {})
        
        # Calculate model size
        model_path = config["path"]
        model_size_mb = 0
        if model_path.exists():
            model_size_mb = round(model_path.stat().st_size / (1024 * 1024), 2)
        
        # Get parameter count
        params = 0
        if MODELS_LOADED.get(model_key, False):
            params = sum(p.numel() for p in MODELS[model_key].parameters())
        
        model_info = {
            "type": "classification",
            "key": model_key,
            "name": config["name"],
            "scenario": config["scenario"],
            "loaded": MODELS_LOADED.get(model_key, False),
            "error": MODEL_LOAD_ERRORS.get(model_key, None) if not MODELS_LOADED.get(model_key, False) else None,
            "architecture": config["architecture"],
            "parameters": params,
            "model_size_mb": model_size_mb,
            "test_accuracy": round(test_metrics.get("accuracy", 0), 4),
            "test_precision": round(test_metrics.get("precision", 0), 4),
            "test_recall": round(test_metrics.get("recall", 0), 4),
            "test_f1": round(test_metrics.get("f1_score", 0), 4),
            "best_epoch": metrics.get("best_epoch"),
        }
        
        # Add production flag if applicable
        if config.get("is_production", False):
            model_info["is_production"] = True
        
        models_info.append(model_info)
    
    # Add YOLO detection model
    yolo_size_mb = 0
    if YOLO_MODEL_PATH.exists():
        yolo_size_mb = round(YOLO_MODEL_PATH.stat().st_size / (1024 * 1024), 2)
    
    yolo_info = {
        "type": "detection",
        "key": "yolo_bacteria",
        "name": "YOLO Bacteria Detection Model",
        "scenario": "Object Detection: Locates bacteria regions in microscopic images",
        "loaded": YOLO_LOADED,
        "error": YOLO_LOAD_ERROR if not YOLO_LOADED else None,
        "architecture": "YOLOv8",
        "model_size_mb": yolo_size_mb,
        "purpose": "Pre-detection stage to locate bacteria before Gram classification",
        "output": "Bounding boxes with confidence scores",
    }
    
    models_info.append(yolo_info)
    
    # Count loaded models
    loaded_classification = sum(1 for v in MODELS_LOADED.values() if v)
    loaded_detection = 1 if YOLO_LOADED else 0
    total_loaded = loaded_classification + loaded_detection
    
    return {
        "total_models": len(MODEL_REGISTRY) + 1,  # +1 for YOLO
        "classification_models": len(MODEL_REGISTRY),
        "detection_models": 1,
        "loaded_models": total_loaded,
        "models": models_info
    }


@app.post("/api/test/predict")
async def test_model_prediction(
    file: UploadFile = File(...),
    model: str = Query(
        default="resnet50",
        description="Model to use for prediction. Options: scenario_1, scenario_2, resnet50, resnet101, vgg16, vgg19"
    )
):
    """
    Test image classification using a selected pre-trained model.
    
    This endpoint allows testing different models on uploaded images.
    Use GET /api/test/models to see all available models and their details.
    
    Query Parameters:
    - model: Model identifier (scenario_1, scenario_2, resnet50, resnet101, vgg16, vgg19)
    
    Response includes:
    - Prediction result (class, confidence, probabilities)
    - Model metadata (name, accuracy, parameters, scenario)
    - Inference time in milliseconds
    """
    # Validate model key
    if model not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model key: '{model}'. Available models: {available_models}"
        )
    
    # Check if model is loaded
    if not MODELS_LOADED.get(model, False):
        error = MODEL_LOAD_ERRORS.get(model, "Unknown error")
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model}' tidak tersedia: {error}"
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File harus berupa gambar (image/*)."
        )
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Run prediction with selected model
        prediction = _predict_with_model(image, model)
        
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail="Format gambar tidak didukung atau file rusak."
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal melakukan prediksi: {exc}"
        ) from exc
    
    return {"filename": file.filename, **prediction}


@app.post("/api/test/predict-with-detection")
async def test_model_prediction_with_yolo(
    file: UploadFile = File(...),
    model: str = Query(
        default="resnet50",
        description="Model to use for classification. Options: scenario_1, scenario_2, resnet50, resnet101, vgg16, vgg19"
    ),
    confidence: float = Query(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="YOLO confidence threshold for bacteria detection (0.01-1.0)"
    ),
    visualize: bool = Query(
        default=False,
        description="Include annotated image with bounding boxes and labels in the response"
    )
):
    """
    Two-stage prediction: YOLO bacteria detection + Gram classification.
    
    This endpoint provides a more robust prediction workflow:
    1. First, uses YOLO to detect bacteria regions in the image
    2. Then, classifies each detected region using the selected model
    
    This ensures that only actual bacteria are classified, reducing false positives
    from background noise or non-bacteria objects in the image.
    
    Query Parameters:
    - model: Model identifier for classification (scenario_1, scenario_2, resnet50, resnet101, vgg16, vgg19)
    - confidence: YOLO confidence threshold (default: 0.25, range: 0.01-1.0)
    - visualize: Include annotated image with bounding boxes and labels (default: false)
    
    Response includes:
    - Detection results (number of bacteria found, bounding boxes)
    - Classification for each detected bacterium
    - Summary statistics (Gram positive/negative counts)
    - Total processing time breakdown
    - Optional: Base64-encoded annotated image with visualizations
    """
    # Validate model key
    if model not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model key: '{model}'. Available models: {available_models}"
        )
    
    # Check if classification model is loaded
    if not MODELS_LOADED.get(model, False):
        error = MODEL_LOAD_ERRORS.get(model, "Unknown error")
        raise HTTPException(
            status_code=503,
            detail=f"Classification model '{model}' tidak tersedia: {error}"
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File harus berupa gambar (image/*)."
        )
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Run YOLO detection + classification pipeline
        result = _detect_and_classify(image, model, conf_threshold=confidence, return_annotated_image=visualize)
        
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail="Format gambar tidak didukung atau file rusak."
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal melakukan deteksi dan klasifikasi: {exc}"
        ) from exc
    
    return {
        "filename": file.filename,
        "yolo_confidence_threshold": confidence,
        "visualization_enabled": visualize,
        **result
    }


@app.post("/api/test/visualize")
async def test_model_prediction_with_visualization(
    file: UploadFile = File(...),
    model: str = Query(
        default="resnet50",
        description="Model to use for classification. Options: scenario_1, scenario_2, resnet50, resnet101"
    ),
    confidence: float = Query(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="YOLO confidence threshold for bacteria detection (0.01-1.0)"
    ),
    format: str = Query(
        default="png",
        description="Output image format. Options: png, jpeg"
    )
):
    """
    YOLO detection + Gram classification with direct image visualization response.
    
    This endpoint returns the annotated image directly as image bytes instead of JSON.
    Use this when you want to display the visualization directly in a browser or save it as a file.
    
    The returned image includes:
    - Bounding boxes around detected bacteria
    - Color-coded labels: Green for G+ (Gram Positive), Red for G- (Gram Negative)  
    - Confidence percentages for each classification
    
    Query Parameters:
    - model: Model identifier for classification (scenario_1, scenario_2, resnet50, resnet101)
    - confidence: YOLO confidence threshold (default: 0.25, range: 0.01-1.0)
    - format: Output format (png or jpeg, default: png)
    
    Returns:
    - Raw image bytes with Content-Type header set appropriately
    """
    # Validate model key
    if model not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model key: '{model}'. Available models: {available_models}"
        )
    
    # Check if classification model is loaded
    if not MODELS_LOADED.get(model, False):
        error = MODEL_LOAD_ERRORS.get(model, "Unknown error")
        raise HTTPException(
            status_code=503,
            detail=f"Classification model '{model}' tidak tersedia: {error}"
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File harus berupa gambar (image/*)."
        )
    
    # Validate format
    if format.lower() not in ["png", "jpeg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid format. Supported formats: png, jpeg"
        )
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Run YOLO detection + classification pipeline
        result = _detect_and_classify(image, model, conf_threshold=confidence, return_annotated_image=True)
        
        # Check if bacteria were detected
        if not result.get("bacteria_detected", False):
            # Return original image with a "No bacteria detected" message overlaid
            from app.visualization import image_to_bytes
            from PIL import ImageDraw, ImageFont
            
            # Create a copy of the original image to draw on
            annotated_image = image.copy()
            draw = ImageDraw.Draw(annotated_image)
            
            # Try to use a decent font, fallback to default
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except (OSError, IOError):
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 24)
                except (OSError, IOError):
                    font = ImageFont.load_default()
            
            # Draw "No bacteria detected" message in center
            message = "No bacteria detected"
            width, height = annotated_image.size
            text_width = draw.textlength(message, font=font) if hasattr(draw, 'textlength') else len(message) * 12
            text_height = 24
            
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            
            # Draw background rectangle
            padding = 10
            draw.rectangle([
                x - padding, y - padding,
                x + text_width + padding, y + text_height + padding
            ], fill=(0, 0, 0, 180))
            
            # Draw text
            draw.text((x, y), message, fill=(255, 255, 255), font=font)
            
            # Convert to bytes
            image_bytes = image_to_bytes(annotated_image, format=format.upper())
            
            # Set content type
            content_type = f"image/{format.lower()}"
            
            return Response(content=image_bytes, media_type=content_type)
        
        # Check if visualization was successful
        if "annotated_image_base64" not in result:
            error_msg = result.get("visualization_error", "Unknown visualization error")
            raise HTTPException(
                status_code=500,
                detail=f"Visualization failed: {error_msg}"
            )
        
        # Convert base64 back to image bytes in requested format
        from app.visualization import image_to_bytes
        import base64
        
        # Decode base64 image
        annotated_image_b64 = result["annotated_image_base64"]
        image_data = base64.b64decode(annotated_image_b64)
        annotated_image = Image.open(io.BytesIO(image_data))
        
        # Convert to requested format
        image_bytes = image_to_bytes(annotated_image, format=format.upper())
        
        # Set content type
        content_type = f"image/{format.lower()}"
        
        return Response(content=image_bytes, media_type=content_type)
        
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail="Format gambar tidak didukung atau file rusak."
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal melakukan deteksi dan visualisasi: {exc}"
        ) from exc


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
