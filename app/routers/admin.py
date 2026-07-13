import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Optional
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import (
    AIModel,
    Classification,
    ModelRetrainConfig,
    ModelTrainingStatus,
    User,
)
from app.schemas import (
    BenchmarkResponse,
    YoloBenchmarkAllResponse,
    YoloBenchmarkResponse,
    ModelUploadResponse,
    ModelUploadResponse,
    BenchmarkResponse,
    ActiveModelResponse,
    AIModelSummaryResponse,
    BestModelResponse,
    MessageResponse,
    PaginatedResponse,
    RetrainConfigResponse,
    RetrainConfigUpdateRequest,
    RetrainModelOptionResponse,
    RetrainStartRequest,
    RetrainStartResponse,
    RoleListResponse,
    TrainingJobResponse,
    UserCreateRequest,
    UserResponseSchema,
    UserUpdateRequest,
)
from app.utils import paginate_query, get_local_now


router = APIRouter(prefix="/api/admin", tags=["Admin Management"])

VALID_ROLES = ["Admin", "Analis", "Dokter"]
ROLE_MAP = {role.lower(): role for role in VALID_ROLES}
ROOT_DIR = Path(__file__).resolve().parents[2]

@router.get("/models/trend")
def get_trend_data(
    period: str = Query("daily", description="Periode: daily, weekly, monthly, yearly"),
    days: int = Query(30, description="Jumlah hari untuk distribusi confidence"),
    db: Session = Depends(get_db),
):
    """Ambil data distribusi confidence model & drift untuk admin.
    
    Menggantikan chart tren Gram Positif/Negatif yang tidak relevan secara klinis
    dengan visualisasi distribusi confidence dan pergeseran (drift) performa model.
    """
    from datetime import date, timedelta
    from sqlalchemy import case
    from app.utils import get_local_now

    now = get_local_now()
    today = now.date()

    # --- CONFIDENCE DISTRIBUTION: Current Period (last `days` days) ---
    current_start = today - timedelta(days=days)
    current_end = today

    def _get_confidence_distribution(start_date, end_date):
        rows = (
            db.query(
                case(
                    (Classification.confidence_score >= 0.9, "90-100%"),
                    (Classification.confidence_score >= 0.8, "80-90%"),
                    (Classification.confidence_score >= 0.7, "70-80%"),
                    else_="< 70%",
                ).label("range"),
                func.count().label("count"),
            )
            .filter(Classification.confidence_score.isnot(None))
            .filter(Classification.classified_at.isnot(None))
            .filter(func.date(Classification.classified_at) >= start_date)
            .filter(func.date(Classification.classified_at) <= end_date)
            .group_by("range")
            .all()
        )

        conf_map = {"90-100%": 0, "80-90%": 0, "70-80%": 0, "< 70%": 0}
        for row in rows:
            conf_map[row.range] = row.count

        total = sum(conf_map.values()) or 1  # Avoid division by zero
        return [
            {
                "range": r,
                "count": conf_map[r],
                "percentage": round((conf_map[r] / total) * 100, 1),
            }
            for r in ["90-100%", "80-90%", "70-80%", "< 70%"]
        ]

    confidence_current = _get_confidence_distribution(current_start, current_end)
    confidence_previous = _get_confidence_distribution(
        current_start - timedelta(days=days),
        current_start - timedelta(days=1),
    )

    # --- DRIFT CALCULATION ---
    drift = {}
    for curr, prev in zip(confidence_current, confidence_previous):
        drift[curr["range"]] = {
            "delta_count": curr["count"] - prev["count"],
            "delta_percentage": round(curr["percentage"] - prev["percentage"], 1),
        }

    # --- ACCURACY TREND ---
    def _get_avg_accuracy(start_date, end_date):
        result = (
            db.query(func.avg(Classification.confidence_score))
            .filter(Classification.confidence_score.isnot(None))
            .filter(Classification.classified_at.isnot(None))
            .filter(func.date(Classification.classified_at) >= start_date)
            .filter(func.date(Classification.classified_at) <= end_date)
            .scalar()
        )
        return round(float(result), 4) if result else 0.0

    avg_confidence_current = _get_avg_confidence(current_start, current_end, db)
    avg_confidence_previous = _get_avg_confidence(
        current_start - timedelta(days=days),
        current_start - timedelta(days=1),
        db,
    )

    total_current = sum(c["count"] for c in confidence_current)
    total_previous = sum(c["count"] for c in confidence_previous)

    # --- DRIFT DIRECTION ---
    low_conf_current = confidence_current[2]["percentage"] + confidence_current[3]["percentage"]
    low_conf_previous = confidence_previous[2]["percentage"] + confidence_previous[3]["percentage"]

    if low_conf_current < low_conf_previous - 2:
        drift_direction = "membaik"
    elif low_conf_current > low_conf_previous + 2:
        drift_direction = "menurun"
    else:
        drift_direction = "stabil"

    return {
        "confidence_current": confidence_current,
        "confidence_previous": confidence_previous,
        "drift": drift,
        "summary": {
            "total_classifications_current": total_current,
            "total_classifications_previous": total_previous,
            "avg_confidence_current": avg_confidence_current,
            "avg_confidence_previous": avg_confidence_previous,
            "drift_direction": drift_direction,
            "period_days": days,
        },
    }


def _get_avg_confidence(start_date, end_date, session: Session):
    """Helper to get average confidence score for a date range."""
    result = (
        session.query(func.avg(Classification.confidence_score))
        .filter(Classification.confidence_score.isnot(None))
        .filter(Classification.classified_at.isnot(None))
        .filter(func.date(Classification.classified_at) >= start_date)
        .filter(func.date(Classification.classified_at) <= end_date)
        .scalar()
    )
    return round(float(result), 4) if result else 0.0


TRAINING_DATA_DIR = ROOT_DIR / "data" / "retrain"
TRAINING_DATA_TRAIN = TRAINING_DATA_DIR / "train"
TRAINING_DATA_VAL = TRAINING_DATA_DIR / "val"
DEFAULT_BASE_DATA_DIR = ROOT_DIR / "data" / "processed"
UNIFIED_SCRIPT = ROOT_DIR / "train_retrain.py"

MODEL_SCRIPT_MAP = {
    "resnet50": {
        "arch": "resnet50",
        "script": UNIFIED_SCRIPT,
        "model_path": ROOT_DIR / "models" / "best_model_resnet50.pth",
        "metrics_path": ROOT_DIR / "models" / "metrics_resnet50.json",
        "name_prefix": "retrain_resnet50",
    },
    "resnet101": {
        "arch": "resnet101",
        "script": UNIFIED_SCRIPT,
        "model_path": ROOT_DIR / "models" / "best_model_resnet101.pth",
        "metrics_path": ROOT_DIR / "models" / "metrics_resnet101.json",
        "name_prefix": "retrain_resnet101",
    },
    "efficientnet_b0": {
        "arch": "efficientnet_b0",
        "script": UNIFIED_SCRIPT,
        "model_path": ROOT_DIR / "models" / "best_model_efficientnet_b0.pth",
        "metrics_path": ROOT_DIR / "models" / "metrics_efficientnet_b0.json",
        "name_prefix": "retrain_efficientnet_b0",
    },
    "efficientnet_b3": {
        "arch": "efficientnet_b3",
        "script": UNIFIED_SCRIPT,
        "model_path": ROOT_DIR / "models" / "best_model_efficientnet_b3.pth",
        "metrics_path": ROOT_DIR / "models" / "metrics_efficientnet_b3.json",
        "name_prefix": "retrain_efficientnet_b3",
    },
    "densenet121": {
        "arch": "densenet121",
        "script": UNIFIED_SCRIPT,
        "model_path": ROOT_DIR / "models" / "best_model_densenet121.pth",
        "metrics_path": ROOT_DIR / "models" / "metrics_densenet121.json",
        "name_prefix": "retrain_densenet121",
    },
}


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _normalize_role(role_value: Optional[str]) -> Optional[str]:
    if role_value is None:
        return None
    return ROLE_MAP.get(role_value.strip().lower())


def _decimal_to_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_delta(current: Optional[object], reference: Optional[object]) -> Optional[float]:
    current_val = _decimal_to_float(current)
    reference_val = _decimal_to_float(reference)
    if current_val is None or reference_val is None:
        return None
    return round(current_val - reference_val, 6)


def _build_model_summary(
    model: AIModel,
    active_reference: Optional[AIModel],
    recommended_id: Optional[int],
) -> AIModelSummaryResponse:
    return AIModelSummaryResponse(
        id=model.id,
        model_name=model.model_name,
        task_type=model.model_type,
        version=model.version,
        accuracy=_decimal_to_float(model.accuracy),
        f1_score=_decimal_to_float(model.f1_score),
        precision_score=_decimal_to_float(model.precision_score),
        recall_score=_decimal_to_float(model.recall_score),
        inference_time_s=_decimal_to_float(model.inference_time_s),
        status="active" if model.is_active else "inactive",
        is_active=bool(model.is_active),
        is_recommended=model.id == recommended_id,
        delta_acc=_compute_delta(model.accuracy, active_reference.accuracy if active_reference else None),
        delta_f1=_compute_delta(model.f1_score, active_reference.f1_score if active_reference else None),
        delta_time=_compute_delta(model.inference_time_s, active_reference.inference_time_s if active_reference else None),
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def _recommended_per_task(db: Session) -> Dict[str, int]:
    rows = (
        db.query(AIModel)
        .order_by(
            AIModel.model_type.asc(),
            AIModel.f1_score.desc().nullslast(),
            AIModel.accuracy.desc().nullslast(),
        )
        .all()
    )
    mapping: Dict[str, int] = {}
    for row in rows:
        if row.model_type not in mapping:
            mapping[row.model_type] = row.id
    return mapping


def _active_per_task(db: Session) -> Dict[str, AIModel]:
    rows = db.query(AIModel).filter(AIModel.is_active.is_(True)).all()
    return {row.model_type: row for row in rows}


def _ensure_training_dirs() -> None:
    for split in (TRAINING_DATA_TRAIN, TRAINING_DATA_VAL):
        for cls in ("gram_negative", "gram_positive"):
            (split / cls).mkdir(parents=True, exist_ok=True)


def _clear_training_dirs() -> None:
    if TRAINING_DATA_DIR.exists():
        shutil.rmtree(TRAINING_DATA_DIR)
    _ensure_training_dirs()


def _copy_dir_images(src_dir: Path, dst_dir: Path) -> int:
    if not src_dir.exists():
        return 0
    copied = 0
    for cls in ("gram_negative", "gram_positive"):
        src_cls = src_dir / cls
        dst_cls = dst_dir / cls
        dst_cls.mkdir(parents=True, exist_ok=True)
        if not src_cls.exists():
            continue
        for file_path in src_cls.iterdir():
            if file_path.is_file():
                target = dst_cls / file_path.name
                if target.exists():
                    target = dst_cls / f"base_{int(time.time() * 1000)}_{file_path.name}"
                shutil.copy2(file_path, target)
                copied += 1
    return copied


def _label_to_class_dir(gram_label: Optional[str]) -> Optional[str]:
    if not gram_label:
        return None
    if gram_label == "Positif":
        return "gram_positive"
    if gram_label == "Negatif":
        return "gram_negative"
    return None


def _build_retrain_dataset_from_db(db: Session, val_ratio: float = 0.2) -> Dict[str, int]:
    _clear_training_dirs()

    # Copy base dataset if it exists, otherwise skip gracefully
    base_train = _copy_dir_images(DEFAULT_BASE_DATA_DIR / "train", TRAINING_DATA_TRAIN) if DEFAULT_BASE_DATA_DIR.exists() else 0
    base_val = _copy_dir_images(DEFAULT_BASE_DATA_DIR / "val", TRAINING_DATA_VAL) if DEFAULT_BASE_DATA_DIR.exists() else 0

    rows = (
        db.query(Classification)
        .filter(Classification.validation_gram.isnot(None))
        .order_by(Classification.id.asc())
        .all()
    )

    added_train = 0
    added_val = 0
    # Track per-class counts so each class gets at least 1 val image
    class_counts = {}
    for idx, row in enumerate(rows, start=1):
        cls_name = _label_to_class_dir(row.validation_gram)
        if not cls_name:
            continue
        if not row.image_path:
            continue

        src = Path(row.image_path)
        if not src.is_absolute():
            src = (ROOT_DIR / row.image_path).resolve()
        if not src.exists() or not src.is_file():
            continue

        cls_count = class_counts.get(cls_name, 0) + 1
        class_counts[cls_name] = cls_count

        # Ensure first image of each class goes to val
        use_val = (idx % 100) < int(val_ratio * 100) or cls_count == 1
        split_dir = TRAINING_DATA_VAL if use_val else TRAINING_DATA_TRAIN
        dst_dir = split_dir / cls_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        stem = src.stem
        suffix = src.suffix or ".jpg"
        dst = dst_dir / f"crop_{row.id}_{stem}{suffix}"
        try:
            shutil.copy2(src, dst)
            if use_val:
                added_val += 1
            else:
                added_train += 1
        except Exception:
            continue

    return {
        "base_train": base_train,
        "base_val": base_val,
        "crop_train": added_train,
        "crop_val": added_val,
        "total_train": base_train + added_train,
        "total_val": base_val + added_val,
    }


def _resolve_training_script(model: AIModel) -> Dict[str, Path]:
    key = (model.model_name or "").strip().lower()
    if "resnet50" in key:
        return MODEL_SCRIPT_MAP["resnet50"]
    if "resnet101" in key:
        return MODEL_SCRIPT_MAP["resnet101"]
    if "efficientnet-b0" in key or "efficientnet_b0" in key:
        return MODEL_SCRIPT_MAP["efficientnet_b0"]
    if "efficientnet-b3" in key or "efficientnet_b3" in key:
        return MODEL_SCRIPT_MAP["efficientnet_b3"]
    if "densenet121" in key or "dense net121" in key:
        return MODEL_SCRIPT_MAP["densenet121"]
    raise HTTPException(
        status_code=400,
        detail="Retrain saat ini mendukung ResNet50, ResNet101, EfficientNet-B0, EfficientNet-B3, dan DenseNet121",
    )


def _supports_retrain(model: AIModel) -> bool:
    key = (model.model_name or "").strip().lower()
    return (
        ("resnet50" in key)
        or ("resnet101" in key)
        or ("efficientnet-b0" in key)
        or ("efficientnet_b0" in key)
        or ("efficientnet-b3" in key)
        or ("efficientnet_b3" in key)
        or ("densenet121" in key)
    )


def _run_training_job(
    job_id: int,
    script: Path,
    model_output: Path,
    metrics_output: Path,
    name_prefix: str,
    epochs_head: int,
    epochs_ft: int,
    batch_size: int,
    version_label: Optional[str] = None,
) -> None:
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        def _save_job_state() -> None:
            try:
                db.add(job)
                db.commit()
            except Exception:
                db.rollback()
                raise

        job = db.query(ModelTrainingStatus).filter(ModelTrainingStatus.id == job_id).first()
        if not job:
            return

        # Generate unique filenames so retrained models don't overwrite pre-trained ones
        timestamp = get_local_now().strftime('%Y%m%d_%H%M%S')
        model_output = model_output.parent / f"{model_output.stem}_retrain_{timestamp}{model_output.suffix}"
        metrics_output = metrics_output.parent / f"{metrics_output.stem}_retrain_{timestamp}{metrics_output.suffix}"

        cmd = [
            sys.executable,
            "-u",
            str(script),
            "--data",
            str(TRAINING_DATA_DIR),
            "--epochs-head",
            str(epochs_head),
            "--epochs-ft",
            str(epochs_ft),
            "--batch-size",
            str(batch_size),
            "--save-model",
            str(model_output),
            "--save-metrics",
            str(metrics_output),
            "--name",
            f"{name_prefix}_{get_local_now().strftime('%Y%m%d_%H%M%S')}",
            "--arch",
            name_prefix.replace("retrain_", ""),
        ]

        print(f"\n[RETRAIN #{job_id}] Starting training...")
        print(f"[RETRAIN #{job_id}] CWD: {ROOT_DIR}")
        print(f"[RETRAIN #{job_id}] CMD: {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )

        logs: List[str] = []
        total_epochs = max(1, epochs_head + epochs_ft)
        if job.total_epochs is None:
            job.total_epochs = total_epochs
        job.progress = max(float(job.progress or 0.0), 1.0)
        _save_job_state()

        last_flush = time.time()

        if proc.stdout:
            for line in proc.stdout:
                text_line = line.strip()
                if text_line:
                    print(f"[RETRAIN #{job_id}] {text_line}")
                    logs.append(text_line)
                    if len(logs) > 40:
                        logs = logs[-40:]

                epoch_match = re.search(r"['\"]?epoch['\"]?\s*:\s*(\d+)", text_line)
                if epoch_match:
                    parsed_epoch = int(epoch_match.group(1))
                    if parsed_epoch > 0:
                        job.current_epoch = min(parsed_epoch, total_epochs)
                    else:
                        job.current_epoch = min((job.current_epoch or 0) + 1, total_epochs)
                    job.progress = min(99.0, (job.current_epoch / total_epochs) * 100.0)
                    job.error_message = "\n".join(logs[-20:])
                    _save_job_state()
                    last_flush = time.time()
                elif text_line and (time.time() - last_flush) >= 2.0:
                    job.error_message = "\n".join(logs[-20:])
                    _save_job_state()
                    last_flush = time.time()

        ret = proc.wait()

        job.end_time = get_local_now()
        job.progress = 100.0

        if ret == 0:
            job.status = "COMPLETED"
            job.error_message = "\n".join(logs[-20:]) if logs else "Training completed"
            _save_job_state()
            print(f"[RETRAIN #{job_id}] ✅ COMPLETED — exit code {ret}")

            # Insert new model version + run benchmark
            try:
                from app.main import benchmark_model

                model_row = db.query(AIModel).filter(AIModel.id == job.model_id).first()
                base_name = model_row.model_name if model_row else "retrained"

                # ---- Auto-versioning ----
                # Find existing versions for this model, parse semver (vMAJOR.MINOR)
                existing = (
                    db.query(AIModel.version)
                    .filter(AIModel.model_name == base_name)
                    .all()
                )
                latest_major = 0
                latest_minor = 0
                for (ver,) in existing:
                    m = re.match(r"v?(\d+)\.(\d+)", str(ver or ""))
                    if m:
                        major = int(m.group(1))
                        minor = int(m.group(2))
                        if major > latest_major or (major == latest_major and minor > latest_minor):
                            latest_major = major
                            latest_minor = minor

                if version_label:
                    # Minor bump: v1.0 → v1.1
                    new_version = f"v{latest_major}.{latest_minor + 1}" if latest_major > 0 else "v1.1"
                else:
                    # Major bump: v1.0 → v2.0
                    new_version = f"v{latest_major + 1}.0" if latest_major > 0 else "v1.0"

                new_model = AIModel(
                    model_name=base_name,
                    model_type=model_row.model_type if model_row else "Gram Classification",
                    version=new_version,
                    model_file_path=str(model_output),
                    is_active=False,
                )
                db.add(new_model)
                db.flush()

                # Try to load and benchmark the new model
                from app.main import load_all_models as reload_models
                from app.main import MODEL_REGISTRY, MODELS, MODELS_LOADED, DEVICE, _load_state_dict

                config = MODEL_REGISTRY.get(base_name)
                if config and model_output.exists():
                    try:
                        model_instance = config["class"]().to(DEVICE)
                        sd = _load_state_dict(model_output)
                        model_instance.load_state_dict(sd, strict=True)
                        model_instance.eval()
                        MODELS[base_name] = model_instance
                        MODELS_LOADED[base_name] = True

                        test_path = os.environ.get("TEST_DATA_PATH")
                        if test_path and Path(test_path).exists():
                            metrics = benchmark_model(base_name, test_path)
                            new_model.accuracy = metrics["accuracy"]
                            new_model.precision_score = metrics["precision"]
                            new_model.recall_score = metrics["recall"]
                            new_model.f1_score = metrics["f1"]
                            new_model.inference_time_s = metrics["inference_time_s"]
                            job.error_message = (job.error_message or "") + f"\nBenchmark: acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}"
                    except Exception as bench_err:
                        job.error_message = (job.error_message or "") + f"\nBenchmark failed: {bench_err}"
                    finally:
                        db.add(new_model)
                        db.flush()
                        _save_job_state()
            except Exception as seed_err:
                job.error_message = (job.error_message or "") + f"\nPost-train seeding failed: {seed_err}"
                _save_job_state()
        else:
            job.status = "FAILED"
            job.error_message = "\n".join(logs[-20:]) if logs else "Training failed"
            _save_job_state()
            print(f"[RETRAIN #{job_id}] ❌ FAILED — exit code {ret}")
            print(f"[RETRAIN #{job_id}] Last logs:\n{job.error_message}")
    except Exception as exc:
        print(f"[RETRAIN #{job_id}] ❌ EXCEPTION: {exc}")
        import traceback
        traceback.print_exc()
        db.rollback()
        job = db.query(ModelTrainingStatus).filter(ModelTrainingStatus.id == job_id).first()
        if job:
            try:
                job.status = "FAILED"
                job.end_time = get_local_now()
                job.error_message = f"Retrain exception: {exc}"
                db.add(job)
                db.commit()
            except Exception:
                db.rollback()
        return
    finally:
        db.close()


@router.get("/users", response_model=PaginatedResponse[UserResponseSchema])
def list_users(
    search: Optional[str] = Query(None, description="Search by name, username, or email"),
    role: Optional[str] = Query(None, description="Filter by role"),
    status: Optional[str] = Query(None, description="Filter by status: active/inactive"),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    query = db.query(User)

    if search:
        like = f"%{search}%"
        query = query.filter(
            or_(
                User.full_name.ilike(like),
                User.username.ilike(like),
                User.email.ilike(like),
            )
        )

    if role:
        query = query.filter(User.role == role)

    if status:
        normalized = status.lower()
        if normalized not in {"active", "inactive"}:
            raise HTTPException(status_code=400, detail="status harus active atau inactive")
        query = query.filter(User.is_active.is_(normalized == "active"))

    query = query.order_by(User.created_at.desc())
    users, meta = paginate_query(query, page, per_page)
    return PaginatedResponse[UserResponseSchema](data=users, meta=meta)


@router.get("/users/{user_id}", response_model=UserResponseSchema)
def get_user_detail(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")
    return user


@router.post("/users", response_model=UserResponseSchema, status_code=201)
def create_user(payload: UserCreateRequest, db: Session = Depends(get_db)):
    normalized_role = _normalize_role(payload.role)
    if not normalized_role:
        raise HTTPException(status_code=400, detail="Role tidak valid")

    existing_username = db.query(User).filter(User.username == payload.username).first()
    if existing_username:
        raise HTTPException(status_code=400, detail="Username sudah digunakan")

    if payload.email:
        existing_email = db.query(User).filter(User.email == payload.email).first()
        if existing_email:
            raise HTTPException(status_code=400, detail="Email sudah digunakan")

    hashed = _hash_password(payload.password)
    db_user = User(
        full_name=payload.full_name,
        username=payload.username,
        email=payload.email,
        role=normalized_role,
        hashed_password=hashed,
        is_active=payload.is_active,
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@router.put("/users/{user_id}", response_model=UserResponseSchema)
def update_user(user_id: int, payload: UserUpdateRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")

    if payload.username and payload.username != user.username:
        exists_username = db.query(User).filter(User.username == payload.username).first()
        if exists_username:
            raise HTTPException(status_code=400, detail="Username sudah digunakan")
        user.username = payload.username

    if payload.email and payload.email != user.email:
        exists_email = db.query(User).filter(User.email == payload.email).first()
        if exists_email:
            raise HTTPException(status_code=400, detail="Email sudah digunakan")
        user.email = payload.email
    elif payload.email == "":
        user.email = None

    if payload.full_name is not None:
        user.full_name = payload.full_name

    if payload.role is not None:
        normalized_role = _normalize_role(payload.role)
        if not normalized_role:
            raise HTTPException(status_code=400, detail="Role tidak valid")
        user.role = normalized_role

    if payload.is_active is not None:
        user.is_active = payload.is_active

    new_password = payload.new_password or payload.password
    if new_password:
        user.hashed_password = _hash_password(new_password)

    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.delete("/users/{user_id}", status_code=204)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")

    db.delete(user)
    db.commit()


@router.get("/users/roles", response_model=RoleListResponse)
def get_roles():
    return RoleListResponse(roles=VALID_ROLES)


@router.get("/models", response_model=PaginatedResponse[AIModelSummaryResponse])
def get_models(
    task_type: Optional[str] = Query(None, description="Filter berdasarkan task (contoh: Detection/ Classification)"),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    query = db.query(AIModel)
    if task_type:
        query = query.filter(AIModel.model_type == task_type)

    query = query.order_by(AIModel.created_at.desc())
    models, meta = paginate_query(query, page, per_page)

    active_map = _active_per_task(db)
    recommended_map = _recommended_per_task(db)

    summaries = [
        _build_model_summary(model, active_map.get(model.model_type), recommended_map.get(model.model_type))
        for model in models
    ]

    return PaginatedResponse[AIModelSummaryResponse](data=summaries, meta=meta)


@router.get("/models/active", response_model=List[ActiveModelResponse])
def get_active_models(db: Session = Depends(get_db)):
    active_map = _active_per_task(db)
    recommended_map = _recommended_per_task(db)
    task_types = [row[0] for row in db.query(AIModel.model_type).distinct().all()]

    responses: List[ActiveModelResponse] = []
    for task in task_types:
        model = active_map.get(task)
        summary = (
            _build_model_summary(model, model, recommended_map.get(task))
            if model
            else None
        )
        responses.append(ActiveModelResponse(task_type=task, model=summary))

    return responses


@router.get("/models/best", response_model=List[BestModelResponse])
def get_best_models(db: Session = Depends(get_db)):
    recommended_map = _recommended_per_task(db)
    task_types = [row[0] for row in db.query(AIModel.model_type).distinct().all()]
    active_map = _active_per_task(db)

    result: List[BestModelResponse] = []
    for task in task_types:
        recommended_id = recommended_map.get(task)
        model = db.query(AIModel).filter(AIModel.id == recommended_id).first() if recommended_id else None
        summary = (
            _build_model_summary(model, active_map.get(task), recommended_id)
            if model
            else None
        )
        result.append(BestModelResponse(task_type=task, model=summary))

    return result


def _get_or_create_retrain_config(db: Session) -> ModelRetrainConfig:
    config = db.query(ModelRetrainConfig).order_by(ModelRetrainConfig.id.asc()).first()
    if config:
        return config
    config = ModelRetrainConfig()
    db.add(config)
    db.commit()
    db.refresh(config)
    return config


def _calculate_validated_since_last_train(db: Session) -> int:
    last_completed_end = (
        db.query(func.max(ModelTrainingStatus.end_time))
        .filter(ModelTrainingStatus.status == "COMPLETED")
        .scalar()
    )

    query = db.query(func.count(Classification.id)).filter(Classification.validation_gram.isnot(None))
    if last_completed_end:
        query = query.filter(Classification.updated_at >= last_completed_end)
    return int(query.scalar() or 0)


@router.get("/models/retrain-config", response_model=RetrainConfigResponse)
def get_retrain_config(db: Session = Depends(get_db)):
    config = _get_or_create_retrain_config(db)
    validated_count = _calculate_validated_since_last_train(db)

    # update cached counter for transparency
    config.validated_data_since_last_train = validated_count
    db.add(config)
    db.commit()
    db.refresh(config)

    return RetrainConfigResponse(
        auto_retrain_enabled=config.auto_retrain_enabled,
        trigger_count=config.trigger_count,
        validated_data_since_last_train=config.validated_data_since_last_train,
    )


@router.patch("/models/retrain-config", response_model=RetrainConfigResponse)
def update_retrain_config(payload: RetrainConfigUpdateRequest, db: Session = Depends(get_db)):
    config = _get_or_create_retrain_config(db)

    if payload.auto_retrain_enabled is not None:
        config.auto_retrain_enabled = payload.auto_retrain_enabled

    if payload.trigger_count is not None:
        config.trigger_count = payload.trigger_count

    db.add(config)
    db.commit()
    db.refresh(config)

    return RetrainConfigResponse(
        auto_retrain_enabled=config.auto_retrain_enabled,
        trigger_count=config.trigger_count,
        validated_data_since_last_train=config.validated_data_since_last_train,
    )


@router.get("/models/training-jobs", response_model=PaginatedResponse[TrainingJobResponse])
def get_training_jobs(
    status: Optional[str] = Query(None, description="Filter status: TRAINING/COMPLETED/FAILED/IDLE"),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    query = db.query(ModelTrainingStatus)

    if status:
        normalized = status.strip().upper()
        valid_statuses = {"TRAINING", "COMPLETED", "FAILED", "IDLE"}
        if normalized not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail="status harus salah satu: TRAINING, COMPLETED, FAILED, IDLE",
            )
        query = query.filter(ModelTrainingStatus.status == normalized)

    query = query.order_by(ModelTrainingStatus.start_time.desc().nullslast())
    jobs, meta = paginate_query(query, page, per_page)

    model_ids = {job.model_id for job in jobs if job.model_id is not None}
    model_name_map: Dict[int, str] = {}
    if model_ids:
        model_rows = db.query(AIModel.id, AIModel.model_name).filter(AIModel.id.in_(model_ids)).all()
        model_name_map = {row.id: row.model_name for row in model_rows}

    responses: List[TrainingJobResponse] = []
    for job in jobs:
        responses.append(
            TrainingJobResponse(
                job_id=job.id,
                model_id=job.model_id,
                model_name=model_name_map.get(job.model_id) if job.model_id is not None else None,
                status=job.status,
                progress_percent=_decimal_to_float(job.progress),
                started_at=job.start_time,
                finished_at=job.end_time,
                logs_summary=job.error_message,
            )
        )

    return PaginatedResponse[TrainingJobResponse](data=responses, meta=meta)


@router.patch("/models/training-jobs/{job_id}/cancel", response_model=MessageResponse)
def cancel_training_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(ModelTrainingStatus).filter(ModelTrainingStatus.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job tidak ditemukan")

    if job.status != "TRAINING":
        raise HTTPException(status_code=400, detail=f"Job status is '{job.status}', only TRAINING jobs can be cancelled")

    job.status = "CANCELLED"
    job.progress = 0.0
    job.end_time = get_local_now()
    job.error_message = "Cancelled by user"
    db.add(job)
    db.commit()

    return MessageResponse(message=f"Training job #{job_id} berhasil dibatalkan.")


@router.delete("/models/training-jobs/{job_id}", response_model=MessageResponse)
def delete_training_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(ModelTrainingStatus).filter(ModelTrainingStatus.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job tidak ditemukan")

    db.delete(job)
    db.commit()
    return MessageResponse(message=f"Training job #{job_id} berhasil dihapus.")


@router.post("/models/retrain", response_model=RetrainStartResponse)
def start_retrain(payload: RetrainStartRequest, db: Session = Depends(get_db)):
    model = db.query(AIModel).filter(AIModel.id == payload.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model tidak ditemukan")

    running_job = (
        db.query(ModelTrainingStatus)
        .filter(ModelTrainingStatus.status == "TRAINING")
        .first()
    )
    if running_job:
        raise HTTPException(
            status_code=409,
            detail=f"Masih ada job training berjalan (job_id={running_job.id})",
        )

    script_conf = _resolve_training_script(model)

    stats = _build_retrain_dataset_from_db(db, val_ratio=float(payload.val_ratio_crops or 0.2))
    print(f"[RETRAIN] Dataset: train={stats['total_train']} (base={stats['base_train']}, crop={stats['crop_train']}), val={stats['total_val']} (base={stats['base_val']}, crop={stats['crop_val']})")
    if stats["total_train"] == 0 or stats["total_val"] == 0:
        raise HTTPException(
            status_code=400,
            detail="Dataset retrain kosong. Pastikan data awal tersedia dan/atau crop tervalidasi ada.",
        )

    job = ModelTrainingStatus(
        model_id=model.id,
        status="TRAINING",
        progress=0.0,
        start_time=get_local_now(),
        current_epoch=0,
        total_epochs=int((payload.epochs_head or 10) + (payload.epochs_ft or 30)),
        error_message=(
            f"Dataset built: base_train={stats['base_train']}, base_val={stats['base_val']}, "
            f"crop_train={stats['crop_train']}, crop_val={stats['crop_val']}"
        ),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    thread = threading.Thread(
        target=_run_training_job,
        kwargs={
            "job_id": job.id,
            "script": script_conf["script"],
            "model_output": script_conf["model_path"],
            "metrics_output": script_conf["metrics_path"],
            "name_prefix": script_conf["name_prefix"],
            "epochs_head": int(payload.epochs_head or 10),
            "epochs_ft": int(payload.epochs_ft or 30),
            "batch_size": int(payload.batch_size or 32),
            "version_label": payload.version_label,
        },
        daemon=True,
    )
    thread.start()

    return RetrainStartResponse(
        job_id=job.id,
        status=job.status,
        message=(
            f"Retrain dimulai untuk model_id={model.id}. "
            f"Dataset train={stats['total_train']} | val={stats['total_val']}."
        ),
    )


@router.patch("/models/{model_id}/activate", response_model=MessageResponse)
def activate_model(model_id: int, db: Session = Depends(get_db)):
    model = db.query(AIModel).filter(AIModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model tidak ditemukan")

    # Deactivate all other models of the same task type
    db.query(AIModel).filter(
        AIModel.model_type == model.model_type,
        AIModel.id != model_id,
    ).update({"is_active": False}, synchronize_session=False)

    # Activate the selected model
    model.is_active = True
    db.commit()

    return MessageResponse(message=f"Model {model.model_name} v{model.version} berhasil diaktifkan.")




@router.post("/models/benchmark-all", response_model=MessageResponse)
def benchmark_all_models(
    test_data_path: Optional[str] = Query(None),
    sample_count: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    """Run benchmark on ALL classification models and update their metrics in the DB."""
    from app.main import benchmark_model

    models = db.query(AIModel).filter(AIModel.model_type != "Detection").all()
    if not models:
        return MessageResponse(message="Tidak ada model klasifikasi yang terdaftar.")

    test_path = test_data_path or os.environ.get("TEST_DATA_PATH")
    if not test_path or not Path(test_path).exists():
        raise HTTPException(status_code=400, detail="TEST_DATA_PATH tidak ditemukan. Pastikan folder test tersedia.")

    results = []
    errors = []
    for m in models:
        try:
            metrics = benchmark_model(m.model_name, test_data_path=test_path, sample_count=sample_count)
            m.accuracy = metrics["accuracy"]
            m.precision_score = metrics["precision"]
            m.recall_score = metrics["recall"]
            m.f1_score = metrics["f1"]
            m.inference_time_s = metrics["inference_time_s"]
            db.add(m)
            results.append(f"{m.model_name}: acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")
        except Exception as e:
            errors.append(f"{m.model_name}: {e}")

    db.commit()

    msg = f"Berhasil benchmark {len(results)} model."
    if errors:
        msg += f" Gagal: {len(errors)} ({'; '.join(errors[:3])})"
    return MessageResponse(message=msg)

@router.post("/models/benchmark-active", response_model=BenchmarkResponse)
def benchmark_active_model(
    test_data_path: Optional[str] = Query(None),
    sample_count: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    """Benchmark model klasifikasi yang sedang aktif."""
    from app.main import benchmark_model

    active = db.query(AIModel).filter(AIModel.model_type != "Detection", AIModel.is_active.is_(True)).first()
    if not active:
        raise HTTPException(status_code=404, detail="Tidak ada model klasifikasi aktif.")

    test_path = test_data_path or os.environ.get("TEST_DATA_PATH")
    if not test_path or not Path(test_path).exists():
        raise HTTPException(status_code=400, detail="TEST_DATA_PATH tidak ditemukan.")

    metrics = benchmark_model(
        active.model_name,
        test_data_path=test_path,
        sample_count=sample_count,
    )
    active.accuracy = metrics["accuracy"]
    active.precision_score = metrics["precision"]
    active.recall_score = metrics["recall"]
    active.f1_score = metrics["f1"]
    active.inference_time_s = metrics["inference_time_s"]
    db.add(active)
    db.commit()

    return BenchmarkResponse(
        model_id=active.id,
        model_name=active.model_name,
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        inference_time_s=metrics["inference_time_s"],
        num_samples=metrics["num_samples"],
    )



@router.post("/models/{model_id}/benchmark", response_model=BenchmarkResponse)
def benchmark_single_model(
    model_id: int,
    test_data_path: Optional[str] = Query(None),
    sample_count: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    """Run benchmark on an existing model and update its metrics in the DB."""
    from app.main import benchmark_model

    ai_model = db.query(AIModel).filter(AIModel.id == model_id).first()
    if not ai_model:
        raise HTTPException(status_code=404, detail="Model tidak ditemukan")

    if ai_model.model_type == "Detection":
        raise HTTPException(status_code=400, detail="Benchmark hanya tersedia untuk model klasifikasi")

    metrics = benchmark_model(
        ai_model.model_name,
        test_data_path=test_data_path,
        sample_count=sample_count,
    )
    ai_model.accuracy = metrics["accuracy"]
    ai_model.precision_score = metrics["precision"]
    ai_model.recall_score = metrics["recall"]
    ai_model.f1_score = metrics["f1"]
    ai_model.inference_time_s = metrics["inference_time_s"]
    db.add(ai_model)
    db.commit()

    return BenchmarkResponse(
        model_id=ai_model.id,
        model_name=ai_model.model_name,
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        inference_time_s=metrics["inference_time_s"],
        num_samples=metrics["num_samples"],
    )




@router.post("/models/yolo-benchmark", response_model=YoloBenchmarkResponse)
def benchmark_yolo_model(
    model_id: Optional[int] = Query(None, description="ID model YOLO yang akan di-benchmark (opsional, default: model aktif)"),
    test_data_path: Optional[str] = Query(None, description="Path ke dataset YOLO"),
    db: Session = Depends(get_db),
):
    """Benchmark model YOLO (detection) menggunakan ultralytics val."""
    from app.main import benchmark_yolo

    if model_id:
        yolo_model = db.query(AIModel).filter(AIModel.id == model_id, AIModel.model_type == "Detection").first()
        if not yolo_model:
            raise HTTPException(status_code=404, detail=f"Model deteksi dengan ID {model_id} tidak ditemukan.")
    else:
        yolo_model = db.query(AIModel).filter(AIModel.model_type == "Detection", AIModel.is_active.is_(True)).first()
        if not yolo_model:
            yolo_model = db.query(AIModel).filter(AIModel.model_type == "Detection").first()
    if not yolo_model:
        raise HTTPException(status_code=404, detail="Model deteksi YOLO tidak ditemukan di database.")

    try:
        model_file = yolo_model.model_file_path if yolo_model.model_file_path else None
        metrics = benchmark_yolo(test_data_path=test_data_path, model_path=model_file)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal benchmark YOLO: {e}")

    yolo_model.accuracy = metrics["map50"]
    yolo_model.precision_score = metrics["precision"]
    yolo_model.recall_score = metrics["recall"]
    yolo_model.f1_score = metrics["map50_95"]
    db.add(yolo_model)
    db.commit()

    return YoloBenchmarkResponse(
        model_id=yolo_model.id,
        model_name=yolo_model.model_name,
        map50=round(metrics["map50"] * 100, 2),
        map50_95=round(metrics["map50_95"] * 100, 2),
        precision=metrics["precision"],
        recall=metrics["recall"],
        num_samples=metrics["num_samples"],
    )




@router.post("/models/yolo-benchmark-all", response_model=YoloBenchmarkAllResponse)
def benchmark_all_yolo_models(
    test_data_path: Optional[str] = Query(None, description="Path ke dataset YOLO"),
    db: Session = Depends(get_db),
):
    """Benchmark semua model YOLO (detection) dan update metriknya di DB."""
    from app.main import benchmark_yolo

    yolo_models = db.query(AIModel).filter(AIModel.model_type == "Detection").all()
    if not yolo_models:
        return YoloBenchmarkAllResponse(message="Tidak ada model deteksi YOLO yang terdaftar.")

    results = []
    errors = []
    for m in yolo_models:
        try:
            model_file = m.model_file_path if m.model_file_path else None
            metrics = benchmark_yolo(test_data_path=test_data_path, model_path=model_file)
            m.accuracy = metrics["map50"]
            m.precision_score = metrics["precision"]
            m.recall_score = metrics["recall"]
            m.f1_score = metrics["map50_95"]
            db.add(m)
            results.append(YoloBenchmarkResponse(
                model_id=m.id,
                model_name=m.model_name,
                map50=metrics["map50"],
                map50_95=metrics["map50_95"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                num_samples=metrics["num_samples"],
            ))
        except Exception as e:
            errors.append(f"{m.model_name}: {e}")

    db.commit()

    msg = f"Berhasil benchmark {len(results)} model YOLO."
    if errors:
        msg += f" Gagal: {len(errors)}."
    return YoloBenchmarkAllResponse(results=results, errors=errors, message=msg)


@router.post("/models/upload", response_model=ModelUploadResponse)
async def upload_model(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    model_type: str = Form(...),
    version: str = Form("1.0"),
    db: Session = Depends(get_db),
):
    """Upload a new model file (.pth for classification, .pt for detection)."""
    import shutil

    allowed_ext = {".pth", ".pt"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Format file tidak didukung. Gunakan: {', '.join(allowed_ext)}")

    models_dir = Path(__file__).resolve().parents[2] / "models"
    models_dir.mkdir(exist_ok=True)

    safe_name = re.sub(r"[^\w.\-]", "_", file.filename)
    dest = models_dir / f"uploaded_{int(time.time())}_{safe_name}"

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    db_model = AIModel(
        model_name=model_name,
        model_type=model_type,
        version=version,
        model_file_path=str(dest),
        is_active=False,
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)

    benchmark_msg = ""
    if model_type != "Detection":
        try:
            from app.main import benchmark_model
            test_path = os.environ.get("TEST_DATA_PATH")
            if test_path and Path(test_path).exists():
                metrics = benchmark_model(model_name, test_data_path=test_path)
                db_model.accuracy = metrics["accuracy"]
                db_model.precision_score = metrics["precision"]
                db_model.recall_score = metrics["recall"]
                db_model.f1_score = metrics["f1"]
                db_model.inference_time_s = metrics["inference_time_s"]
                db.add(db_model)
                db.commit()
                db.refresh(db_model)
                benchmark_msg = f" Benchmark: akurasi={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}."
        except Exception as bench_err:
            benchmark_msg = f" Benchmark gagal: {bench_err}"

    return ModelUploadResponse(
        id=db_model.id,
        model_name=db_model.model_name,
        model_type=db_model.model_type,
        version=db_model.version,
        model_file_path=db_model.model_file_path,
        is_active=db_model.is_active,
        message="Model berhasil diunggah. Aktifkan melalui endpoint aktivasi." + benchmark_msg,
    )


@router.delete("/models/{model_id}", response_model=MessageResponse)
def delete_model(model_id: int, db: Session = Depends(get_db)):
    model = db.query(AIModel).filter(AIModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model tidak ditemukan")

    db.delete(model)
    db.commit()

    return MessageResponse(message=f"Model {model.model_name} v{model.version} berhasil dihapus.")


@router.get("/models/retrain/options", response_model=List[RetrainModelOptionResponse])
def get_retrain_model_options(db: Session = Depends(get_db)):
    models = db.query(AIModel).order_by(AIModel.created_at.desc()).all()
    return [
        RetrainModelOptionResponse(
            id=m.id,
            model_name=m.model_name,
            version=m.version,
            task_type=m.model_type,
            is_active=bool(m.is_active),
            supports_retrain=_supports_retrain(m),
        )
        for m in models
    ]
