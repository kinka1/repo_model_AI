import hashlib
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
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
    ActiveModelResponse,
    AIModelSummaryResponse,
    BestModelResponse,
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
from app.utils import paginate_query


router = APIRouter(prefix="/api/admin", tags=["Admin Management"])

VALID_ROLES = ["Admin", "Analis", "Dokter"]
ROOT_DIR = Path(__file__).resolve().parents[2]
TRAINING_DATA_DIR = ROOT_DIR / "data" / "retrain"
TRAINING_DATA_TRAIN = TRAINING_DATA_DIR / "train"
TRAINING_DATA_VAL = TRAINING_DATA_DIR / "val"
DEFAULT_BASE_DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_PATH_BY_KEY = {
    "resnet50": ROOT_DIR / "models" / "best_model_resnet50.pth",
    "resnet101": ROOT_DIR / "models" / "best_model_resnet101.pth",
    "efficientnet_b0": ROOT_DIR / "models" / "best_model_efficientnet_b0.pth",
    "efficientnet_b3": ROOT_DIR / "models" / "best_model_efficientnet_b3.pth",
    "densenet121": ROOT_DIR / "models" / "best_model_densenet121.pth",
}
METRICS_PATH_BY_KEY = {
    "resnet50": ROOT_DIR / "models" / "metrics_resnet50.json",
    "resnet101": ROOT_DIR / "models" / "metrics_resnet101.json",
    "efficientnet_b0": ROOT_DIR / "models" / "metrics_efficientnet_b0.json",
    "efficientnet_b3": ROOT_DIR / "models" / "metrics_efficientnet_b3.json",
    "densenet121": ROOT_DIR / "models" / "metrics_densenet121.json",
}

MODEL_SCRIPT_MAP = {
    "resnet50": {
        "script": ROOT_DIR / "train_resnet50_finetune.py",
        "model_path": MODEL_PATH_BY_KEY["resnet50"],
        "metrics_path": METRICS_PATH_BY_KEY["resnet50"],
        "name_prefix": "retrain_resnet50",
    },
    "resnet101": {
        "script": ROOT_DIR / "train_resnet101_finetune.py",
        "model_path": MODEL_PATH_BY_KEY["resnet101"],
        "metrics_path": METRICS_PATH_BY_KEY["resnet101"],
        "name_prefix": "retrain_resnet101",
    },
    "efficientnet_b0": {
        "script": ROOT_DIR / "train_efficientnet_b0_finetune.py",
        "model_path": MODEL_PATH_BY_KEY["efficientnet_b0"],
        "metrics_path": METRICS_PATH_BY_KEY["efficientnet_b0"],
        "name_prefix": "retrain_efficientnet_b0",
    },
    "efficientnet_b3": {
        "script": ROOT_DIR / "train_efficientnet_b3_finetune.py",
        "model_path": MODEL_PATH_BY_KEY["efficientnet_b3"],
        "metrics_path": METRICS_PATH_BY_KEY["efficientnet_b3"],
        "name_prefix": "retrain_efficientnet_b3",
    },
    "densenet121": {
        "script": ROOT_DIR / "train_densenet121_finetune.py",
        "model_path": MODEL_PATH_BY_KEY["densenet121"],
        "metrics_path": METRICS_PATH_BY_KEY["densenet121"],
        "name_prefix": "retrain_densenet121",
    },
}


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


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

    base_train = _copy_dir_images(DEFAULT_BASE_DATA_DIR / "train", TRAINING_DATA_TRAIN)
    base_val = _copy_dir_images(DEFAULT_BASE_DATA_DIR / "val", TRAINING_DATA_VAL)

    rows = (
        db.query(Classification)
        .filter(Classification.validation_gram.isnot(None))
        .order_by(Classification.id.asc())
        .all()
    )

    added_train = 0
    added_val = 0
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

        use_val = (idx % 100) < int(val_ratio * 100)
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
            f"{name_prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        ]

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

        job.end_time = datetime.utcnow()
        job.progress = 100.0

        if ret == 0:
            job.status = "COMPLETED"
            job.error_message = "\n".join(logs[-20:]) if logs else "Training completed"
        else:
            job.status = "FAILED"
            job.error_message = "\n".join(logs[-20:]) if logs else "Training failed"

        _save_job_state()
    except Exception as exc:
        db.rollback()
        job = db.query(ModelTrainingStatus).filter(ModelTrainingStatus.id == job_id).first()
        if job:
            try:
                job.status = "FAILED"
                job.end_time = datetime.utcnow()
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
    if payload.role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail="Role tidak valid")

    existing = (
        db.query(User)
        .filter(or_(User.username == payload.username, User.email == payload.email))
        .first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="Username atau email sudah digunakan")

    hashed = _hash_password(payload.password)
    db_user = User(
        full_name=payload.full_name,
        username=payload.username,
        email=payload.email,
        role=payload.role,
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

    if payload.full_name is not None:
        user.full_name = payload.full_name

    if payload.role is not None:
        if payload.role not in VALID_ROLES:
            raise HTTPException(status_code=400, detail="Role tidak valid")
        user.role = payload.role

    if payload.is_active is not None:
        user.is_active = payload.is_active

    if payload.password:
        user.hashed_password = _hash_password(payload.password)

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
    task_type: Optional[str] = Query(None, description="Filter berdasarkan task (contoh: detection/ classification)"),
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


@router.get("/models/training-jobs", response_model=List[TrainingJobResponse])
def get_training_jobs(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    jobs = (
        db.query(ModelTrainingStatus)
        .order_by(ModelTrainingStatus.start_time.desc().nullslast())
        .limit(limit)
        .all()
    )

    responses: List[TrainingJobResponse] = []
    for job in jobs:
        responses.append(
            TrainingJobResponse(
                job_id=job.id,
                model_id=job.model_id,
                status=job.status,
                progress_percent=_decimal_to_float(job.progress),
                started_at=job.start_time,
                finished_at=job.end_time,
                logs_summary=job.error_message,
            )
        )

    return responses


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
    if stats["total_train"] == 0 or stats["total_val"] == 0:
        raise HTTPException(
            status_code=400,
            detail="Dataset retrain kosong. Pastikan data awal tersedia dan/atau crop tervalidasi ada.",
        )

    job = ModelTrainingStatus(
        model_id=model.id,
        status="TRAINING",
        progress=0.0,
        start_time=datetime.utcnow(),
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
