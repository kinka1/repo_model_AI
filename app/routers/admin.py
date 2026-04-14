import hashlib
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
    RoleListResponse,
    TrainingJobResponse,
    UserCreateRequest,
    UserResponseSchema,
    UserUpdateRequest,
)
from app.utils import paginate_query


router = APIRouter(prefix="/api/admin", tags=["Admin Management"])

VALID_ROLES = ["Admin", "Analis", "Dokter"]


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
