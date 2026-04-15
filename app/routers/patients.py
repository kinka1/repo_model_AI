from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from sqlalchemy import or_, func, and_
import time
from datetime import datetime

from app.database import get_db
from app.models import Patient, Specimen
from app.schemas import PatientCreate, PatientResponse, PatientUpdate, PaginatedResponse
from app.utils import paginate_query
from typing import Optional

router = APIRouter(
    prefix="/api/patients",
    tags=["Patient Management"]
)


def require_admin(request: Request):
    role = getattr(request.state, "role", None)
    if role != "Admin":
        raise HTTPException(status_code=403, detail="Hanya Admin yang boleh mengakses endpoint patient")


def require_admin_or_analis(request: Request):
    role = getattr(request.state, "role", None)
    if role not in {"Admin", "Analis"}:
        raise HTTPException(status_code=403, detail="Hanya Admin atau Analis yang boleh mengakses data patient")

def generate_id_pasien():
    # Simple ID generation: PAS-Timestamp
    return f"PAS-{int(time.time()*1000)}"[:20]

@router.get("", response_model=PaginatedResponse[PatientResponse])
def get_patients(
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    search: Optional[str] = Query(None, description="Search by name or id_pasien"),
    specimen_status: Optional[str] = Query(None, description="Filter by latest specimen status, e.g. pending/waiting_validation/validated"),
    include_no_specimen: bool = Query(True, description="When filtering by specimen_status, include patients without specimen"),
    db: Session = Depends(get_db),
    _role=Depends(require_admin_or_analis),
):
    query = db.query(Patient)
    if search:
        query = query.filter(
            or_(
                Patient.nama_lengkap.ilike(f"%{search}%"),
                Patient.id_pasien.ilike(f"%{search}%")
            )
        )

    if specimen_status:
        latest_specimen_subq = (
            db.query(
                Specimen.patient_id.label("patient_id"),
                func.max(Specimen.uploaded_at).label("max_uploaded_at"),
            )
            .group_by(Specimen.patient_id)
            .subquery()
        )

        latest_status_subq = (
            db.query(
                Specimen.patient_id.label("patient_id"),
                Specimen.status.label("latest_status"),
            )
            .join(
                latest_specimen_subq,
                and_(
                    Specimen.patient_id == latest_specimen_subq.c.patient_id,
                    Specimen.uploaded_at == latest_specimen_subq.c.max_uploaded_at,
                ),
            )
            .subquery()
        )

        query = query.outerjoin(latest_status_subq, latest_status_subq.c.patient_id == Patient.id)

        if include_no_specimen:
            query = query.filter(
                or_(
                    latest_status_subq.c.latest_status == specimen_status,
                    latest_status_subq.c.latest_status.is_(None),
                )
            )
        else:
            query = query.filter(latest_status_subq.c.latest_status == specimen_status)

    query = query.order_by(Patient.patient_date.desc(), Patient.created_at.desc())
    patients, meta = paginate_query(query, page, per_page)
    return PaginatedResponse[PatientResponse](data=patients, meta=meta)

@router.post("", response_model=PatientResponse, status_code=201)
def create_patient(
    patient: PatientCreate,
    db: Session = Depends(get_db),
    _admin=Depends(require_admin),
):
    if patient.jenis_kelamin not in ['Laki-Laki', 'Perempuan']:
        raise HTTPException(status_code=400, detail="jenis_kelamin harus 'Laki-Laki' atau 'Perempuan'")
        
    db_patient = Patient(
        id_pasien=generate_id_pasien(),
        nama_lengkap=patient.nama_lengkap,
        jenis_kelamin=patient.jenis_kelamin,
        tanggal_lahir=patient.tanggal_lahir,
        alamat=patient.alamat,
        no_telepon=patient.no_telepon,
        patient_date=patient.patient_date or patient.date or datetime.utcnow(),
    )
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient


@router.put("/{patient_id}", response_model=PatientResponse)
def update_patient(
    patient_id: int,
    payload: PatientUpdate,
    db: Session = Depends(get_db),
    _admin=Depends(require_admin),
):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Pasien tidak ditemukan")

    if payload.jenis_kelamin is not None and payload.jenis_kelamin not in ["Laki-Laki", "Perempuan"]:
        raise HTTPException(status_code=400, detail="jenis_kelamin harus 'Laki-Laki' atau 'Perempuan'")

    update_data = payload.model_dump(exclude_unset=True)

    if "date" in update_data and "patient_date" not in update_data:
        update_data["patient_date"] = update_data["date"]

    for key, value in update_data.items():
        if key == "date":
            continue
        setattr(patient, key, value)

    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


@router.delete("/{patient_id}", status_code=204)
def delete_patient(
    patient_id: int,
    db: Session = Depends(get_db),
    _admin=Depends(require_admin),
):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Pasien tidak ditemukan")

    db.delete(patient)
    db.commit()
