from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_, func, and_
import time
from datetime import datetime

from app.database import get_db
from app.models import Patient, Specimen
from app.schemas import PatientCreate, PatientResponse
from typing import List

router = APIRouter(
    prefix="/api/patients",
    tags=["Patient Management"]
)

def generate_id_pasien():
    # Simple ID generation: PAS-Timestamp
    return f"PAS-{int(time.time()*1000)}"[:20]

@router.get("", response_model=List[PatientResponse])
def get_patients(
    skip: int = 0, 
    limit: int = 100, 
    search: str = Query(None, description="Search by name or id_pasien"),
    specimen_status: str = Query(None, description="Filter by latest specimen status, e.g. pending/waiting_validation/validated"),
    include_no_specimen: bool = Query(True, description="When filtering by specimen_status, include patients without specimen"),
    db: Session = Depends(get_db)
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

    patients = query.order_by(Patient.patient_date.desc(), Patient.created_at.desc()).offset(skip).limit(limit).all()
    return patients

@router.post("", response_model=PatientResponse, status_code=201)
def create_patient(patient: PatientCreate, db: Session = Depends(get_db)):
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
