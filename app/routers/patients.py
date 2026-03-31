from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_
import time

from app.database import get_db
from app.models import Patient
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
    patients = query.order_by(Patient.created_at.desc()).offset(skip).limit(limit).all()
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
        no_telepon=patient.no_telepon
    )
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient
