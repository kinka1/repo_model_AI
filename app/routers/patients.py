from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from sqlalchemy import or_, func, and_
import time
from datetime import datetime, date

from app.database import get_db
from app.models import Patient, Specimen
from app.schemas import PatientCreate, PatientResponse, PatientUpdate, PaginatedResponse, SatusehatImportRequest
from app.utils import paginate_query, get_local_now
from app.services.satusehat_service import satusehat_service
from typing import Optional

router = APIRouter(
    prefix="/api/patients",
    tags=["Patient Management"]
)


def require_admin(request: Request):
    role = getattr(request.state, "role", None)
    if role != "Admin":
        raise HTTPException(status_code=403, detail="Hanya Admin yang boleh mengakses endpoint patient")


import random
from datetime import datetime


def require_admin_or_analis(request: Request):
    role = getattr(request.state, "role", None)
    if role not in {"Admin", "Analis"}:
        raise HTTPException(status_code=403, detail="Hanya Admin atau Analis yang boleh mengakses data patient")

def generate_id_pasien():
    now = datetime.now()
    rand = random.randint(1000, 9999)
    return f"RM-{now.strftime('%Y%m%d')}-{rand}"

@router.get("", response_model=PaginatedResponse[PatientResponse])
def get_patients(
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    search: Optional[str] = Query(None, description="Search by name or id_pasien"),
    specimen_status: Optional[str] = Query(None, description="Filter by latest specimen status, e.g. pending/waiting_validation/validated"),
    include_no_specimen: bool = Query(True, description="When filtering by specimen_status, include patients without specimen"),
    include_inactive: bool = Query(False, description="Include deactivated (soft-deleted) patients"),
    db: Session = Depends(get_db),
    _role=Depends(require_admin_or_analis),
):
    query = db.query(Patient)

    # Filter active patients by default
    if not include_inactive:
        query = query.filter(Patient.is_active == True)

    if search:
        query = query.filter(
            or_(
                Patient.nama_lengkap.ilike(f"%{search}%"),
                Patient.id_pasien.ilike(f"%{search}%"),
                Patient.nik.ilike(f"%{search}%")
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

    # Populate latest_specimen_id for each patient
    if specimen_status and patients:
        latest_ids = db.query(
            Specimen.patient_id,
            func.max(Specimen.id).label("specimen_id")
        ).filter(
            Specimen.patient_id.in_([p.id for p in patients]),
            Specimen.status == specimen_status
        ).group_by(Specimen.patient_id).all()
        id_map = {row.patient_id: row.specimen_id for row in latest_ids}
        for p in patients:
            setattr(p, "latest_specimen_id", id_map.get(p.id))

    return PaginatedResponse[PatientResponse](data=patients, meta=meta)

@router.post("", response_model=PatientResponse, status_code=201)
def create_patient(
    patient: PatientCreate,
    db: Session = Depends(get_db),
    _admin=Depends(require_admin),
):
    if not patient.nik:
        raise HTTPException(status_code=400, detail="NIK wajib diisi")

    if patient.jenis_kelamin not in ['Laki-Laki', 'Perempuan']:
        raise HTTPException(status_code=400, detail="jenis_kelamin harus 'Laki-Laki' atau 'Perempuan'")

    # Cegah duplikasi pasien berdasarkan NIK
    existing = db.query(Patient).filter(Patient.nik == patient.nik).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Pasien dengan NIK {patient.nik} sudah terdaftar (ID: {existing.id_pasien}). Gunakan pasien yang sudah ada atau impor dari SATUSEHAT.",
        )

    rm_value = (patient.id_pasien or "").strip() or generate_id_pasien()

    db_patient = Patient(
        id_pasien=rm_value,
        nik=patient.nik,
        nama_lengkap=patient.nama_lengkap,
        jenis_kelamin=patient.jenis_kelamin,
        tanggal_lahir=patient.tanggal_lahir,
        alamat=patient.alamat,
        no_telepon=patient.no_telepon,
        patient_date=patient.patient_date or patient.date or get_local_now(),
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

    # Cegah update NIK ke milik pasien lain
    if payload.nik is not None and payload.nik != patient.nik:
        existing = db.query(Patient).filter(Patient.nik == payload.nik, Patient.id != patient_id).first()
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"NIK {payload.nik} sudah terdaftar untuk pasien lain (ID: {existing.id_pasien}).",
            )

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


@router.delete("/{patient_id}")
def delete_patient(patient_id: int, db: Session = Depends(get_db), _role=Depends(require_admin)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    if not patient.is_active:
        raise HTTPException(status_code=400, detail="Pasien sudah dinonaktifkan sebelumnya")

    patient.is_active = False
    db.add(patient)
    db.commit()
    return {"message": "Pasien berhasil dinonaktifkan", "id": patient.id, "is_active": False}


@router.put("/{patient_id}/restore")
def restore_patient(patient_id: int, db: Session = Depends(get_db), _role=Depends(require_admin)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    if patient.is_active:
        raise HTTPException(status_code=400, detail="Pasien sudah aktif")

    patient.is_active = True
    db.add(patient)
    db.commit()
    return {"message": "Pasien berhasil diaktifkan kembali", "id": patient.id, "is_active": True}

@router.post("/import-satusehat", response_model=PatientResponse)
def import_patient_satusehat(
    request_data: SatusehatImportRequest,
    request: Request,
    db: Session = Depends(get_db),
    _role=Depends(require_admin)
):
    try:
        # Panggil API SATUSEHAT
        satusehat_data = satusehat_service.get_patient_by_nik(request_data.nik)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    if "entry" not in satusehat_data or not satusehat_data["entry"]:
        raise HTTPException(status_code=404, detail="Data pasien tidak ditemukan di SATUSEHAT.")

    patient_entry = satusehat_data["entry"][0].get("resource", {})

    # 1. Extract Name (Lebih Aman)
    name = "Unknown"
    names = patient_entry.get("name", [])
    if names and "text" in names[0]:
        name = names[0]["text"]
    elif names and "family" in names[0]:
        given = " ".join(names[0].get("given", []))
        name = f"{given} {names[0].get('family', '')}".strip()

    # 2. Extract birthDate (Dengan Fallback agar tidak 422)
    birth_date_str = patient_entry.get("birthDate")
    if birth_date_str:
        try:
            birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d").date()
        except ValueError:
            birth_date = date(1900, 1, 1) # Fallback jika format salah
    else:
        # Gunakan tanggal default jika data kosong di SATUSEHAT
        birth_date = date(1900, 1, 1)

    # 3. Extract gender (Case-insensitive mapping)
    raw_gender = str(patient_entry.get("gender", "unknown")).lower()
    gender = "Laki-Laki" # Default
    if raw_gender == "female":
        gender = "Perempuan"
    elif raw_gender == "male":
        gender = "Laki-Laki"

    # Extract address and telecom
    address = None
    if "address" in patient_entry and len(patient_entry["address"]) > 0:
        addr = patient_entry["address"][0]
        lines = addr.get("line", [])
        city = addr.get("city", "")
        address = ", ".join(lines)
        if city:
            address += f", {city}"
            
    telecom = None
    if "telecom" in patient_entry and len(patient_entry["telecom"]) > 0:
        telecom = patient_entry["telecom"][0].get("value")

    satusehat_id = patient_entry.get("id")

    # Check if patient already exists by NIK
    existing_patient = db.query(Patient).filter(Patient.nik == request_data.nik).first()
    
    user_id = getattr(request.state, "user_id", None)

    if existing_patient:
        # Update existing patient
        existing_patient.nama_lengkap = name
        existing_patient.jenis_kelamin = gender
        existing_patient.tanggal_lahir = birth_date
        if address:
            existing_patient.alamat = address
        if telecom:
            existing_patient.no_telepon = telecom
        if satusehat_id:
            existing_patient.satusehat_id = satusehat_id
        existing_patient.updated_at = get_local_now()
        
        db.commit()
        db.refresh(existing_patient)
        return existing_patient
    else:
        # Create new patient
        new_patient = Patient(
            id_pasien=generate_id_pasien(),
            nik=request_data.nik,
            nama_lengkap=name,
            jenis_kelamin=gender,
            tanggal_lahir=birth_date,
            alamat=address,
            no_telepon=telecom,
            satusehat_id=satusehat_id,
            patient_date=get_local_now(),
            created_by_user_id=user_id
        )
        db.add(new_patient)
        db.commit()
        db.refresh(new_patient)
        return new_patient


@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient_detail(
    patient_id: int,
    db: Session = Depends(get_db),
    _role=Depends(require_admin_or_analis),
):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Pasien tidak ditemukan")
    
    # Populate latest_specimen_id for the patient
    latest_specimen = (
        db.query(Specimen)
        .filter(Specimen.patient_id == patient.id)
        .order_by(Specimen.id.desc())
        .first()
    )
    if latest_specimen:
        setattr(patient, "latest_specimen_id", latest_specimen.id)
        
    return patient
