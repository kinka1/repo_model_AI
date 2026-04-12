import os
from typing import List, Optional
from datetime import datetime, date
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Patient, Specimen, Classification
from app.schemas import (
    ValidationTask, PatientDetail, 
    ValidationUpdate, ValidationResponse
)

router = APIRouter(
    prefix="/api/dokter",
    tags=["Doctor Section"]
)

def normalize_validation_values(gram: str, bentuk: Optional[str]):
    """
    Normalisasi input user/dokter. Jika pola dikenali ('kokus'/'batang'), ubah ke format standar.
    Jika tidak, biarkan apa adanya (tidak sestrik sebelumnya).
    """
    # Import helper from analysis to avoid duplication, OR just move the logic here
    # Since it's clinical validation, it's better defined here.
    from app.routers.analysis import map_predictions_to_db
    norm_gram = map_predictions_to_db(gram)
    
    norm_bentuk = bentuk
    if bentuk:
        b_lower = bentuk.lower()
        if "batang" in b_lower:
            norm_bentuk = "Batang"
        elif "kokus" in b_lower:
            norm_bentuk = "Kokus"
        else:
            norm_bentuk = bentuk 
            
    return norm_gram, norm_bentuk

def calculate_age(birth_date: date) -> int:
    today = date.today()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))

@router.get("/pending-validation", response_model=List[ValidationTask], deprecated=True)
def get_pending_validation(
    request: Request,
    db: Session = Depends(get_db)
):
    raise HTTPException(
        status_code=410,
        detail="Endpoint /api/dokter/pending-validation sudah deprecated. Gunakan /api/analysis/doctor-queue.",
    )

    """
    Mengambil daftar klasifikasi yang belum divalidasi oleh dokter (validation_gram is Null).
    Serta menyertakan detail informasi pasien dan umur.
    """
    # Join ke Patient agar data bisa ditarik
    query = db.query(Classification).filter(
        Classification.validation_gram == None
    ).join(Patient).join(Specimen)
    
    # Ambil semua record tanpa pagination
    items = query.order_by(Classification.created_at.desc()).all()
    
    results = []
    for item in items:
        # Bangun URL gambar crop
        img_url = f"{request.base_url}static/crops/{item.image_file_name}"
        
        results.append(ValidationTask(
            id=item.id,
            patient=PatientDetail(
                id_pasien=item.patient.id_pasien,
                nama_lengkap=item.patient.nama_lengkap,
                tanggal_lahir=item.patient.tanggal_lahir,
                umur=calculate_age(item.patient.tanggal_lahir),
                jenis_kelamin=item.patient.jenis_kelamin
            ),
            image_url=img_url,
            classification_gram=item.classification_gram,
            classification_bentuk=item.classification_bentuk,
            confidence_score=float(item.confidence_score),
            kode_sample=os.path.splitext(item.specimen.file_name)[0]
        ))
        
    return results

@router.patch("/classification/{classification_id}/validate", response_model=ValidationResponse, deprecated=True)
def validate_classification(
    classification_id: int, 
    payload: ValidationUpdate, 
    db: Session = Depends(get_db)
):
    raise HTTPException(
        status_code=410,
        detail="Endpoint /api/dokter/classification/{id}/validate sudah deprecated. Gunakan /api/analysis/submit-validation.",
    )

    """
    Memperbarui status validasi dokter untuk satu hasil deteksi/crop.
    """
    # Cari record klasifikasi
    classification = db.query(Classification).filter(Classification.id == classification_id).first()
    if not classification:
        raise HTTPException(status_code=404, detail="Data klasifikasi tidak ditemukan.")
    
    # Normalisasi data input agar sesuai dengan check constraint DB
    norm_gram, norm_bentuk = normalize_validation_values(payload.validation_gram, payload.validation_bentuk)
    
    # Update field validasi
    classification.validation_gram = norm_gram
    classification.validation_bentuk = norm_bentuk
    classification.catatan_dokter = payload.catatan_dokter
    classification.reannotated_by_user_id = payload.reannotated_by_user_id
    classification.reannotated_at = datetime.utcnow()
    
    db.add(classification)
    db.commit()
    db.refresh(classification)
    
    return ValidationResponse(
        id=classification.id,
        validation_gram=classification.validation_gram,
        validation_bentuk=classification.validation_bentuk,
        catatan_dokter=classification.catatan_dokter,
        reannotated_at=classification.reannotated_at,
        message="Status validasi berhasil diperbarui."
    )
