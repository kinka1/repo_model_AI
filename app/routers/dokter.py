import os
from typing import List, Optional
from datetime import datetime, date
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.models import Patient, Specimen, Classification, InternalMessage, User
from app.schemas import (
    ValidationTask, PatientDetail,
    ValidationUpdate, ValidationResponse,
    RevisionRequest, UnlockRequest, StatusActionResponse,
)
from app.utils import get_local_now

router = APIRouter(
    prefix="/api/doctor",
    tags=["Doctor Section"]
)

STATUS_WAITING_VALIDATION = "waiting_validation"
STATUS_REVISION = "revision"
STATUS_VALIDATED = "validated"


def _abs_url(path: Optional[str], base_url: str = "") -> str:
    if not path:
        return ""
    normalized = path.replace("\\", "/").lstrip("/")
    if normalized.startswith("http"):
        return normalized
    if normalized.startswith("static/") or normalized.startswith("uploads/"):
        return f"{base_url}/{normalized}"
    return f"{base_url}/uploads/{normalized}"


class ValidationItem(BaseModel):
    id: int
    validation_gram: Optional[str] = None
    validation_bentuk: Optional[str] = None
    catatan: Optional[str] = None
    catatan_dokter: Optional[str] = None
    is_deleted: Optional[bool] = False


class DoctorValidationSubmit(BaseModel):
    specimen_id: int
    validations: List[ValidationItem]
    sync_satusehat: Optional[bool] = False

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


@router.get("/doctor-queue")
def get_doctor_queue(db: Session = Depends(get_db)):
    queue = (
        db.query(Specimen)
        .join(Patient)
        .filter(Specimen.status == STATUS_WAITING_VALIDATION)
        .filter(Patient.is_active == True)
        .order_by(Specimen.uploaded_at.asc())
        .all()
    )

    # Group specimens by patient to avoid showing the same patient multiple times
    from collections import OrderedDict
    grouped: dict[int, dict] = OrderedDict()

    for s in queue:
        pid = s.patient_id
        if pid not in grouped:
            grouped[pid] = {
                "id_pasien": s.patient.id_pasien,
                "nama_pasien": s.patient.nama_lengkap,
                "nik": s.patient.nik,
                "earliest_upload": s.uploaded_at.strftime("%Y-%m-%d %H:%M") if s.uploaded_at else None,
                "total_specimens": 0,
                "total_bakteri": 0,
                "specimens": [],
            }

        g = grouped[pid]
        g["total_specimens"] += 1

        bakteri_count = db.query(Classification).filter(
            Classification.specimen_id == s.id,
            Classification.image_path.like(f"%{os.path.join('crops', str(s.id))}%")
        ).count()
        g["total_bakteri"] += bakteri_count

        g["specimens"].append({
            "id_specimen": s.id,
            "accession_number": s.accession_number or os.path.splitext(s.file_name or "")[0],
            "tanggal_upload": s.uploaded_at.strftime("%Y-%m-%d %H:%M") if s.uploaded_at else None,
            "specimen_type": s.specimen_type,
            "doctor_sender": s.doctor_sender,
            "clinical_diagnosis": s.clinical_diagnosis,
            "collected_at": s.collected_at.isoformat() if s.collected_at else None,
            "received_at": s.received_at.isoformat() if s.received_at else None,
            "validation_status": s.validation_status,
            "total_bakteri": bakteri_count,
        })

    return list(grouped.values())


def _get_patient_specimens(patient_id: int, current_specimen_id: int, base_url: str, db: Session) -> list[dict]:
    """Get all specimens for a patient with their annotated image URLs."""
    from app.models import Specimen as SpecimenModel
    specimens = (
        db.query(SpecimenModel)
        .filter(SpecimenModel.patient_id == patient_id)
        .order_by(SpecimenModel.uploaded_at.desc())
        .all()
    )
    result = []
    for s in specimens:
        annotated_path = f"uploads/specimen_{s.id}_annotated.jpg"
        result.append({
            "specimen_id": s.id,
            "accession_number": s.accession_number or (os.path.splitext(s.file_name or "")[0] if s.file_name else str(s.id)),
            "specimen_type": s.specimen_type,
            "annotated_image_url": _abs_url(annotated_path, base_url) if os.path.exists(annotated_path) else "",
            "main_image_url": _abs_url(s.file_path, base_url) if s.file_path else "",
            "is_current": s.id == current_specimen_id,
            "status": s.status,
            "uploaded_at": s.uploaded_at.strftime("%Y-%m-%d %H:%M") if s.uploaded_at else None,
        })
    return result


def _get_all_classifications(patient_id: int, base_url: str, db: Session) -> list[dict]:
    """Get all classifications from all specimens of a patient, tagged with specimen info."""
    from app.models import Specimen as SpecimenModel
    specimens = (
        db.query(SpecimenModel)
        .filter(SpecimenModel.patient_id == patient_id)
        .order_by(SpecimenModel.uploaded_at.desc())
        .all()
    )
    all_crops = []
    for s in specimens:
        crops = db.query(Classification).filter(Classification.specimen_id == s.id).all()
        label = s.accession_number or f"Spesimen #{s.id}"
        for c in crops:
            # Return raw image_path (relative) — frontend will build the full URL via API_HOST
            all_crops.append({
                "id": c.id,
                "specimen_id": s.id,
                "specimen_label": label,
                "ai_gram": c.classification_gram,
                "classification_bentuk": c.classification_bentuk,
                "confidence": float(c.confidence_score) if c.confidence_score is not None else 0.0,
                "image_path": c.image_path,
                "validation_gram": c.validation_gram,
                "validation_bentuk": c.validation_bentuk,
                "catatan": c.catatan_dokter,
            })
    return all_crops


@router.get("/specimen-details/{specimen_id}")
def get_specimen_details(specimen_id: int, request: Request, db: Session = Depends(get_db)):
    specimen = db.query(Specimen).filter(Specimen.id == specimen_id).first()
    if not specimen:
        raise HTTPException(status_code=404, detail="Spesimen tidak ditemukan")

    crops = db.query(Classification).filter(Classification.specimen_id == specimen_id).all()

    base_url = str(request.base_url).rstrip("/")

    def _url(path: Optional[str]) -> str:
        return _abs_url(path, base_url)

    analyst_name = "-"
    first_crop_with_user = next((c for c in crops if c.classified_by_user_id), None)
    if first_crop_with_user:
        from app.models import User
        user = db.query(User).filter(User.id == first_crop_with_user.classified_by_user_id).first()
        if user:
            analyst_name = user.full_name

    validator_name = "-"
    if specimen.validated_by_user_id:
        from app.models import User
        validator = db.query(User).filter(User.id == specimen.validated_by_user_id).first()
        if validator:
            validator_name = validator.full_name

    patient = specimen.patient
    patient_data = {}
    if patient:
        patient_data = {
            "id_pasien": patient.id_pasien,
            "nik": patient.nik,
            "nama": patient.nama_lengkap,
            "tanggal_lahir": patient.tanggal_lahir.strftime("%d %b %Y") if patient.tanggal_lahir else "-",
            "umur": f"{calculate_age(patient.tanggal_lahir)} Tahun" if patient.tanggal_lahir else "-",
            "jenis_kelamin": patient.jenis_kelamin,
            "alamat": patient.alamat,
            "no_telepon": patient.no_telepon,
            "registration_date": patient.patient_date.strftime("%d %b %Y %H:%M") if patient.patient_date else "-"
        }

    annotated_path = f"uploads/specimen_{specimen.id}_annotated.jpg"

    # Include messages for this specimen
    messages = (
        db.query(InternalMessage)
        .filter(InternalMessage.specimen_id == specimen_id)
        .order_by(InternalMessage.created_at.asc())
        .all()
    )
    messages_data = []
    for msg in messages:
        sender = msg.sender
        messages_data.append({
            "id": msg.id,
            "sender_id": msg.sender_id,
            "sender_name": sender.full_name if sender else "Unknown",
            "sender_role": sender.role if sender else "Unknown",
            "message_text": msg.message_text,
            "created_at": msg.created_at.strftime("%d %b %Y %H:%M") if msg.created_at else None,
        })

    return {
        "specimen_id": specimen.id,
        "patient_id": specimen.patient_id,
        "specimen_code": specimen.accession_number or (os.path.splitext(specimen.file_name)[0] if specimen.file_name else str(specimen.id)),
        "accession_number": specimen.accession_number,
        "specimen_type": specimen.specimen_type,
        "doctor_sender": specimen.doctor_sender,
        "clinical_diagnosis": specimen.clinical_diagnosis,
        "collected_at": specimen.collected_at.isoformat() if specimen.collected_at else None,
        "received_at": specimen.received_at.isoformat() if specimen.received_at else None,
        "microscope_type": specimen.microscope_type,
        "magnification": specimen.magnification,
        "image_resolution": specimen.image_resolution,
        "analyst_note": specimen.analyst_note,
        "validation_status": specimen.validation_status,
        "validated_at": specimen.validated_at.isoformat() if specimen.validated_at else None,
        "validator": validator_name,
        "patient_name": patient.nama_lengkap if patient else "Unknown",
        "patient": patient_data,
        "analyst": analyst_name,
        "created_at": specimen.uploaded_at.strftime("%d %b %Y %H:%M") if specimen.uploaded_at else "-",
        "main_image_url": _url(specimen.file_path),
        "annotated_image_url": _url(annotated_path) if os.path.exists(annotated_path) else "",
        "all_specimens": _get_patient_specimens(specimen.patient_id, specimen.id, base_url, db),
        "all_classifications": _get_all_classifications(specimen.patient_id, base_url, db),
        "classifications": [
            {
                "id": c.id,
                "ai_gram": c.classification_gram,
                "classification_bentuk": c.classification_bentuk,
                "confidence": float(c.confidence_score) if c.confidence_score is not None else 0.0,
                "roi_bbox": c.roi_bbox,
                "roi_source": c.roi_source,
                "image_url": _url(c.image_path),
                "validation_gram": c.validation_gram,
                "validation_bentuk": c.validation_bentuk,
                "catatan": c.catatan_dokter,
            }
            for c in crops
        ],
        "messages": messages_data,
    }


@router.post("/submit-validation")
def submit_doctor_validation(data: DoctorValidationSubmit, request: Request, db: Session = Depends(get_db)):
    specimen = db.query(Specimen).filter(Specimen.id == data.specimen_id).first()
    if not specimen:
        raise HTTPException(status_code=404, detail="Spesimen tidak ditemukan")

    # Validate that all classifications exist (across any specimen)
    affected_specimen_ids = set()
    for val in data.validations:
        crop = db.query(Classification).filter(Classification.id == val.id).first()
        if not crop:
            raise HTTPException(
                status_code=400,
                detail=f"Classification id {val.id} tidak ditemukan",
            )
        affected_specimen_ids.add(crop.specimen_id)

    # 1. Simpan Validasi ke Database Lokal
    for val in data.validations:
        crop = db.query(Classification).filter(Classification.id == val.id).first()
        if val.is_deleted:
            db.delete(crop)
            continue

        crop.validation_gram = val.validation_gram
        crop.validation_bentuk = val.validation_bentuk
        note_value = val.catatan if val.catatan is not None else val.catatan_dokter
        if note_value is not None:
            crop.catatan_dokter = note_value
        crop.reannotated_by_user_id = getattr(request.state, "user_id", None)
        crop.reannotated_at = get_local_now()

    db.commit()

    # Set all affected specimens to validated
    for sid in affected_specimen_ids:
        affected_spec = db.query(Specimen).filter(Specimen.id == sid).first()
        if affected_spec:
            remaining = db.query(Classification).filter(Classification.specimen_id == sid).count()
            affected_spec.status = STATUS_VALIDATED
            affected_spec.validation_status = "rejected" if remaining == 0 else "validated"
            affected_spec.validated_by_user_id = getattr(request.state, "user_id", None)
            affected_spec.validated_at = get_local_now()
            db.add(affected_spec)

    db.commit()

    # 2. Sinkronisasi ke SATUSEHAT (Opsional) - use primary specimen
    satusehat_msg = ""
    if data.sync_satusehat:
        remaining_primary = db.query(Classification).filter(Classification.specimen_id == data.specimen_id).count()
        if remaining_primary > 0:
            try:
                from app.services.satusehat_service import satusehat_service
                ORG_ID = os.getenv("SATUSEHAT_ORG_ID")
                if not ORG_ID:
                    raise ValueError("SATUSEHAT_ORG_ID belum dikonfigurasi di .env")

                patient = specimen.patient
                if not patient or not patient.satusehat_id:
                    raise ValueError("ID Pasien SATUSEHAT tidak ditemukan. Pastikan pasien diimpor via NIK.")

                # Hitung Kesimpulan untuk SATUSEHAT
                crops = db.query(Classification).filter(Classification.specimen_id == data.specimen_id).all()
                pos_kokus = sum(1 for c in crops if c.validation_gram == "Positif" and c.validation_bentuk == "Kokus")
                pos_batang = sum(1 for c in crops if c.validation_gram == "Positif" and c.validation_bentuk == "Batang")
                neg_kokus = sum(1 for c in crops if c.validation_gram == "Negatif" and c.validation_bentuk == "Kokus")
                neg_batang = sum(1 for c in crops if c.validation_gram == "Negatif" and c.validation_bentuk == "Batang")

                conclusions = []
                if pos_kokus > 0: conclusions.append(f"Gram Positif Kokus ({pos_kokus})")
                if pos_batang > 0: conclusions.append(f"Gram Positif Batang ({pos_batang})")
                if neg_kokus > 0: conclusions.append(f"Gram Negatif Kokus ({neg_kokus})")
                if neg_batang > 0: conclusions.append(f"Gram Negatif Batang ({neg_batang})")

                final_conclusion = ", ".join(conclusions) if conclusions else "Tidak ditemukan bakteri"

                # Cari Encounter dari SATUSEHAT
                encounter_id = satusehat_service.get_encounter_by_patient(patient.satusehat_id)
                if not encounter_id:
                    print(f"Tidak ada Encounter untuk pasien {patient.satusehat_id}, lewati sync SATUSEHAT.")
                    satusehat_msg = ", namun tidak ada Encounter SATUSEHAT untuk pasien ini"
                else:
                    # Kirim Observation (dengan encounter reference)
                    obs_res = satusehat_service.send_observation(patient.satusehat_id, final_conclusion, ORG_ID, encounter_id)
                    obs_id = obs_res.get("id")

                    # Kirim DiagnosticReport
                    if obs_id:
                        acc_num = specimen.accession_number or str(specimen.id)
                        satusehat_service.send_diagnostic_report(patient.satusehat_id, obs_id, ORG_ID, acc_num)
                        satusehat_msg = " dan tersinkronisasi ke SATUSEHAT"
                    else:
                        satusehat_msg = ", namun gagal mendapatkan ID Observation dari SATUSEHAT"

            except Exception as e:
                print(f"SATUSEHAT Sync Error: {str(e)}")
                return {
                    "success": True,
                    "status": "partial_success",
                    "message": f"Validasi berhasil disimpan di lokal, namun gagal sinkronisasi ke SATUSEHAT: {str(e)}"
                }

    return {
        "success": True,
        "message": f"Validasi berhasil disimpan{satusehat_msg}."
    }


@router.post("/request-revision/{specimen_id}")
def request_revision(
    specimen_id: int,
    payload: RevisionRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Dokter mengembalikan spesimen ke Analis untuk direvisi.
    Hanya role Dokter dan Admin yang diizinkan.
    """
    user_role = getattr(request.state, "role", None)
    if user_role not in ("Dokter", "Admin"):
        raise HTTPException(status_code=403, detail="Hanya Dokter dan Admin yang dapat meminta revisi")

    specimen = db.query(Specimen).filter(Specimen.id == specimen_id).first()
    if not specimen:
        raise HTTPException(status_code=404, detail="Spesimen tidak ditemukan")

    # Update specimen status
    specimen.status = STATUS_REVISION
    specimen.validation_status = "pending"
    specimen.validated_by_user_id = None
    specimen.validated_at = None
    specimen.uploaded_at = get_local_now()  # Reset timestamp agar muncul di urutan teratas antrean analis
    db.add(specimen)

    # Create internal message with revision reason
    message = InternalMessage(
        specimen_id=specimen_id,
        sender_id=getattr(request.state, "user_id", None),
        message_text=payload.message,
    )
    db.add(message)
    db.commit()

    return {
        "success": True,
        "message": "Spesimen dikembalikan untuk revisi",
        "new_status": STATUS_REVISION,
    }


@router.post("/unlock/{specimen_id}")
def unlock_specimen(
    specimen_id: int,
    payload: UnlockRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Dokter membuka kunci spesimen yang sudah tervalidasi untuk diproses ulang.
    Hanya role Dokter dan Admin yang diizinkan.
    """
    user_role = getattr(request.state, "role", None)
    if user_role not in ("Dokter", "Admin"):
        raise HTTPException(status_code=403, detail="Hanya Dokter dan Admin yang dapat membuka kunci")

    specimen = db.query(Specimen).filter(Specimen.id == specimen_id).first()
    if not specimen:
        raise HTTPException(status_code=404, detail="Spesimen tidak ditemukan")

    if specimen.status != STATUS_VALIDATED:
        raise HTTPException(status_code=400, detail="Hanya spesimen dengan status 'validated' yang dapat di-unlock")

    # Reset specimen validation fields
    specimen.status = STATUS_REVISION
    specimen.validation_status = "pending"
    specimen.validated_by_user_id = None
    specimen.validated_at = None
    specimen.uploaded_at = get_local_now()
    db.add(specimen)

    # Clear catatan_dokter on ALL Classification records for this specimen
    db.query(Classification).filter(
        Classification.specimen_id == specimen_id
    ).update(
        {Classification.catatan_dokter: None},
        synchronize_session=False,
    )

    # Create internal message explaining why it was unlocked
    unlock_message = payload.message or "Validasi dibatalkan oleh Dokter. Spesimen dapat diproses ulang."
    message = InternalMessage(
        specimen_id=specimen_id,
        sender_id=getattr(request.state, "user_id", None),
        message_text=unlock_message,
    )
    db.add(message)
    db.commit()

    return {
        "success": True,
        "message": "Validasi dibatalkan. Spesimen dapat diproses ulang.",
        "new_status": STATUS_REVISION,
    }


@router.get("/pending-validation", response_model=List[ValidationTask], deprecated=True)
def get_pending_validation(
    request: Request,
    db: Session = Depends(get_db)
):
    raise HTTPException(
        status_code=410,
        detail="Endpoint /api/doctor/pending-validation sudah deprecated. Gunakan /api/doctor/doctor-queue.",
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
        detail="Endpoint /api/doctor/classification/{id}/validate sudah deprecated. Gunakan /api/doctor/submit-validation.",
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
    classification.reannotated_at = get_local_now()
    
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


def _get_chart_stats(filter_type: str, db: Session):
    from datetime import date, timedelta
    from sqlalchemy import func
    today = date.today()
    result = []

    # Helper to count specimens with active patients only
    def _count_specimens(*filters):
        return db.query(func.count(Specimen.id)).join(Patient).filter(
            Patient.is_active == True,
            *filters
        ).scalar() or 0
    
    if filter_type == "Hari Ini":
        # 2-hour intervals: 08:00, 10:00, 12:00, 14:00, 16:00, 18:00, 20:00
        intervals = [
            ("08:00", 8, 10),
            ("10:00", 10, 12),
            ("12:00", 12, 14),
            ("14:00", 14, 16),
            ("16:00", 16, 18),
            ("18:00", 18, 20),
            ("20:00", 20, 22)
        ]
        for label, start_h, end_h in intervals:
            masuk = _count_specimens(
                func.date(Specimen.uploaded_at) == today,
                func.extract('hour', Specimen.uploaded_at) >= start_h,
                func.extract('hour', Specimen.uploaded_at) < end_h
            )
            
            validated = _count_specimens(
                Specimen.status == "validated",
                func.date(Specimen.validated_at) == today,
                func.extract('hour', Specimen.validated_at) >= start_h,
                func.extract('hour', Specimen.validated_at) < end_h
            )
            
            result.append({
                "name": label,
                "masuk": masuk,
                "selesai": validated,
                "validated": validated,
                "pending": max(0, masuk - validated)
            })
            
    elif filter_type == "Harian":
        days = []
        for i in range(6, -1, -1):
            d = today - timedelta(days=i)
            days.append(d)
            
        day_labels = {
            0: "Sen", 1: "Sel", 2: "Rab", 3: "Kam", 4: "Jum", 5: "Sab", 6: "Min"
        }
        
        for d in days:
            label = day_labels[d.weekday()]
            masuk = _count_specimens(
                func.date(Specimen.uploaded_at) == d
            )
            
            validated = _count_specimens(
                Specimen.status == "validated",
                func.date(Specimen.validated_at) == d
            )
            
            result.append({
                "name": label,
                "masuk": masuk,
                "selesai": validated,
                "validated": validated,
                "pending": max(0, masuk - validated)
            })
            
    elif filter_type == "Mingguan":
        for i in range(4):
            start_d = today - timedelta(days=(4-i)*7)
            end_d = today - timedelta(days=(3-i)*7)
            
            masuk = _count_specimens(
                func.date(Specimen.uploaded_at) >= start_d,
                func.date(Specimen.uploaded_at) < end_d
            )
            
            validated = _count_specimens(
                Specimen.status == "validated",
                func.date(Specimen.validated_at) >= start_d,
                func.date(Specimen.validated_at) < end_d
            )
            
            result.append({
                "name": f"Mg {i+1}",
                "masuk": masuk,
                "selesai": validated,
                "validated": validated,
                "pending": max(0, masuk - validated)
            })
            
    elif filter_type == "Bulanan":
        months = [
            ("Jan", 1), ("Feb", 2), ("Mar", 3), ("Apr", 4), ("Mei", 5), ("Jun", 6),
            ("Jul", 7), ("Agt", 8), ("Sep", 9), ("Okt", 10), ("Nov", 11), ("Des", 12)
        ]
        curr_year = today.year
        for label, m_idx in months:
            masuk = _count_specimens(
                func.extract('year', Specimen.uploaded_at) == curr_year,
                func.extract('month', Specimen.uploaded_at) == m_idx
            )
            
            validated = _count_specimens(
                Specimen.status == "validated",
                func.extract('year', Specimen.validated_at) == curr_year,
                func.extract('month', Specimen.validated_at) == m_idx
            )
            
            result.append({
                "name": label,
                "masuk": masuk,
                "selesai": validated,
                "validated": validated,
                "pending": max(0, masuk - validated)
            })
            
    elif filter_type == "Tahunan":
        curr_year = today.year
        for i in range(3, -1, -1):
            y = curr_year - i
            masuk = _count_specimens(
                func.extract('year', Specimen.uploaded_at) == y
            )
            
            validated = _count_specimens(
                Specimen.status == "validated",
                func.extract('year', Specimen.validated_at) == y
            )
            
            result.append({
                "name": str(y),
                "masuk": masuk,
                "selesai": validated,
                "validated": validated,
                "pending": max(0, masuk - validated)
            })
            
    return result


@router.get("/dashboard-chart")
def get_doctor_dashboard_chart(filter: str = Query("Harian"), db: Session = Depends(get_db)):
    valid_filters = {"Hari Ini", "Harian", "Mingguan", "Bulanan", "Tahunan"}
    if filter not in valid_filters:
        filter = "Harian"
    return _get_chart_stats(filter, db)


@router.get("/dashboard-stats")
def get_doctor_dashboard_stats(db: Session = Depends(get_db)):
    """Mengembalikan metrik Clinical Decision Support untuk dashboard dokter.
    
    - concordance_rate: Persentase kecocokan antara prediksi AI vs validasi dokter
    - critical_anomalies_count: Jumlah kasus dengan anomali kritis
    - anomalies: Daftar anomali yang memerlukan perhatian segera
    """
    from datetime import timedelta

    # --- AI CONCORDANCE RATE ---
    # Compare classification_gram (AI) vs validation_gram (Doctor)
    concordance_query = (
        db.query(
            Classification.classification_gram,
            Classification.validation_gram,
        )
        .filter(Classification.validation_gram.isnot(None))
        .filter(Classification.classification_gram.isnot(None))
        .all()
    )

    total_validated = len(concordance_query)
    matches = sum(
        1 for c in concordance_query
        if c.classification_gram == c.validation_gram
    )
    concordance_rate = round((matches / total_validated) * 100, 1) if total_validated > 0 else 0.0

    # --- CRITICAL ANOMALIES ---
    # Find cases where:
    # 1. AI and doctor disagree AND the condition is critical (AI predicted Negatif but Doctor says Positif, or vice versa)
    # 2. High urgency cases waiting too long for validation
    
    # Discrepancies between AI and Doctor
    discrepancies = (
        db.query(Classification, Specimen, Patient)
        .join(Specimen, Classification.specimen_id == Specimen.id)
        .join(Patient, Specimen.patient_id == Patient.id)
        .filter(Classification.validation_gram.isnot(None))
        .filter(Classification.classification_gram.isnot(None))
        .filter(Classification.classification_gram != Classification.validation_gram)
        .filter(Patient.is_active == True)
        .order_by(Classification.updated_at.desc())
        .limit(20)
        .all()
    )

    # Critical: AI said Negatif but Doctor validated Positif (potential missed diagnosis)
    # Also: Long-waiting high priority cases
    anomalies = []
    
    for cls, spec, patient in discrepancies:
        is_critical = cls.classification_gram == "Negatif" and cls.validation_gram == "Positif"
        
        # Determine discrepancy type
        if cls.classification_gram == "Positif" and cls.validation_gram == "Negatif":
            issue = "AI: Positif, Dokter: Negatif (Overdiagnosis AI)"
            severity = "medium"
        elif cls.classification_gram == "Negatif" and cls.validation_gram == "Positif":
            issue = "AI: Negatif, Dokter: Positif (Underdiagnosis AI — KRITIS)"
            severity = "high"
        else:
            continue

        anomalies.append({
            "id": cls.id,
            "nama_pasien": patient.nama_lengkap,
            "id_pasien": patient.id_pasien,
            "spekimen_id": spec.id,
            "issue": issue,
            "severity": severity,
            "confidence_ai": float(cls.confidence_score) if cls.confidence_score else 0,
            "tanggal": cls.updated_at.strftime("%Y-%m-%d %H:%M") if cls.updated_at else None,
        })

    # Also find high-priority cases waiting > 2 hours
    long_waiting = (
        db.query(Specimen, Patient)
        .join(Patient, Specimen.patient_id == Patient.id)
        .filter(Specimen.status == STATUS_WAITING_VALIDATION)
        .filter(Specimen.uploaded_at.isnot(None))
        .filter(Specimen.uploaded_at < get_local_now() - timedelta(hours=2))
        .filter(Patient.is_active == True)
        .order_by(Specimen.uploaded_at.asc())
        .limit(10)
        .all()
    )

    for spec, patient in long_waiting:
        wait_hours = round((datetime.now() - spec.uploaded_at).total_seconds() / 3600, 1)
        anomalies.append({
            "id": spec.id,
            "nama_pasien": patient.nama_lengkap,
            "id_pasien": patient.id_pasien,
            "spekimen_id": spec.id,
            "issue": f"Menunggu validasi selama {wait_hours} jam — melebihi batas waktu",
            "severity": "high",
            "confidence_ai": None,
            "tanggal": spec.uploaded_at.strftime("%Y-%m-%d %H:%M") if spec.uploaded_at else None,
        })

    # Sort: high severity first
    severity_order = {"high": 0, "medium": 1, "low": 2}
    anomalies.sort(key=lambda a: severity_order.get(a["severity"], 99))

    critical_count = sum(1 for a in anomalies if a["severity"] == "high")

    return {
        "concordance_rate": concordance_rate,
        "concordance_total": total_validated,
        "concordance_match": matches,
        "critical_anomalies_count": critical_count,
        "total_anomalies": len(anomalies),
        "anomalies": anomalies[:10],  # Top 10 only
    }

