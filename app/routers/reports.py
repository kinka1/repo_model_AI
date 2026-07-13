import os
from datetime import date, datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Classification, Patient, Specimen, User
from app.schemas import (
    MedicalReportResponse,
    ReportClassificationDetail,
    ReportClinicalData,
    ReportEvidenceImage,
    ReportPatientData,
    ReportResultSummary,
)
from app.utils import get_local_now

router = APIRouter(prefix="/api/reports", tags=["Laporan Medis"])


def _calculate_age(birth_date: date) -> int:
    today = date.today()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))


def _normalize_shape(shape: Optional[str]) -> str:
    if not shape:
        return "Tidak teridentifikasi"
    s_lower = shape.lower()
    if "kokus" in s_lower or "coccus" in s_lower:
        return "Kokus"
    if "batang" in s_lower or "rod" in s_lower or "bacillus" in s_lower:
        return "Batang"
    return "Lainnya"


def _generate_conclusion(summary: ReportResultSummary) -> str:
    detected_strains: List[str] = []

    if summary.gram_positif_kokus > 0:
        detected_strains.append("Gram-positif bentuk kokus")
    if summary.gram_positif_batang > 0:
        detected_strains.append("Gram-positif bentuk batang")
    if summary.gram_negatif_kokus > 0:
        detected_strains.append("Gram-negatif bentuk kokus")
    if summary.gram_negatif_batang > 0:
        detected_strains.append("Gram-negatif bentuk batang")

    if not detected_strains:
        return "Tidak ditemukan bakteri pada spesimen."

    return f"Ditemukan bakteri {' dan '.join(detected_strains)}."


def _get_user_name(user_id: Optional[int], db: Session) -> Optional[str]:
    if user_id is None:
        return None
    user = db.query(User).filter(User.id == user_id).first()
    return user.full_name if user else "Pengguna tidak ditemukan"


def _resolve_doctor_and_analyst(classifications: List[Classification], db: Session) -> Dict[str, Optional[str]]:
    doctor_id = None
    analyst_id = None

    for c in classifications:
        if c.reannotated_by_user_id:
            doctor_id = c.reannotated_by_user_id
        if c.classified_by_user_id:
            analyst_id = c.classified_by_user_id

    return {
        "dokter": _get_user_name(doctor_id, db),
        "analis": _get_user_name(analyst_id, db) or "Sistem AI",
    }


def _abs_url(path: Optional[str], request: Request) -> str:
    if not path:
        return ""

    # Normalize path separator and strip leading slashes
    normalized = path.replace("\\", "/").lstrip("/")

    base_url = str(request.base_url).rstrip("/")

    if normalized.startswith("http"):
        return normalized

    # If it starts with static/ or uploads/, return with leading slash
    if normalized.startswith("static/") or normalized.startswith("uploads/"):
        return f"{base_url}/{normalized}"
    
    # Default fallback: assume it's in uploads
    return f"{base_url}/uploads/{normalized}"


@router.get("/specimen/{specimen_id}", response_model=MedicalReportResponse)
def get_medical_report(specimen_id: int, request: Request, db: Session = Depends(get_db)):
    specimen = (
        db.query(Specimen)
        .filter(Specimen.id == specimen_id)
        .join(Patient)
        .first()
    )

    if not specimen:
        raise HTTPException(status_code=404, detail="Spesimen tidak ditemukan.")

    patient = specimen.patient
    classifications = (
        db.query(Classification)
        .filter(Classification.specimen_id == specimen_id)
        .all()
    )

    if not classifications:
        raise HTTPException(
            status_code=404,
            detail="Belum ada data klasifikasi untuk spesimen ini.",
        )

    # Agregasi hasil (dihitung di Python agar kompatibel lintas DB)
    gp_kokus = 0
    gp_batang = 0
    gn_kokus = 0
    gn_batang = 0

    for c in classifications:
        gram = c.validation_gram or c.classification_gram
        bentuk = _normalize_shape(c.validation_bentuk or c.classification_bentuk)

        if gram == "Positif":
            if bentuk == "Batang":
                gp_batang += 1
            else:
                # Kokus or unknown shape (ML only predicts gram, not shape)
                gp_kokus += 1
        elif gram == "Negatif":
            if bentuk == "Batang":
                gn_batang += 1
            else:
                gn_kokus += 1

    summary = ReportResultSummary(
        total_objek=len(classifications),
        gram_positif_kokus=gp_kokus,
        gram_positif_batang=gp_batang,
        gram_negatif_kokus=gn_kokus,
        gram_negatif_batang=gn_batang,
        kesimpulan="",
        catatan_dokter=next(
            (c.catatan_dokter for c in classifications if c.catatan_dokter), None
        ),
    )
    summary.kesimpulan = _generate_conclusion(summary)

    evidence_images = [
        ReportEvidenceImage(
            image_url=_abs_url(c.image_path, request),
            label=f"Gram {c.validation_gram or c.classification_gram}, Bentuk {_normalize_shape(c.validation_bentuk or c.classification_bentuk)}",
        )
        for c in classifications
    ]

    classification_details = [
        ReportClassificationDetail(
            id=c.id,
            roi_bbox=c.roi_bbox,
            classification_gram=c.classification_gram,
            classification_bentuk=c.classification_bentuk,
            validation_gram=c.validation_gram,
            validation_bentuk=c.validation_bentuk,
            image_url=_abs_url(c.image_path, request),
            label=f"Gram {c.validation_gram or c.classification_gram}, Bentuk {_normalize_shape(c.validation_bentuk or c.classification_bentuk)}",
        )
        for c in classifications
    ]

    personnel = _resolve_doctor_and_analyst(classifications, db)

    validator_name = _get_user_name(specimen.validated_by_user_id, db)

    return MedicalReportResponse(
        id_laporan=str(specimen.id),
        tanggal_cetak=get_local_now(),
        specimen_id=specimen.id,
        pasien=ReportPatientData(
            id_pasien=patient.id_pasien,
            nik=patient.nik,
            nama=patient.nama_lengkap,
            tanggal_lahir=patient.tanggal_lahir,
            umur=_calculate_age(patient.tanggal_lahir),
            jenis_kelamin=patient.jenis_kelamin,
        ),
        data_klinis=ReportClinicalData(
            tanggal_sampel=specimen.uploaded_at,
            jenis_spesimen=specimen.specimen_type or "Pewarnaan Gram",
            accession_number=specimen.accession_number,
            doctor_sender=specimen.doctor_sender,
            clinical_diagnosis=specimen.clinical_diagnosis,
            collected_at=specimen.collected_at,
            received_at=specimen.received_at,
            microscope_type=specimen.microscope_type,
            magnification=specimen.magnification,
            image_resolution=specimen.image_resolution,
            analyst_note=specimen.analyst_note,
            validation_status=specimen.validation_status,
            validated_at=specimen.validated_at,
            validator=validator_name,
            analis=personnel["analis"],
            dokter=personnel["dokter"],
        ),
        ringkasan_hasil=summary,
        gambar_bukti=evidence_images,
        main_image_url=_abs_url(specimen.file_path, request),
        classifications=classification_details,
    )
