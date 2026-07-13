import os
import shutil
import uuid
import logging
import statistics
from typing import List, Optional
from datetime import datetime, date
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, Request, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, case
from pydantic import BaseModel

from app.database import get_db
from app.models import Patient, Specimen, Classification, AIModel
from app.schemas import (
    SpecimenUploadResponse, AnalysisSessionSubmit, AnalysisSessionResponse, 
    AnalysisProcessResponse, AnalysisReportResponse, AnalysisCountDetail,
    PaginatedResponse
)
from app.utils import paginate_query, get_local_now

router = APIRouter(
    prefix="/api/analyst",
    tags=["Analyst Section"]
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
logger = logging.getLogger(__name__)

# Keep original specimen files to allow cross-device previews. Set to 0/false to delete after classify.
KEEP_SPECIMEN_FILES = os.getenv("KEEP_SPECIMEN_FILES", "1").lower() not in {"0", "false", "no"}

STATUS_PENDING = "pending"
STATUS_WAITING_VALIDATION = "waiting_validation"
STATUS_REVISION = "revision"
STATUS_VALIDATED = "validated"


def check_not_locked(specimen: Specimen, request: Request):
    """Raises 403 if specimen is validated and user is Analis."""
    if specimen.status == "validated" and getattr(request.state, "role", None) == "Analis":
        raise HTTPException(
            status_code=403,
            detail="Data sudah tervalidasi (Selesai). Tidak dapat diubah. Hubungi Dokter untuk membuka kunci."
        )


def _status_label_from_specimen(status: Optional[str], total: int, validated: int) -> str:
    """Normalize backend status into a single UI label space."""
    if status == STATUS_VALIDATED:
        return "Selesai"
    if status == STATUS_WAITING_VALIDATION:
        return "Menunggu Validasi Dokter"
    if status == STATUS_REVISION:
        return "Revisi Analis"

    # Fallback for old records without/incorrect status
    if total == 0:
        return "Menunggu sampel"
    if validated == total:
        return "Selesai"
    return "Menunggu Validasi Dokter"


def _effective_gram_label(row: Classification) -> Optional[str]:
    """Use doctor validation as source of truth when present."""
    return row.validation_gram or row.classification_gram


def _cleanup_original_specimen_file(file_path: Optional[str], specimen_id: Optional[int] = None) -> None:
    """Delete original uploaded specimen file safely; keep crop files intact."""
    if not file_path:
        return

    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info("[CLEANUP] removed original specimen file specimen_id=%s path=%s", specimen_id, file_path)
    except Exception as e:
        logger.warning("[CLEANUP] failed removing original specimen file specimen_id=%s path=%s err=%s", specimen_id, file_path, e)


def _crops_dir_for_specimen(specimen_id: int) -> str:
    """Physical storage directory for crop images (served via /static and /uploads mounts)."""
    return os.path.join(UPLOAD_DIR, "crops", str(specimen_id))


class RoiIn(BaseModel):
    x: int
    y: int
    width: int
    height: int
    source: Optional[str] = "manual"


class ImageMetaIn(BaseModel):
    microscope_type: Optional[str] = None
    magnification: Optional[str] = None
    image_resolution: Optional[str] = None


class ClassifyRequest(BaseModel):
    rois: List[RoiIn]
    analyst_note: Optional[str] = None
    image_metadata: Optional[ImageMetaIn] = None
    accession_number: Optional[str] = None
    specimen_type: Optional[str] = None
    clinical_diagnosis: Optional[str] = None
    doctor_sender: Optional[str] = None

def map_predictions_to_db(gram_label: str) -> str:
    # Map from ML output format to DB constraints ['Positif', 'Negatif']
    # If it doesn't match, return original (less strict)
    label_lower = gram_label.lower()
    if "positive" in label_lower or "g+" in label_lower or "positif" in label_lower:
        return "Positif"
    elif "negative" in label_lower or "g-" in label_lower or "negatif" in label_lower:
        return "Negatif"
    return gram_label # Allow original string if no match found

def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


@router.post("/upload-specimen", response_model=SpecimenUploadResponse)
def upload_specimen(
    patient_id: int = Form(...),
    accession_number: Optional[str] = Form(None),
    specimen_type: Optional[str] = Form(None),
    doctor_sender: Optional[str] = Form(None),
    clinical_diagnosis: Optional[str] = Form(None),
    collected_at: Optional[str] = Form(None),
    received_at: Optional[str] = Form(None),
    microscope_type: Optional[str] = Form(None),
    magnification: Optional[str] = Form(None),
    image_resolution: Optional[str] = Form(None),
    analyst_note: Optional[str] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Validasi Pasien
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Pasien tidak ditemukan.")
    
    # Generate unique filename
    ext = os.path.splitext(file.filename)[1]
    unique_filename = f"specimen_{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # Save physical file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Create DB entry for Specimen
    # Auto-generate accession number if empty (not from filename)
    accession_value = (accession_number or "").strip() or None
    if not accession_value:
        accession_value = f"SPC-{get_local_now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"

    specimen = Specimen(
        patient_id=patient_id,
        accession_number=accession_value,
        specimen_type=(specimen_type or "").strip() or None,
        doctor_sender=(doctor_sender or "").strip() or None,
        clinical_diagnosis=(clinical_diagnosis or "").strip() or None,
        collected_at=_parse_datetime(collected_at),
        received_at=_parse_datetime(received_at),
        microscope_type=(microscope_type or "").strip() or None,
        magnification=(magnification or "").strip() or None,
        image_resolution=(image_resolution or "").strip() or None,
        analyst_note=(analyst_note or "").strip() or None,
        validation_status="pending",
        file_name=file.filename,
        file_path=file_path,
        uploaded_at=get_local_now()
    )
    
    db.add(specimen)
    db.commit()
    db.refresh(specimen)
    
    return specimen

@router.post("/submit", response_model=AnalysisSessionResponse, deprecated=True)
def submit_analysis(payload: AnalysisSessionSubmit, db: Session = Depends(get_db)):
    """
    Submit ke antrean dokter TANPA membuat baris Classification baru.
    Endpoint ini hanya:
    1) validasi specimen/patient,
    2) opsional update metadata klasifikasi yang sudah ada,
    3) update status specimen menjadi waiting_validation.
    """

    raise HTTPException(
        status_code=410,
        detail="Endpoint /api/analyst/submit sudah deprecated. Gunakan alur detect -> classify. Status spesimen otomatis di-set pada /api/analyst/classify/{specimen_id}.",
    )


@router.post("/detect/{specimen_id}")
def detect_specimen(specimen_id: int, request: Request, db: Session = Depends(get_db)):
    from PIL import Image
    from app.ai_pipeline import detect_and_crop

    specimen = db.query(Specimen).filter(Specimen.id == specimen_id).first()
    if not specimen:
        raise HTTPException(status_code=404, detail="Specimen not found")

    # Immutability check: Analis cannot modify validated specimens
    check_not_locked(specimen, request)

    if not specimen.file_path or not os.path.exists(specimen.file_path):
        raise HTTPException(status_code=404, detail="Specimen file not found")

    try:
        pil_image = Image.open(specimen.file_path).convert("RGB")
        detections = detect_and_crop(pil_image)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO detection failed: {e}")

    return {
        "specimen_id": specimen.id,
        "total_detected": len(detections),
        "results": [
            {
                "bbox": d["box"],
                "yolo_confidence": float(d["confidence"]),
            }
            for d in detections
        ],
    }


@router.post("/classify/{specimen_id}")
def classify_specimen(specimen_id: int, payload: ClassifyRequest, request: Request, db: Session = Depends(get_db)):
    from PIL import Image
    from app.main import _predict_tensor, MODEL_LOADED, MODEL_LOAD_ERROR

    specimen = db.query(Specimen).filter(Specimen.id == specimen_id).first()
    if not specimen:
        raise HTTPException(status_code=404, detail="Specimen not found")

    # Immutability check: Analis cannot modify validated specimens
    check_not_locked(specimen, request)

    if not specimen.file_path or not os.path.exists(specimen.file_path):
        raise HTTPException(status_code=404, detail="Specimen file not found")

    if not payload.rois:
        raise HTTPException(status_code=400, detail="ROI list is empty")

    # Automatically clear old classification results and crops folder to allow re-classification
    db.query(Classification).filter(Classification.specimen_id == specimen.id).delete()
    crops_dir = _crops_dir_for_specimen(specimen.id)
    if os.path.exists(crops_dir):
        try:
            shutil.rmtree(crops_dir)
        except Exception as e:
            logger.warning("[CLASSIFY] failed to delete old crops dir: %s", e)

    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail=f"Model belum siap: {MODEL_LOAD_ERROR}")

    active_model = db.query(AIModel).filter(AIModel.is_active == True).first()
    model_id = active_model.id if active_model else None

    img = Image.open(specimen.file_path).convert("RGB")
    crops_dir = _crops_dir_for_specimen(specimen.id)
    os.makedirs(crops_dir, exist_ok=True)

    db_rows = []
    response_rows = []
    cnn_inference_per_roi_ms = []
    cnn_total_inference_ms = 0.0
    logger.info(
        "[CLASSIFY] start specimen_id=%s patient_id=%s total_rois=%s",
        specimen.id,
        specimen.patient_id,
        len(payload.rois),
    )

    for idx, r in enumerate(payload.rois):
        x1 = int(r.x)
        y1 = int(r.y)
        x2 = int(r.x + r.width)
        y2 = int(r.y + r.height)

        # Clamp ROI ke batas gambar asli
        x1 = max(0, min(x1, img.width - 1))
        y1 = max(0, min(y1, img.height - 1))
        x2 = max(x1 + 1, min(x2, img.width))
        y2 = max(y1 + 1, min(y2, img.height))

        crop = img.crop((x1, y1, x2, y2))
        crop_filename = f"crop_{idx}.jpg"
        crop_path = os.path.join(crops_dir, crop_filename)
        crop.save(crop_path, format="JPEG")

        pred = _predict_tensor(crop)
        gram = map_predictions_to_db(pred["prediction"])
        cnn_inference_ms = float(pred.get("inference_ms", 0.0))
        cnn_total_inference_ms += cnn_inference_ms
        cnn_inference_per_roi_ms.append(cnn_inference_ms)

        row = Classification(
            patient_id=specimen.patient_id,
            specimen_id=specimen.id,
            image_file_name=crop_filename,
            image_path=crop_path,
            roi_bbox=[r.x, r.y, r.width, r.height],
            roi_source=r.source or "manual",
            classified_by_model_id=model_id,
            classification_gram=gram,
            classification_bentuk=None,
            confidence_score=float(pred["confidence"]),
            classified_by_user_id=getattr(request.state, "user_id", None),
            classified_at=get_local_now(),
        )
        db_rows.append(row)

    db.add_all(db_rows)
    db.flush()

    total_pos = sum(1 for row in db_rows if row.classification_gram == "Positif")
    total_neg = sum(1 for row in db_rows if row.classification_gram == "Negatif")

    for i, row in enumerate(db_rows):
        roi = payload.rois[i]
        image_url = f"/uploads/crops/{specimen.id}/{row.image_file_name}"
        response_rows.append(
            {
                "classification_id": row.id,
                "bbox": [roi.x, roi.y, roi.x + roi.width, roi.y + roi.height],
                "classification_gram": row.classification_gram,
                "classification_confidence": float(row.confidence_score),
                "cnn_inference_ms": round(cnn_inference_per_roi_ms[i], 2),
                "image_file_name": image_url,
                "roi_source": roi.source,
            }
        )

    specimen.total_detected = len(db_rows)
    specimen.status = STATUS_WAITING_VALIDATION
    specimen.validation_status = "pending"

    if payload.analyst_note:
        specimen.analyst_note = payload.analyst_note

    if payload.image_metadata:
        if payload.image_metadata.microscope_type:
            specimen.microscope_type = payload.image_metadata.microscope_type
        if payload.image_metadata.magnification:
            specimen.magnification = payload.image_metadata.magnification
        if payload.image_metadata.image_resolution:
            specimen.image_resolution = payload.image_metadata.image_resolution

    if payload.accession_number:
        specimen.accession_number = payload.accession_number
    if payload.specimen_type:
        specimen.specimen_type = payload.specimen_type
    if payload.clinical_diagnosis:
        specimen.clinical_diagnosis = payload.clinical_diagnosis
    if payload.doctor_sender:
        specimen.doctor_sender = payload.doctor_sender

    db.add(specimen)
    db.commit()

    # Generate etched annotated image (bounding boxes drawn directly onto the original)
    try:
        from PIL import ImageDraw, ImageFont
        annotated_img = img.copy()
        draw = ImageDraw.Draw(annotated_img)
        try:
            font = ImageFont.truetype("arial.ttf", max(14, min(img.width, img.height) // 40))
        except (OSError, IOError):
            font = ImageFont.load_default()

        for row in db_rows:
            bbox = row.roi_bbox  # [x, y, w, h]
            if not bbox:
                continue
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            is_positive = row.classification_gram == "Positif"
            color = (139, 92, 246) if is_positive else (236, 72, 153)  # purple for G+, pink for G-
            # Draw rectangle
            draw.rectangle([x, y, x + w, y + h], outline=color, width=max(2, min(w, h) // 40))
            # Draw label background
            label = f"G+ {float(row.confidence_score)*100:.0f}%" if is_positive else f"G- {float(row.confidence_score)*100:.0f}%"
            try:
                bbox_text = draw.textbbox((x, y - 18), label, font=font)
            except AttributeError:
                bbox_text = [x, y - 18, x + len(label) * 8, y - 2]
            draw.rectangle(bbox_text, fill=color)
            draw.text((x, y - 18), label, fill="white", font=font)

        annotated_filename = f"specimen_{specimen.id}_annotated.jpg"
        annotated_path = os.path.join(UPLOAD_DIR, annotated_filename)
        annotated_img.save(annotated_path, format="JPEG", quality=90)
        specimen_annotated_url = f"/uploads/{annotated_filename}"
    except Exception as e:
        logger.warning("[CLASSIFY] Could not generate annotated image: %s", e)
        specimen_annotated_url = None

    if not KEEP_SPECIMEN_FILES:
        _cleanup_original_specimen_file(specimen.file_path, specimen.id)
        _cleanup_original_specimen_file(specimen.file_path, specimen.id)

    logger.info(
        "[CLASSIFY] done specimen_id=%s total_saved=%s gram_positif=%s gram_negatif=%s status=%s cnn_total_inference_ms=%.2f cnn_avg_inference_ms=%.2f",
        specimen.id,
        len(db_rows),
        total_pos,
        total_neg,
        specimen.status,
        cnn_total_inference_ms,
        (cnn_total_inference_ms / len(db_rows)) if db_rows else 0.0,
    )
    logger.debug("[CLASSIFY] results specimen_id=%s data=%s", specimen.id, response_rows)

    return {
        "specimen_id": specimen.id,
        "total_classified": len(response_rows),
        "cnn_total_inference_ms": round(cnn_total_inference_ms, 2),
        "cnn_avg_inference_ms": round((cnn_total_inference_ms / len(response_rows)) if response_rows else 0.0, 2),
        "specimen_image_path": specimen.file_path,
        "annotated_image_url": specimen_annotated_url,
        "results": response_rows,
    }

@router.post("/process-specimen", response_model=AnalysisProcessResponse, deprecated=True)
def process_specimen_automated(
    request: Request,
    patient_id: int = Form(...),
    file: UploadFile = File(...),
    manual_rois: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    raise HTTPException(
        status_code=410,
        detail="Endpoint /api/analyst/process-specimen sudah deprecated. Gunakan /api/analyst/detect/{specimen_id} lalu /api/analyst/classify/{specimen_id}.",
    )

    from app.ai_pipeline import detect_and_crop
    from app.main import _predict_tensor, MODEL_LOADED, MODEL_LOAD_ERROR
    from PIL import Image
    import io
    import json

    # 1. Validasi Pasien
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Pasien tidak ditemukan.")
        
    # Ambil Model AI Master
    active_model = db.query(AIModel).filter(AIModel.is_active == True).first()
    model_id = active_model.id if active_model else None

    # 2. Simpan Spesimen
    specimens_dir = os.path.join(UPLOAD_DIR, "specimens")
    os.makedirs(specimens_dir, exist_ok=True)
    
    ext = os.path.splitext(file.filename)[1]
    specimen_filename = f"specimen_{uuid.uuid4().hex}{ext}"
    specimen_path = os.path.join(specimens_dir, specimen_filename)
    
    with open(specimen_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    db_specimen = Specimen(
        patient_id=patient_id,
        file_name=file.filename,
        file_path=specimen_path,
        uploaded_at=get_local_now()
    )
    db.add(db_specimen)
    db.commit()
    db.refresh(db_specimen)

    # 3. Eksekusi Pencarian / YOLO
    try:
        pil_image = Image.open(specimen_path)
        crops_data = []
        
        if manual_rois:
            rois = json.loads(manual_rois)
            for r in rois:
                x = int(r.get('x', 0))
                y = int(r.get('y', 0))
                w = int(r.get('width', 0))
                h = int(r.get('height', 0))
                
                # Pastikan rgb seperti YOLO
                rgb_img = pil_image.convert("RGB")
                cropped = rgb_img.crop((x, y, x + w, y + h))
                crops_data.append({
                    "crop": cropped,
                    "box": [x, y, x+w, y+h],
                    "confidence": 1.0 # Manual ROI has 100% detection confidence
                })
        else:
            crops_data = detect_and_crop(pil_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses gambar/YOLO: {e}")

    # Pastikan model klasifikasi siap sebelum proses crop
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail=f"Model belum siap: {MODEL_LOAD_ERROR}")

    # 4. Simpan Crop & Klasifikasi dengan ResNet
    crops_dir = os.path.join("static", "crops", str(db_specimen.id))
    os.makedirs(crops_dir, exist_ok=True)
    
    db_classifications = []
    response_crops = []
    
    # Import processedcrop schema inline kalau blm di top module
    from app.schemas import ProcessedCrop
    
    for idx, crop_info in enumerate(crops_data):
        crop_img = crop_info["crop"]
        x1, y1, x2, y2 = crop_info["box"]
        
        # Simpan file crop secara fisik
        crop_filename = f"crop_{idx}.jpg"
        crop_path = os.path.join(crops_dir, crop_filename)
        # Convert ke RGB agar JPG compatible (kalau dari RGBA)
        if crop_img.mode in ("RGBA", "P"): 
            crop_img = crop_img.convert("RGB")
        crop_img.save(crop_path, format="JPEG")
        
        # Prediksi Klasifikasi
        try:
            pred_res = _predict_tensor(crop_img)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gagal melakukan klasifikasi crop: {e}")
        gram_mapped = map_predictions_to_db(pred_res["prediction"])
        # YOLO dari user hanya "bakteri", jadi bentuk dikosongkan (diserahkan ke dokter)
        bentuk = None 

        db_class = Classification(
            patient_id=patient_id,
            specimen_id=db_specimen.id,
            image_file_name=crop_filename,
            image_path=crop_path,
            classified_by_model_id=model_id,
            classification_gram=gram_mapped,
            classification_bentuk=bentuk,
            confidence_score=float(pred_res["confidence"]),
        )
        db_classifications.append(db_class)
    
    if db_classifications:
        db.add_all(db_classifications)
        db.flush()

        for i, cls in enumerate(db_classifications):
            x1, y1, x2, y2 = crops_data[i]["box"]
            image_url = f"/static/crops/{db_specimen.id}/{cls.image_file_name}"
            response_crops.append(ProcessedCrop(
                classification_id=cls.id,
                bbox=[x1, y1, x2, y2],
                yolo_confidence=float(crops_data[i]["confidence"]),
                image_file_name=image_url,
                classification_gram=cls.classification_gram,
                classification_confidence=float(cls.confidence_score),
            ))

        # Simpan total objek terdeteksi ke tabel spesimen
        db_specimen.total_detected = len(db_classifications)
        db_specimen.status = STATUS_WAITING_VALIDATION
        db.add(db_specimen)
        db.commit()

        # Setelah klasifikasi selesai dan masuk antrean dokter, hapus file asli specimen
        _cleanup_original_specimen_file(db_specimen.file_path, db_specimen.id)
        
    return AnalysisProcessResponse(
        specimen_id=db_specimen.id,
        total_detected=len(crops_data),
        results=response_crops,
        message=f"Berhasil mendeteksi {len(crops_data)} bakteri dan dikirim ke sistem klasifikasi."
    )


@router.get("/report", response_model=PaginatedResponse[AnalysisReportResponse])
def get_analysis_report(
    date_filter: Optional[date] = Query(None, description="Format: YYYY-MM-DD"), 
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1),
    db: Session = Depends(get_db)
):
    """
    Mengambil ringkasan laporan analisis untuk setiap spesimen.
    Bisa difilter berdasarkan tanggal tertentu dan dipagination.
    """
    # Query dasar: Ambil data terbaru untuk setiap pasien
    from sqlalchemy import func, and_
    
    # Subquery untuk mencari uploaded_at terbaru per patient_id
    subquery = db.query(
        Specimen.patient_id, 
        func.max(Specimen.uploaded_at).label("max_date")
    ).group_by(Specimen.patient_id).subquery()
    
    # Query utama dengan join ke subquery
    query = db.query(Specimen).join(
        subquery, 
        and_(
            Specimen.patient_id == subquery.c.patient_id,
            Specimen.uploaded_at == subquery.c.max_date
        )
    ).join(Patient).filter(Patient.is_active == True)
    
    # Filter tanggal jika ada
    if date_filter:
        from sqlalchemy import cast, Date
        query = query.filter(cast(Specimen.uploaded_at, Date) == date_filter)
    
    # Ambil spesimen terbaru hasil filter dengan pagination
    paginated_specimens, pagination_meta = paginate_query(query.order_by(Specimen.uploaded_at.desc()), page, per_page)
    
    results = []
    for spec in paginated_specimens:
        # Hitung detail jumlah gram positif dan negatif
        classifications = db.query(Classification).filter(Classification.specimen_id == spec.id).all()

        # Backward compatibility untuk data lama (sebelum specimen_id diset)
        if not classifications:
            classifications = db.query(Classification).filter(
                Classification.image_path.like(f"%{os.path.join('crops', str(spec.id))}%")
            ).all()
        
        total = len(classifications)
        positive = sum(1 for c in classifications if _effective_gram_label(c) == "Positif")
        negative = sum(1 for c in classifications if _effective_gram_label(c) == "Negatif")
        
        # Logika Status Validasi
        validated = sum(1 for c in classifications if c.validation_gram is not None)
        
        status = _status_label_from_specimen(getattr(spec, "status", None), total, validated)
            
        results.append(AnalysisReportResponse(
            created_at=spec.uploaded_at,
            date=spec.uploaded_at.date(),
            nama_pasien=spec.patient.nama_lengkap,
            kode_sample=os.path.splitext(spec.file_name)[0],
            detail_jumlah=AnalysisCountDetail(
                positif=positive,
                negatif=negative
            ),
            status_validasi=status
        ))
        
    return PaginatedResponse(data=results, meta=pagination_meta)


@router.get("/history")
def get_analysis_history(db: Session = Depends(get_db)):
    rows = (
        db.query(Specimen, Patient)
        .join(Patient, Patient.id == Specimen.patient_id)
        .filter(Patient.is_active == True)
        .order_by(Specimen.uploaded_at.desc())
        .all()
    )

    from collections import OrderedDict
    grouped: dict[int, dict] = OrderedDict()

    for spec, patient in rows:
        pid = patient.id
        if pid not in grouped:
            grouped[pid] = {
                "patient_id": patient.id,
                "id_pasien": patient.id_pasien,
                "nama_pasien": patient.nama_lengkap,
                "nik": patient.nik,
                "earliest_upload": spec.uploaded_at.strftime("%Y-%m-%d %H:%M") if spec.uploaded_at else None,
                "total_specimens": 0,
                "total_g_positif": 0,
                "total_g_negatif": 0,
                "specimens": [],
                "overall_status": None,
            }

        g = grouped[pid]
        g["total_specimens"] += 1

        classifications = db.query(Classification).filter(Classification.specimen_id == spec.id).all()
        if not classifications:
            legacy_rows = db.query(Classification).filter(
                Classification.image_path.like(f"%{os.path.join('crops', str(spec.id))}%")
            ).all()
            if legacy_rows:
                classifications = legacy_rows

        total_pos = sum(1 for c in classifications if _effective_gram_label(c) == "Positif")
        total_neg = sum(1 for c in classifications if _effective_gram_label(c) == "Negatif")
        total_klasifikasi = len(classifications)
        belum_validasi = sum(1 for c in classifications if c.validation_gram is None)

        g["total_g_positif"] += total_pos
        g["total_g_negatif"] += total_neg

        spec_status = _status_label_from_specimen(spec.status, total_klasifikasi, total_klasifikasi - belum_validasi)
        if g["overall_status"] is None:
            if spec.status == "revision":
                g["overall_status"] = "Revisi Analis"
            elif spec.status == "waiting_validation" or spec_status == "Menunggu Validasi Dokter":
                g["overall_status"] = "Menunggu Validasi"

        tanggal_validasi = None
        if spec.status == STATUS_VALIDATED and classifications:
            latest_update = max((c.updated_at for c in classifications if c.updated_at), default=None)
            if latest_update:
                tanggal_validasi = latest_update.strftime("%Y-%m-%d %H:%M")

        g["specimens"].append({
            "id_specimen": spec.id,
            "accession_number": spec.accession_number,
            "tanggal": spec.uploaded_at.strftime("%Y-%m-%d %H:%M") if spec.uploaded_at else None,
            "tanggal_validasi": tanggal_validasi,
            "total_g_positif": total_pos,
            "total_g_negatif": total_neg,
            "status": spec_status,
        })

    for v in grouped.values():
        if v["overall_status"] is None:
            v["overall_status"] = "Selesai"

    return list(grouped.values())


@router.get("/dashboard-chart")
def get_analyst_dashboard_chart(filter: str = Query("Harian"), db: Session = Depends(get_db)):
    """Mengembalikan data chart berbasis PASIEN (bukan spesimen) untuk dashboard analis.
    
    - 'selesai': jumlah pasien dengan semua spesimen sudah tervalidasi
    - 'pending': jumlah pasien yang masih memiliki spesimen menunggu diproses/divalidasi
    """
    from datetime import date, timedelta
    from sqlalchemy import func
    today = date.today()

    valid_filters = {"Hari Ini", "Harian", "Mingguan", "Bulanan", "Tahunan"}
    if filter not in valid_filters:
        filter = "Harian"

    def _count_patients_by_status(start_date, end_date, status_condition):
        """Count distinct patients whose latest specimen matches status_condition."""
        # Subquery: latest uploaded_at per patient
        latest_spec_subq = (
            db.query(
                Specimen.patient_id,
                func.max(Specimen.uploaded_at).label("max_uploaded")
            )
            .join(Patient, Patient.id == Specimen.patient_id)
            .filter(Patient.is_active == True)
            .filter(func.date(Specimen.uploaded_at) >= start_date)
            .filter(func.date(Specimen.uploaded_at) <= end_date)
            .group_by(Specimen.patient_id)
            .subquery()
        )

        # Subquery: status of the latest specimen per patient
        latest_status_subq = (
            db.query(
                Specimen.patient_id,
                Specimen.status.label("latest_status"),
            )
            .join(
                latest_spec_subq,
                func.coalesce(Specimen.patient_id, -1) == func.coalesce(latest_spec_subq.c.patient_id, -1),
                # Filter will be applied below, we just need the subquery structure
            )
            .filter(Specimen.uploaded_at == latest_spec_subq.c.max_uploaded)
            .subquery()
        )

        # Use OR for status matching
        return db.query(func.count(func.distinct(Specimen.patient_id))).join(
            Patient, Patient.id == Specimen.patient_id
        ).filter(
            Patient.is_active == True,
            func.date(Specimen.uploaded_at) >= start_date,
            func.date(Specimen.uploaded_at) <= end_date,
            status_condition
        ).scalar() or 0

    result = []

    if filter == "Hari Ini":
        intervals = [
            ("08:00", 8, 10), ("10:00", 10, 12), ("12:00", 12, 14),
            ("14:00", 14, 16), ("16:00", 16, 18), ("18:00", 18, 20), ("20:00", 20, 22)
        ]
        for label, start_h, end_h in intervals:
            selesai = (
                db.query(func.count(func.distinct(Patient.id)))
                .join(Specimen, Specimen.patient_id == Patient.id)
                .filter(Patient.is_active == True)
                .filter(func.date(Specimen.uploaded_at) == today)
                .filter(func.extract('hour', Specimen.uploaded_at) >= start_h)
                .filter(func.extract('hour', Specimen.uploaded_at) < end_h)
                .filter(Specimen.status == "validated")
                .scalar() or 0
            )
            pending = (
                db.query(func.count(func.distinct(Patient.id)))
                .join(Specimen, Specimen.patient_id == Patient.id)
                .filter(Patient.is_active == True)
                .filter(func.date(Specimen.uploaded_at) == today)
                .filter(func.extract('hour', Specimen.uploaded_at) >= start_h)
                .filter(func.extract('hour', Specimen.uploaded_at) < end_h)
                .filter(Specimen.status.in_(["pending", "waiting_validation"]))
                .scalar() or 0
            )
            result.append({"name": label, "masuk": pending + selesai, "selesai": selesai, "validated": selesai, "pending": pending})

    elif filter == "Harian":
        day_labels = {0: "Sen", 1: "Sel", 2: "Rab", 3: "Kam", 4: "Jum", 5: "Sab", 6: "Min"}
        for i in range(6, -1, -1):
            d = today - timedelta(days=i)
            selesai = (
                db.query(func.count(func.distinct(Patient.id)))
                .join(Specimen, Specimen.patient_id == Patient.id)
                .filter(Patient.is_active == True)
                .filter(func.date(Specimen.uploaded_at) == d)
                .filter(Specimen.status == "validated")
                .scalar() or 0
            )
            pending = (
                db.query(func.count(func.distinct(Patient.id)))
                .join(Specimen, Specimen.patient_id == Patient.id)
                .filter(Patient.is_active == True)
                .filter(func.date(Specimen.uploaded_at) == d)
                .filter(Specimen.status.in_(["pending", "waiting_validation"]))
                .scalar() or 0
            )
            result.append({"name": day_labels[d.weekday()], "masuk": pending + selesai, "selesai": selesai, "validated": selesai, "pending": pending})

    elif filter == "Mingguan":
        for i in range(4):
            start_d = today - timedelta(days=(4 - i) * 7)
            end_d = today - timedelta(days=(3 - i) * 7)
            selesai = (
                db.query(func.count(func.distinct(Patient.id)))
                .join(Specimen, Specimen.patient_id == Patient.id)
                .filter(Patient.is_active == True)
                .filter(func.date(Specimen.uploaded_at) >= start_d)
                .filter(func.date(Specimen.uploaded_at) < end_d)
                .filter(Specimen.status == "validated")
                .scalar() or 0
            )
            pending = (
                db.query(func.count(func.distinct(Patient.id)))
                .join(Specimen, Specimen.patient_id == Patient.id)
                .filter(Patient.is_active == True)
                .filter(func.date(Specimen.uploaded_at) >= start_d)
                .filter(func.date(Specimen.uploaded_at) < end_d)
                .filter(Specimen.status.in_(["pending", "waiting_validation"]))
                .scalar() or 0
            )
            result.append({"name": f"Mg {i+1}", "masuk": pending + selesai, "selesai": selesai, "validated": selesai, "pending": pending})

    elif filter == "Bulanan":
        months = [("Jan", 1), ("Feb", 2), ("Mar", 3), ("Apr", 4), ("Mei", 5), ("Jun", 6),
                  ("Jul", 7), ("Agt", 8), ("Sep", 9), ("Okt", 10), ("Nov", 11), ("Des", 12)]
        curr_year = today.year
        for label, m_idx in months:
            selesai = (
                db.query(func.count(func.distinct(Patient.id)))
                .join(Specimen, Specimen.patient_id == Patient.id)
                .filter(Patient.is_active == True)
                .filter(func.extract('year', Specimen.uploaded_at) == curr_year)
                .filter(func.extract('month', Specimen.uploaded_at) == m_idx)
                .filter(Specimen.status == "validated")
                .scalar() or 0
            )
            pending = (
                db.query(func.count(func.distinct(Patient.id)))
                .join(Specimen, Specimen.patient_id == Patient.id)
                .filter(Patient.is_active == True)
                .filter(func.extract('year', Specimen.uploaded_at) == curr_year)
                .filter(func.extract('month', Specimen.uploaded_at) == m_idx)
                .filter(Specimen.status.in_(["pending", "waiting_validation"]))
                .scalar() or 0
            )
            result.append({"name": label, "masuk": pending + selesai, "selesai": selesai, "validated": selesai, "pending": pending})

    elif filter == "Tahunan":
        curr_year = today.year
        for i in range(3, -1, -1):
            y = curr_year - i
            selesai = (
                db.query(func.count(func.distinct(Patient.id)))
                .join(Specimen, Specimen.patient_id == Patient.id)
                .filter(Patient.is_active == True)
                .filter(func.extract('year', Specimen.uploaded_at) == y)
                .filter(Specimen.status == "validated")
                .scalar() or 0
            )
            pending = (
                db.query(func.count(func.distinct(Patient.id)))
                .join(Specimen, Specimen.patient_id == Patient.id)
                .filter(Patient.is_active == True)
                .filter(func.extract('year', Specimen.uploaded_at) == y)
                .filter(Specimen.status.in_(["pending", "waiting_validation"]))
                .scalar() or 0
            )
            result.append({"name": str(y), "masuk": pending + selesai, "selesai": selesai, "validated": selesai, "pending": pending})

    return result


@router.get("/dashboard-stats")
def get_analyst_dashboard_stats(filter: str = Query("Harian"), db: Session = Depends(get_db)):
    """Mengembalikan metrik laboratorium untuk dashboard analis.
    
    - pending_queue_count: Jumlah spesimen dengan status 'pending' (belum diproses)
    - rejected_error_count: Jumlah spesimen dengan status error/ditolak
    - median_processing_time_seconds: Median waktu proses dari upload hingga klasifikasi selesai
    - throughput: Data grafik spesimen diterima vs terkirim ke dokter
    """
    from datetime import date, timedelta
    from sqlalchemy import func, literal_column
    import statistics

    # Helper to count distinct patients with active status
    def _count_patients(*filters):
        return db.query(func.count(func.distinct(Patient.id))).join(
            Specimen, Specimen.patient_id == Patient.id
        ).filter(
            Patient.is_active == True,
            *filters
        ).scalar() or 0

    # 1. Pending Queue Count (patients with pending specimens)
    pending_count = _count_patients(
        Specimen.status == STATUS_PENDING
    )

    # 2. Rejected / Error Patients Count
    rejected_count = _count_patients(
        Specimen.validation_status == "rejected"
    )

    error_count = _count_patients(
        Specimen.status != STATUS_PENDING,
        Specimen.status != STATUS_WAITING_VALIDATION,
        Specimen.status != STATUS_VALIDATED,
    )

    total_rejected = rejected_count + error_count

    # 3. Median Processing Time
    # Processing time = time from upload to when classification was made (classified_at - uploaded_at)
    processing_times = (
        db.query(
            func.extract('epoch', Classification.classified_at - Specimen.uploaded_at).label('processing_seconds')
        )
        .join(Specimen, Classification.specimen_id == Specimen.id)
        .filter(Classification.classified_at.isnot(None))
        .filter(Specimen.uploaded_at.isnot(None))
        .filter(Classification.classified_at >= Specimen.uploaded_at)
        .all()
    )

    times_in_seconds = [float(row.processing_seconds) for row in processing_times if row.processing_seconds is not None]
    median_processing_time = round(statistics.median(times_in_seconds), 1) if times_in_seconds else 0

    # 4. Throughput Data (Pasien dengan spesimen diterima vs dikirim ke dokter)
    today = date.today()
    throughput = []

    # Helper to count distinct patients joined with active filter
    def _count_throughput_patients(*filters):
        return db.query(func.count(func.distinct(Patient.id))).join(
            Specimen, Specimen.patient_id == Patient.id
        ).filter(
            Patient.is_active == True,
            *filters
        ).scalar() or 0

    def _get_periods(f):
        if f == "Hari Ini":
            intervals = [
                ("08:00", 8, 10), ("10:00", 10, 12), ("12:00", 12, 14),
                ("14:00", 14, 16), ("16:00", 16, 18), ("18:00", 18, 20), ("20:00", 20, 22)
            ]
            for label, start_h, end_h in intervals:
                diterima = _count_throughput_patients(
                    func.date(Specimen.uploaded_at) == today,
                    func.extract('hour', Specimen.uploaded_at) >= start_h,
                    func.extract('hour', Specimen.uploaded_at) < end_h
                )

                terkirim = _count_throughput_patients(
                    Specimen.status.in_([STATUS_WAITING_VALIDATION, STATUS_VALIDATED]),
                    func.date(Specimen.uploaded_at) == today,
                    func.extract('hour', Specimen.uploaded_at) >= start_h,
                    func.extract('hour', Specimen.uploaded_at) < end_h
                )

                throughput.append({"name": label, "diterima": diterima, "terkirim": terkirim})
        elif f == "Harian":
            day_labels = {0: "Sen", 1: "Sel", 2: "Rab", 3: "Kam", 4: "Jum", 5: "Sab", 6: "Min"}
            for i in range(6, -1, -1):
                d = today - timedelta(days=i)
                diterima = _count_throughput_patients(
                    func.date(Specimen.uploaded_at) == d
                )

                terkirim = _count_throughput_patients(
                    Specimen.status.in_([STATUS_WAITING_VALIDATION, STATUS_VALIDATED]),
                    func.date(Specimen.uploaded_at) == d
                )

                throughput.append({"name": day_labels[d.weekday()], "diterima": diterima, "terkirim": terkirim})
        elif f == "Mingguan":
            for i in range(4):
                start_d = today - timedelta(days=(4 - i) * 7)
                end_d = today - timedelta(days=(3 - i) * 7)
                diterima = _count_throughput_patients(
                    func.date(Specimen.uploaded_at) >= start_d,
                    func.date(Specimen.uploaded_at) < end_d
                )

                terkirim = _count_throughput_patients(
                    Specimen.status.in_([STATUS_WAITING_VALIDATION, STATUS_VALIDATED]),
                    func.date(Specimen.uploaded_at) >= start_d,
                    func.date(Specimen.uploaded_at) < end_d
                )

                throughput.append({"name": f"Mg {i+1}", "diterima": diterima, "terkirim": terkirim})
        elif f == "Bulanan":
            months = [("Jan", 1), ("Feb", 2), ("Mar", 3), ("Apr", 4), ("Mei", 5), ("Jun", 6),
                      ("Jul", 7), ("Agt", 8), ("Sep", 9), ("Okt", 10), ("Nov", 11), ("Des", 12)]
            curr_year = today.year
            for label, m_idx in months:
                diterima = _count_throughput_patients(
                    func.extract('year', Specimen.uploaded_at) == curr_year,
                    func.extract('month', Specimen.uploaded_at) == m_idx
                )

                terkirim = _count_throughput_patients(
                    Specimen.status.in_([STATUS_WAITING_VALIDATION, STATUS_VALIDATED]),
                    func.extract('year', Specimen.uploaded_at) == curr_year,
                    func.extract('month', Specimen.uploaded_at) == m_idx
                )

                throughput.append({"name": label, "diterima": diterima, "terkirim": terkirim})
        elif f == "Tahunan":
            curr_year = today.year
            for i in range(3, -1, -1):
                y = curr_year - i
                diterima = _count_throughput_patients(
                    func.extract('year', Specimen.uploaded_at) == y
                )

                terkirim = _count_throughput_patients(
                    Specimen.status.in_([STATUS_WAITING_VALIDATION, STATUS_VALIDATED]),
                    func.extract('year', Specimen.uploaded_at) == y
                )

                throughput.append({"name": str(y), "diterima": diterima, "terkirim": terkirim})

    _get_periods(filter)

    # 5. Waiting & validated count (active patients only)
    waiting_count = _count_patients(
        Specimen.status == STATUS_WAITING_VALIDATION
    )
    validated_count = _count_patients(
        Specimen.status == STATUS_VALIDATED
    )

    # 6. Historical comparison for queue counts
    def _get_previous_count(days_ago: int, status_filter: str):
        from datetime import timedelta
        start_prev = today - timedelta(days=days_ago * 2)
        end_prev = today - timedelta(days=days_ago)
        return _count_throughput_patients(
            Specimen.status == status_filter,
            func.date(Specimen.uploaded_at) >= start_prev,
            func.date(Specimen.uploaded_at) < end_prev,
        )

    trend_period_map = {"Hari Ini": 1, "Harian": 7, "Mingguan": 30, "Bulanan": 90, "Tahunan": 365}
    ref_days = trend_period_map.get(filter, 30)

    prev_pending = _get_previous_count(ref_days, STATUS_PENDING)
    prev_waiting = _get_previous_count(ref_days, STATUS_WAITING_VALIDATION)
    prev_validated = _get_previous_count(ref_days, STATUS_VALIDATED)
    prev_revision = _get_previous_count(ref_days, STATUS_REVISION)
    prev_diproses = sum(t.get("diterima", 0) for t in throughput) if len(throughput) >= 2 else 0

    current_validated = validated_count
    current_revision = _count_patients(Specimen.status == STATUS_REVISION)
    current_diproses = sum(t.get("diterima", 0) for t in throughput)

    def _calc_trend(current: int, previous: int):
        if previous == 0 and current == 0:
            return None
        if previous == 0:
            return {"direction": "up", "value": 100, "current": current, "previous": previous}
        change = ((current - previous) / previous) * 100
        return {
            "direction": "up" if change > 0 else ("down" if change < 0 else "stable"),
            "value": round(abs(change), 1),
            "current": current,
            "previous": previous,
        }

    # Hitung inflow (specimen baru pending) & outflow (specimen yang diklasifikasi)
    from datetime import timedelta
    _period_start = today - timedelta(days=trend_period_map.get(filter, 30))

    inflow_pending = db.query(func.count(Specimen.id)).join(
        Patient, Patient.id == Specimen.patient_id
    ).filter(
        Patient.is_active == True,
        Specimen.status == STATUS_PENDING,
        func.date(Specimen.uploaded_at) >= _period_start,
        func.date(Specimen.uploaded_at) <= today,
    ).scalar() or 0

    outflow_pending = db.query(func.count(Specimen.id)).join(
        Patient, Patient.id == Specimen.patient_id
    ).join(
        Classification, Classification.specimen_id == Specimen.id
    ).filter(
        Patient.is_active == True,
        func.date(Classification.classified_at) >= _period_start,
        func.date(Classification.classified_at) <= today,
    ).scalar() or 0

    inflow_waiting = outflow_pending
    outflow_waiting = db.query(func.count(Specimen.id)).join(
        Patient, Patient.id == Specimen.patient_id
    ).filter(
        Patient.is_active == True,
        Specimen.status == STATUS_VALIDATED,
        func.date(Specimen.validated_at) >= _period_start,
        func.date(Specimen.validated_at) <= today,
    ).scalar() or 0

    return {
        "pending_queue_count": pending_count,
        "waiting_validation_count": waiting_count,
        "validated_count": validated_count,
        "rejected_error_count": total_rejected,
        "median_processing_time_seconds": median_processing_time,
        "median_processing_time_display": _format_duration(median_processing_time),
        "throughput": throughput,
        "trends": {
            "pending": _calc_trend(pending_count, prev_pending),
            "waiting": _calc_trend(waiting_count, prev_waiting),
            "diproses": _calc_trend(current_validated, prev_validated),
            "revision": _calc_trend(current_revision, prev_revision),
        },
        "flow": {
            "pending_inflow": inflow_pending,
            "pending_outflow": outflow_pending,
            "waiting_inflow": inflow_waiting,
            "waiting_outflow": outflow_waiting,
        },
    }


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable Indonesian duration."""
    if seconds < 60:
        return f"{int(seconds)} detik"
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)} menit {int(seconds % 60)} detik"
    hours = minutes / 60
    return f"{int(hours)} jam {int(minutes % 60)} menit"

