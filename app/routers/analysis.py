import os
import shutil
import uuid
import logging
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
from app.utils import paginate_query

router = APIRouter(
    prefix="/api/analyst",
    tags=["Analyst Section"]
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
logger = logging.getLogger(__name__)

STATUS_PENDING = "pending"
STATUS_WAITING_VALIDATION = "waiting_validation"
STATUS_VALIDATED = "validated"


def _status_label_from_specimen(status: Optional[str], total: int, validated: int) -> str:
    """Normalize backend status into a single UI label space."""
    if status == STATUS_VALIDATED:
        return "Selesai"
    if status == STATUS_WAITING_VALIDATION:
        return "Menunggu Validasi Dokter"

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


class ClassifyRequest(BaseModel):
    rois: List[RoiIn]

def map_predictions_to_db(gram_label: str) -> str:
    # Map from ML output format to DB constraints ['Positif', 'Negatif']
    # If it doesn't match, return original (less strict)
    label_lower = gram_label.lower()
    if "positive" in label_lower or "g+" in label_lower or "positif" in label_lower:
        return "Positif"
    elif "negative" in label_lower or "g-" in label_lower or "negatif" in label_lower:
        return "Negatif"
    return gram_label # Allow original string if no match found

@router.post("/upload-specimen", response_model=SpecimenUploadResponse)
def upload_specimen(
    patient_id: int = Form(...),
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
    specimen = Specimen(
        patient_id=patient_id,
        file_name=file.filename,
        file_path=file_path,
        uploaded_at=datetime.utcnow()
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
def detect_specimen(specimen_id: int, db: Session = Depends(get_db)):
    from PIL import Image
    from app.ai_pipeline import detect_and_crop

    specimen = db.query(Specimen).filter(Specimen.id == specimen_id).first()
    if not specimen:
        raise HTTPException(status_code=404, detail="Specimen not found")

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
def classify_specimen(specimen_id: int, payload: ClassifyRequest, db: Session = Depends(get_db)):
    from PIL import Image
    from app.main import _predict_tensor, MODEL_LOADED, MODEL_LOAD_ERROR

    specimen = db.query(Specimen).filter(Specimen.id == specimen_id).first()
    if not specimen:
        raise HTTPException(status_code=404, detail="Specimen not found")

    if not specimen.file_path or not os.path.exists(specimen.file_path):
        raise HTTPException(status_code=404, detail="Specimen file not found")

    if not payload.rois:
        raise HTTPException(status_code=400, detail="ROI list is empty")

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
            classified_by_model_id=model_id,
            classification_gram=gram,
            classification_bentuk=None,
            confidence_score=float(pred["confidence"]),
        )
        db_rows.append(row)

    db.add_all(db_rows)
    db.flush()

    total_pos = sum(1 for row in db_rows if row.classification_gram == "Positif")
    total_neg = sum(1 for row in db_rows if row.classification_gram == "Negatif")

    for i, row in enumerate(db_rows):
        roi = payload.rois[i]
        image_url = f"/static/crops/{specimen.id}/{row.image_file_name}"
        response_rows.append(
            {
                "classification_id": row.id,
                "bbox": [roi.x, roi.y, roi.x + roi.width, roi.y + roi.height],
                "classification_gram": row.classification_gram,
                "classification_confidence": float(row.confidence_score),
                "cnn_inference_ms": round(cnn_inference_per_roi_ms[i], 2),
                "image_file_name": image_url,
                "source": roi.source,
            }
        )

    specimen.total_detected = len(db_rows)
    specimen.status = STATUS_WAITING_VALIDATION
    db.add(specimen)
    db.commit()

    # Setelah klasifikasi selesai dan masuk antrean dokter, hapus file asli specimen
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
        uploaded_at=datetime.utcnow()
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
            image_url = f"{request.base_url}static/crops/{db_specimen.id}/{cls.image_file_name}"
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
    ).join(Patient)
    
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
        .order_by(Specimen.uploaded_at.desc())
        .all()
    )

    history_list = []
    for spec, patient in rows:
        classifications = db.query(Classification).filter(Classification.specimen_id == spec.id).all()

        # Backward compatibility untuk data lama (sebelum specimen_id diset)
        if not classifications:
            legacy_rows = db.query(Classification).filter(
                Classification.image_path.like(f"%{os.path.join('crops', str(spec.id))}%")
            ).all()
            if legacy_rows:
                classifications = legacy_rows

        total_g_positif = sum(1 for c in classifications if _effective_gram_label(c) == "Positif")
        total_g_negatif = sum(1 for c in classifications if _effective_gram_label(c) == "Negatif")
        total_klasifikasi = len(classifications)
        belum_validasi = sum(1 for c in classifications if c.validation_gram is None)

        status = _status_label_from_specimen(
            spec.status,
            total_klasifikasi,
            total_klasifikasi - belum_validasi,
        )

        history_list.append(
            {
                "id_specimen": spec.id,
                "nama_pasien": patient.nama_lengkap,
                "tanggal": spec.uploaded_at.strftime("%Y-%m-%d %H:%M") if spec.uploaded_at else None,
                "total_g_positif": total_g_positif,
                "total_g_negatif": total_g_negatif,
                "status": status,
            }
        )

    return history_list
