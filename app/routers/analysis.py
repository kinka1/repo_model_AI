import os
import shutil
import uuid
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.models import Patient, Specimen, Classification, AIModel
from app.schemas import SpecimenUploadResponse, AnalysisSessionSubmit, AnalysisSessionResponse, AnalysisProcessResponse

router = APIRouter(
    prefix="/api/analysis",
    tags=["Analysis Process"]
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


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
    if "Positive" in gram_label or "G+" in gram_label:
        return "Positif"
    elif "Negative" in gram_label or "G-" in gram_label:
        return "Negatif"
    return "Positif" # Default safe fallback

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

@router.post("/submit", response_model=AnalysisSessionResponse)
def submit_analysis(payload: AnalysisSessionSubmit, db: Session = Depends(get_db)):
    """
    Simpan hasil batch model dari image-image potongan spesimen ke tabel Classifications.
    Setiap baris yang baru disave akan diabaikan `validation_gram` nya (Null) sehingga otomatis masuk ke "Antrean Dokter".
    """
    
    # Cek pasien
    patient = db.query(Patient).filter(Patient.id == payload.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Pasien tidak ditemukan. Tolong upload dengan ID asli.")
        
    # Ambil AI Model yang aktif (Asumsi hanya ada 1 untuk session ini)
    active_model = db.query(AIModel).filter(AIModel.is_active == True).first()
    model_id = active_model.id if active_model else None
    
    classifications_to_save = []
    
    for crop in payload.crops:
        # Pengecekan constraint Enum
        gram_mapped = map_predictions_to_db(crop.classification_gram)
        bentuk = crop.classification_bentuk if crop.classification_bentuk in ['Batang', 'Kokus'] else None
        
        entry = Classification(
            patient_id=payload.patient_id,
            image_file_name=crop.image_file_name,
            image_path=f"uploads/crops/{crop.image_file_name}", # Bisa diganti berdasarkan base API logic image upload anda
            classified_by_model_id=model_id,
            classification_gram=gram_mapped,
            classification_bentuk=bentuk,
            confidence_score=crop.confidence_score,
            # validation_gram dan validation_bentuk dibiarkan None supaya terlempar ke timelines dokter
        )
        
        classifications_to_save.append(entry)
        
    db.bulk_save_objects(classifications_to_save)
    db.commit()
    
    return AnalysisSessionResponse(
        message="Sesi analisis berhasil disimpan dan dimasukkan ke antrean validasi dokter.",
        total_submitted=len(classifications_to_save)
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
    crops_dir = os.path.join("static", "crops", str(specimen.id))
    os.makedirs(crops_dir, exist_ok=True)

    db_rows = []
    response_rows = []

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

        row = Classification(
            patient_id=specimen.patient_id,
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

    for i, row in enumerate(db_rows):
        roi = payload.rois[i]
        image_url = f"/static/crops/{specimen.id}/{row.image_file_name}"
        response_rows.append(
            {
                "classification_id": row.id,
                "bbox": [roi.x, roi.y, roi.x + roi.width, roi.y + roi.height],
                "classification_gram": row.classification_gram,
                "classification_confidence": float(row.confidence_score),
                "image_file_name": image_url,
                "source": roi.source,
            }
        )

    specimen.total_detected = len(db_rows)
    db.add(specimen)
    db.commit()

    return {
        "specimen_id": specimen.id,
        "total_classified": len(response_rows),
        "results": response_rows,
    }

@router.post("/process-specimen", response_model=AnalysisProcessResponse)
def process_specimen_automated(
    request: Request,
    patient_id: int = Form(...),
    file: UploadFile = File(...),
    manual_rois: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
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
        db.add(db_specimen)
        db.commit()
        
    return AnalysisProcessResponse(
        specimen_id=db_specimen.id,
        total_detected=len(crops_data),
        results=response_crops,
        message=f"Berhasil mendeteksi {len(crops_data)} bakteri dan dikirim ke sistem klasifikasi."
    )

@router.delete("/cleanup/{specimen_id}")
def cleanup_specimen(specimen_id: int, db: Session = Depends(get_db)):
    specimen = db.query(Specimen).filter(Specimen.id == specimen_id).first()

    if not specimen:
        return {"message": "Specimen sudah terhapus (mengabaikan request ganda dari frontend)."}

    # 1. Hapus file fisik gambar UTAMA
    if specimen.file_path and os.path.exists(specimen.file_path):
        os.remove(specimen.file_path)

    # 2. HAPUS FOLDER CROP SPESIMEN INSTAN
    # Menghapus static/crops/{specimen_id}/ beserta seluruh isinya
    crop_folder = os.path.join("static", "crops", str(specimen_id))
    if os.path.exists(crop_folder):
        shutil.rmtree(crop_folder, ignore_errors=True)

    # 3. Hapus record database untuk crop milik specimen ini saja
    crop_folder_fragment = os.path.join("crops", str(specimen_id))
    crops_in_db = db.query(Classification).filter(
        Classification.image_path.like(f"%{crop_folder_fragment}%")
    ).all()

    for crop in crops_in_db:
        db.delete(crop)

    # 4. Hapus record spesimen utama dari database
    db.delete(specimen)

    db.commit()

    return {
        "message": f"Folder spesimen {specimen_id} dan seluruh isinya berhasil dihapus secara atomik."
    }
