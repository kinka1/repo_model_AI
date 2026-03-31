import os
import shutil
import uuid
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Patient, Specimen, Classification, AIModel
from app.schemas import SpecimenUploadResponse, AnalysisSessionSubmit, AnalysisSessionResponse, AnalysisProcessResponse

router = APIRouter(
    prefix="/api/analysis",
    tags=["Analysis Process"]
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

@router.post("/process-specimen", response_model=AnalysisProcessResponse)
def process_specimen_automated(
    request: Request,
    patient_id: int = Form(...),
    file: UploadFile = File(...),
    manual_rois: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    from app.ai_pipeline import detect_and_crop
    from app.main import _predict_tensor
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

    # 4. Simpan Crop & Klasifikasi dengan ResNet
    crops_dir = os.path.join(UPLOAD_DIR, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    
    db_classifications = []
    response_crops = []
    
    # Import processedcrop schema inline kalau blm di top module
    from app.schemas import ProcessedCrop
    
    for idx, crop_info in enumerate(crops_data):
        crop_img = crop_info["crop"]
        bbox = crop_info["box"]
        yolo_conf = crop_info["confidence"]
        
        # Simpan file crop secara fisik
        crop_filename = f"crop_{db_specimen.id}_{idx}_{uuid.uuid4().hex[:6]}.jpg"
        crop_path = os.path.join(crops_dir, crop_filename)
        # Convert ke RGB agar JPG compatible (kalau dari RGBA)
        if crop_img.mode in ("RGBA", "P"): 
            crop_img = crop_img.convert("RGB")
        crop_img.save(crop_path, format="JPEG")
        
        # Prediksi Klasifikasi
        try:
            pred_res = _predict_tensor(crop_img)
            resnet_conf = pred_res["confidence"]
            gram_class = pred_res["prediction"]
        except Exception:
            # Fallback jika model resnet belum di load / error
            resnet_conf = 0.0
            gram_class = "Error"
            
        gram_mapped = map_predictions_to_db(gram_class)
        # YOLO dari user hanya "bakteri", jadi bentuk dikosongkan (diserahkan ke dokter)
        bentuk = None 

        db_class = Classification(
            patient_id=patient_id,
            image_file_name=crop_filename,
            image_path=crop_path,
            classified_by_model_id=model_id,
            classification_gram=gram_mapped,
            classification_bentuk=bentuk,
            confidence_score=float(resnet_conf),
        )
        db_classifications.append(db_class)
    
    if db_classifications:
        db.bulk_save_objects(db_classifications)
        
        # Simpan total objek terdeteksi ke tabel spesimen
        db_specimen.total_detected = len(db_classifications)
        db.add(db_specimen)
        
        db.commit()
        
        # Fetch inserted ones for their IDs to return to API
        # Bulk save nggak return IDs, so query them
        inserted_classes = db.query(Classification).filter(
            Classification.patient_id == patient_id,
            Classification.image_path.like(f"%crop_{db_specimen.id}_%")
        ).order_by(Classification.id.asc()).all()
        # Because ordering and counting aren't strictly 1-to-1 thread safe, we map them sequentially
        # Assumes single request processes these.
        for i, ic in enumerate(inserted_classes):
            if i < len(crops_data):
                try:
                    cinfo = crops_data[i]
                    image_url = f"{request.base_url}static/crops/{ic.image_file_name}"
                    response_crops.append(ProcessedCrop(
                        classification_id=ic.id,
                        bbox=cinfo["box"],
                        yolo_confidence=float(cinfo["confidence"]),
                        image_file_name=image_url,
                        classification_gram=ic.classification_gram,
                        classification_confidence=float(ic.confidence_score)
                    ))
                except Exception:
                    pass
        
    return AnalysisProcessResponse(
        specimen_id=db_specimen.id,
        total_detected=len(crops_data),
        results=response_crops,
        message=f"Berhasil mendeteksi {len(crops_data)} bakteri dan dikirim ke sistem klasifikasi."
    )
