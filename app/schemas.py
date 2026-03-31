from pydantic import BaseModel, Field
from datetime import date, datetime
from typing import List, Optional

# ===============================
# PATIENT SCHEMAS
# ===============================

class PatientBase(BaseModel):
    nama_lengkap: str = Field(..., max_length=100)
    jenis_kelamin: str = Field(..., description="Laki-Laki atau Perempuan")
    tanggal_lahir: date
    alamat: Optional[str] = None
    no_telepon: Optional[str] = None

class PatientCreate(PatientBase):
    pass

class PatientResponse(PatientBase):
    id: int
    id_pasien: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# ===============================
# ANALYSIS SCHEMAS
# ===============================

class SpecimenUploadResponse(BaseModel):
    id: int
    patient_id: int
    file_name: str
    file_path: str
    uploaded_at: datetime

    class Config:
        from_attributes = True

class PredictionResult(BaseModel):
    filename: str
    prediction: str
    class_id: int
    confidence: float
    probabilities: dict
    inference_ms: float

class BatchPredictResponse(BaseModel):
    total_images: int
    predictions: List[PredictionResult]
    summary: dict

class ClassificationSubmit(BaseModel):
    image_file_name: str
    # Simpan hasil crop AI
    classification_gram: str
    classification_bentuk: Optional[str] = None
    confidence_score: float

class AnalysisSessionSubmit(BaseModel):
    patient_id: int
    specimen_id: Optional[int] = None
    # Array dari hasil crop
    crops: List[ClassificationSubmit]

class AnalysisSessionResponse(BaseModel):
    message: str
    total_submitted: int

class ProcessedCrop(BaseModel):
    classification_id: int
    bbox: List[int]
    yolo_confidence: float
    image_file_name: str
    classification_gram: str
    classification_confidence: float

class AnalysisProcessResponse(BaseModel):
    specimen_id: int
    total_detected: int
    results: List[ProcessedCrop]
    message: str
