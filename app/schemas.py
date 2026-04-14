from pydantic import BaseModel, Field
from datetime import date, datetime
from typing import List, Optional, Generic, TypeVar

T = TypeVar("T")

# ===============================
# PAGINATION SCHEMAS
# ===============================

class PaginationMeta(BaseModel):
    total: int
    page: int
    per_page: int
    last_page: int

class PaginatedResponse(BaseModel, Generic[T]):
    data: List[T]
    meta: PaginationMeta

# ===============================
# PATIENT SCHEMAS
# ===============================

class PatientBase(BaseModel):
    nama_lengkap: str = Field(..., max_length=100)
    jenis_kelamin: str = Field(..., description="Laki-Laki atau Perempuan")
    tanggal_lahir: date
    alamat: Optional[str] = None
    no_telepon: Optional[str] = None
    patient_date: Optional[datetime] = None
    date: Optional[datetime] = None

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

# ===============================
# REPORT SCHEMAS
# ===============================

class AnalysisCountDetail(BaseModel):
    positif: int
    negatif: int

class AnalysisReportResponse(BaseModel):
    created_at: datetime
    date: date
    nama_pasien: str
    kode_sample: str
    detail_jumlah: AnalysisCountDetail
    status_validasi: str

# ===============================
# VALIDATION SCHEMAS
# ===============================

class ValidationUpdate(BaseModel):
    validation_gram: str
    validation_bentuk: Optional[str] = None
    catatan_dokter: Optional[str] = None
    reannotated_by_user_id: Optional[int] = None

class ValidationResponse(BaseModel):
    id: int
    validation_gram: Optional[str]
    validation_bentuk: Optional[str]
    catatan_dokter: Optional[str]
    reannotated_at: Optional[datetime]
    message: str

# ===============================
# DOCTOR VALIDATION SCHEMAS
# ===============================

class PatientDetail(BaseModel):
    id_pasien: str
    nama_lengkap: str
    tanggal_lahir: date
    umur: int
    jenis_kelamin: str

class ValidationTask(BaseModel):
    id: int  # classification_id
    patient: PatientDetail
    image_url: str
    classification_gram: str
    classification_bentuk: Optional[str]
    confidence_score: float
    kode_sample: str


