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


class PatientUpdate(BaseModel):
    nama_lengkap: Optional[str] = Field(None, max_length=100)
    jenis_kelamin: Optional[str] = Field(None, description="Laki-Laki atau Perempuan")
    tanggal_lahir: Optional[date] = None
    alamat: Optional[str] = None
    no_telepon: Optional[str] = None
    patient_date: Optional[datetime] = None
    date: Optional[datetime] = None

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
    classification_gram: str
    classification_bentuk: Optional[str] = None
    confidence_score: float

class AnalysisSessionSubmit(BaseModel):
    patient_id: int
    specimen_id: Optional[int] = None
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
# USER MANAGEMENT SCHEMAS
# ===============================

class UserBaseSchema(BaseModel):
    full_name: str
    username: str
    email: Optional[str] = None
    role: str
    is_active: bool = True

class UserCreateRequest(UserBaseSchema):
    password: str = Field(..., min_length=6)

class UserUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    username: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None
    password: Optional[str] = Field(None, min_length=6)
    new_password: Optional[str] = Field(None, min_length=6)

class UserResponseSchema(BaseModel):
    id: int
    full_name: str
    username: str
    email: Optional[str] = None
    role: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class RoleListResponse(BaseModel):
    roles: List[str]

# ===============================
# MODEL MANAGEMENT SCHEMAS
# ===============================

class AIModelSummaryResponse(BaseModel):
    id: int
    model_name: str
    task_type: str
    version: str
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    inference_time_s: Optional[float] = None
    status: str
    is_active: bool
    is_recommended: bool
    delta_acc: Optional[float] = None
    delta_f1: Optional[float] = None
    delta_time: Optional[float] = None
    created_at: datetime
    updated_at: datetime

class ActiveModelResponse(BaseModel):
    task_type: str
    model: Optional[AIModelSummaryResponse] = None

class BestModelResponse(BaseModel):
    task_type: str
    model: Optional[AIModelSummaryResponse] = None

class RetrainConfigResponse(BaseModel):
    auto_retrain_enabled: bool
    trigger_count: int
    validated_data_since_last_train: int

class RetrainConfigUpdateRequest(BaseModel):
    auto_retrain_enabled: Optional[bool] = None
    trigger_count: Optional[int] = Field(None, ge=1)

class TrainingJobResponse(BaseModel):
    job_id: int
    model_id: Optional[int]
    model_name: Optional[str] = None
    status: str
    progress_percent: Optional[float] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    logs_summary: Optional[str] = None


class RetrainStartRequest(BaseModel):
    model_id: int
    epochs_head: Optional[int] = Field(10, ge=1, le=200)
    epochs_ft: Optional[int] = Field(30, ge=1, le=400)
    batch_size: Optional[int] = Field(32, ge=1, le=256)
    val_ratio_crops: Optional[float] = Field(0.2, gt=0.0, lt=1.0)


class RetrainStartResponse(BaseModel):
    job_id: int
    status: str
    message: str


class RetrainModelOptionResponse(BaseModel):
    id: int
    model_name: str
    version: str
    task_type: str
    is_active: bool
    supports_retrain: bool

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
    id: int
    patient: PatientDetail
    image_url: str
    classification_gram: str
    classification_bentuk: Optional[str]
    confidence_score: float
    kode_sample: str

# ===============================
# MEDICAL REPORT SCHEMAS
# ===============================

class ReportPatientData(BaseModel):
    id_pasien: str
    nama: str
    tanggal_lahir: date
    umur: int
    jenis_kelamin: str

class ReportClinicalData(BaseModel):
    tanggal_sampel: datetime
    jenis_spesimen: str = "Pewarnaan Gram"
    analis: Optional[str] = "N/A"
    dokter: Optional[str] = "N/A"

class ReportResultSummary(BaseModel):
    total_objek: int
    gram_positif_kokus: int
    gram_positif_batang: int
    gram_negatif_kokus: int
    gram_negatif_batang: int
    kesimpulan: Optional[str] = None
    catatan_dokter: Optional[str] = None

class ReportEvidenceImage(BaseModel):
    image_url: str
    label: str

class MedicalReportResponse(BaseModel):
    id_laporan: str = Field(..., description="ID unik untuk laporan ini, biasanya sama dengan specimen_id")
    tanggal_cetak: datetime
    specimen_id: int
    pasien: ReportPatientData
    data_klinis: ReportClinicalData
    ringkasan_hasil: ReportResultSummary
    gambar_bukti: List[ReportEvidenceImage] = []


# ===============================
# AUTH SCHEMAS
# ===============================


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthUserResponse(BaseModel):
    id: int
    full_name: str
    username: str
    email: Optional[str] = None
    role: str
    is_active: bool


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    user: AuthUserResponse


class AuthMessageResponse(BaseModel):
    message: str


class ForgotPasswordRequest(BaseModel):
    username_or_email: str


class ForgotPasswordResponse(BaseModel):
    message: str
    reset_token: Optional[str] = None
    expires_at: Optional[datetime] = None


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(..., min_length=6)


class ChangePasswordRequest(BaseModel):
    new_password: str = Field(..., min_length=6)
