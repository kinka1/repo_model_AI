from sqlalchemy import Boolean, Column, Integer, String, Float, DateTime, ForeignKey, Date, Numeric, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base
from .utils import get_local_now

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=False)
    role = Column(String(20), nullable=False) # 'Admin', 'Analis', 'Dokter'
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=get_local_now)
    updated_at = Column(DateTime, default=get_local_now, onupdate=get_local_now)
    last_login = Column(DateTime, nullable=True)

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash = Column(String(255), unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    is_revoked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=get_local_now)

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False, index=True)
    table_name = Column(String(50), nullable=True)
    record_id = Column(Integer, nullable=True)
    old_value = Column(JSON, nullable=True)
    new_value = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    created_at = Column(DateTime, default=get_local_now, index=True)

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    dataset_name = Column(String(100), nullable=False)
    dataset_file_path = Column(String(255), nullable=True)
    total_images = Column(Integer, default=0)
    gram_positive_count = Column(Integer, default=0)
    gram_negative_count = Column(Integer, default=0)
    uploaded_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=get_local_now)

class AIModel(Base):
    __tablename__ = "ai_models"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)
    version = Column(String(20), nullable=False)
    model_file_path = Column(String(255), nullable=True)
    accuracy = Column(Numeric(5, 4), nullable=True)
    precision_score = Column(Numeric(5, 4), nullable=True)
    recall_score = Column(Numeric(5, 4), nullable=True)
    f1_score = Column(Numeric(5, 4), nullable=True)
    inference_time_s = Column(Float, nullable=True)
    is_active = Column(Boolean, default=False, index=True)
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=get_local_now)
    updated_at = Column(DateTime, default=get_local_now, onupdate=get_local_now)


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_hash = Column(String(255), unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    is_used = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=get_local_now)
    updated_at = Column(DateTime, default=get_local_now, onupdate=get_local_now)

class ModelTrainingStatus(Base):
    __tablename__ = "model_training_status"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("ai_models.id", ondelete="CASCADE"), nullable=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    status = Column(String(20), nullable=False, index=True) # 'TRAINING', 'COMPLETED', 'FAILED', 'IDLE'
    # Progress disimpan sebagai persentase 0.00 - 100.00
    progress = Column(Numeric(5, 2), default=0.0)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    current_epoch = Column(Integer, nullable=True)
    total_epochs = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=get_local_now)


class ModelRetrainConfig(Base):
    __tablename__ = "model_retrain_config"

    id = Column(Integer, primary_key=True, index=True)
    auto_retrain_enabled = Column(Boolean, default=False, nullable=False)
    trigger_count = Column(Integer, default=500, nullable=False)
    validated_data_since_last_train = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=get_local_now)
    updated_at = Column(DateTime, default=get_local_now, onupdate=get_local_now)

class InternalMessage(Base):
    """Thread pesan internal antara Analis dan Dokter untuk keperluan revisi."""
    __tablename__ = "internal_messages"

    id = Column(Integer, primary_key=True, index=True)
    specimen_id = Column(Integer, ForeignKey("specimens.id", ondelete="CASCADE"), nullable=False, index=True)
    sender_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    message_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=get_local_now)

    # Relationships
    sender = relationship("User", lazy="joined")
    specimen = relationship("Specimen")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    id_pasien = Column(String(20), unique=True, index=True, nullable=False)
    nik = Column(String(32), unique=True, index=True, nullable=True)
    nama_lengkap = Column(String(100), index=True, nullable=False)
    jenis_kelamin = Column(String(20), nullable=False) # 'Laki-Laki', 'Perempuan'
    tanggal_lahir = Column(Date, nullable=False)
    alamat = Column(Text, nullable=True)
    no_telepon = Column(String(20), nullable=True)
    satusehat_id = Column(String(100), nullable=True)
    patient_date = Column(DateTime, default=get_local_now, nullable=False, index=True)
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    created_at = Column(DateTime, default=get_local_now, index=True)
    updated_at = Column(DateTime, default=get_local_now, onupdate=get_local_now)

    # Relationships
    classifications = relationship("Classification", back_populates="patient", cascade="all, delete-orphan")
    specimens = relationship("Specimen", back_populates="patient", cascade="all, delete-orphan")

    @property
    def date(self):
        return self.patient_date

class Specimen(Base):
    """Tabel tambahan untuk menyimpan metadata gambar spesimen asli (utuh)."""
    __tablename__ = "specimens"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    accession_number = Column(String(255), unique=False, index=True, nullable=True)
    specimen_type = Column(String(100), nullable=True)
    doctor_sender = Column(String(100), nullable=True)
    clinical_diagnosis = Column(Text, nullable=True)
    collected_at = Column(DateTime, nullable=True)
    received_at = Column(DateTime, nullable=True)

    microscope_type = Column(String(100), nullable=True)
    magnification = Column(String(50), nullable=True)
    image_resolution = Column(String(50), nullable=True)
    analyst_note = Column(Text, nullable=True)

    file_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    status = Column(String(20), default="pending")
    validation_status = Column(String(20), default="pending")
    validated_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    validated_at = Column(DateTime, nullable=True)
    total_detected = Column(Integer, default=0, nullable=True)
    uploaded_at = Column(DateTime, default=get_local_now)
    
    # Relationship
    patient = relationship("Patient", back_populates="specimens")
    classifications = relationship("Classification", back_populates="specimen", cascade="all, delete-orphan")

class Classification(Base):
    __tablename__ = "classifications"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    specimen_id = Column(Integer, ForeignKey("specimens.id", ondelete="CASCADE"), nullable=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    image_file_name = Column(String(255), nullable=False)
    image_path = Column(String(500), nullable=False)
    roi_bbox = Column(JSON, nullable=True)
    roi_source = Column(String(20), nullable=True)
    classified_by_model_id = Column(Integer, ForeignKey("ai_models.id"), nullable=True, index=True)
    
    classification_gram = Column(String(20), nullable=True)
    classification_bentuk = Column(String(20), nullable=True)
    confidence_score = Column(Numeric(5, 4), nullable=True)
    
    validation_gram = Column(String(20), nullable=True)
    validation_bentuk = Column(String(20), nullable=True)
    
    catatan_dokter = Column(Text, nullable=True)
    
    reannotated_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    reannotated_at = Column(DateTime, nullable=True)
    
    classified_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    classified_at = Column(DateTime, default=get_local_now, index=True)
    
    created_at = Column(DateTime, default=get_local_now)
    updated_at = Column(DateTime, default=get_local_now, onupdate=get_local_now)

    # Relationships
    patient = relationship("Patient", back_populates="classifications")
    specimen = relationship("Specimen", back_populates="classifications")
