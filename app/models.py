from sqlalchemy import Boolean, Column, Integer, String, Float, DateTime, ForeignKey, Date, Numeric, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=False)
    role = Column(String(20), nullable=False) # 'Admin', 'Analis', 'Dokter'
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash = Column(String(255), unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    is_revoked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

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
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    dataset_name = Column(String(100), nullable=False)
    dataset_file_path = Column(String(255), nullable=True)
    total_images = Column(Integer, default=0)
    gram_positive_count = Column(Integer, default=0)
    gram_negative_count = Column(Integer, default=0)
    uploaded_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

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
    is_active = Column(Boolean, default=False, index=True)
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ModelTrainingStatus(Base):
    __tablename__ = "model_training_status"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("ai_models.id", ondelete="CASCADE"), nullable=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    status = Column(String(20), nullable=False, index=True) # 'TRAINING', 'COMPLETED', 'FAILED', 'IDLE'
    progress = Column(Numeric(3, 2), default=0.0)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    current_epoch = Column(Integer, nullable=True)
    total_epochs = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    id_pasien = Column(String(20), unique=True, index=True, nullable=False)
    nama_lengkap = Column(String(100), index=True, nullable=False)
    jenis_kelamin = Column(String(20), nullable=False) # 'Laki-Laki', 'Perempuan'
    tanggal_lahir = Column(Date, nullable=False)
    alamat = Column(Text, nullable=True)
    no_telepon = Column(String(20), nullable=True)
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    classifications = relationship("Classification", back_populates="patient", cascade="all, delete-orphan")
    specimens = relationship("Specimen", back_populates="patient", cascade="all, delete-orphan")

class Specimen(Base):
    """Tabel tambahan untuk menyimpan metadata gambar spesimen asli (utuh)."""
    __tablename__ = "specimens"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    total_detected = Column(Integer, default=0, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    patient = relationship("Patient", back_populates="specimens")

class Classification(Base):
    __tablename__ = "classifications"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    image_file_name = Column(String(255), nullable=False)
    image_path = Column(String(500), nullable=False)
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
    classified_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship back to Patient
    patient = relationship("Patient", back_populates="classifications")
