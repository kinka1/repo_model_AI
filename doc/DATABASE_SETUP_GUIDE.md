# 🗄️ DATABASE SETUP GUIDE - PostgreSQL

## 📦 Instalasi PostgreSQL

### Windows (Recommended Method)

#### **Option 1: Official Installer**
```powershell
# Download installer dari:
# https://www.postgresql.org/download/windows/

# Atau gunakan Chocolatey:
choco install postgresql14 -y

# Atau gunakan Scoop:
scoop install postgresql
```

#### **Option 2: Docker (Recommended untuk Development)**
```powershell
# Pull PostgreSQL image
docker pull postgres:14-alpine

# Run PostgreSQL container
docker run --name bacteria-db `
  -e POSTGRES_PASSWORD=your_password `
  -e POSTGRES_USER=postgres `
  -e POSTGRES_DB=bacteria_classification `
  -p 5432:5432 `
  -v postgres-data:/var/lib/postgresql/data `
  -d postgres:14-alpine

# Verify running
docker ps
```

---

## ⚙️ Konfigurasi Database

### 1. Create Database
```sql
-- Connect ke PostgreSQL
psql -U postgres

-- Create database
CREATE DATABASE bacteria_classification;

-- Connect ke database
\c bacteria_classification

-- Verify
\dt
```

### 2. Run Schema
```powershell
# Dari root project directory
psql -U postgres -d bacteria_classification -f database_schema.sql
```

### 3. Verify Tables
```sql
-- List all tables
\dt

-- Check users table
SELECT * FROM users;

-- Check ai_models
SELECT * FROM ai_models;
```

---

## 🐍 Python SQLAlchemy Setup

### 1. Install Dependencies
```powershell
pip install sqlalchemy psycopg2-binary alembic asyncpg
```

### 2. Create Database Connection

Saya buatkan file `app/db_connection.py`:

```python
"""
Database connection using SQLAlchemy
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
import os
from dotenv import load_dotenv

load_dotenv()

# Database URL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/bacteria_classification"
)

# Create engine
engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,  # For development
    echo=True,  # Log SQL queries (disable in production)
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# Dependency for FastAPI
def get_db():
    """
    Database session dependency for FastAPI
    
    Usage:
        @app.get("/users")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### 3. Create SQLAlchemy Models

File `app/db_models.py`:

```python
"""
SQLAlchemy ORM Models
"""
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, 
    ForeignKey, Text, Enum as SQLEnum, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
from app.db_connection import Base
from app.database import UserRole, Gender, GramClassification, ShapeClassification, ModelStatus
import enum


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    role = Column(SQLEnum(UserRole), nullable=False, default=UserRole.ANALIS)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    created_patients = relationship("Patient", back_populates="created_by")
    classifications = relationship("Classification", foreign_keys="Classification.classified_by_user_id")
    validations = relationship("Classification", foreign_keys="Classification.reannotated_by_user_id")


class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    id_pasien = Column(String(50), unique=True, nullable=False, index=True)
    nama_lengkap = Column(String(100), nullable=False)
    jenis_kelamin = Column(SQLEnum(Gender), nullable=False)
    tanggal_lahir = Column(DateTime, nullable=False)
    alamat = Column(Text)
    no_telepon = Column(String(20))
    created_by_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    created_by = relationship("User", back_populates="created_patients")
    classifications = relationship("Classification", back_populates="patient", cascade="all, delete-orphan")


class AIModel(Base):
    __tablename__ = "ai_models"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)
    version = Column(String(20), nullable=False)
    model_file_path = Column(String(500), nullable=False)
    accuracy = Column(Float)
    precision_score = Column(Float)
    recall_score = Column(Float)
    f1_score = Column(Float)
    is_active = Column(Boolean, default=False)
    created_by_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('model_name', 'version', name='uq_model_name_version'),
        Index('idx_one_active_model', 'is_active', unique=True, postgresql_where=(is_active == True)),
    )
    
    # Relationships
    classifications = relationship("Classification", back_populates="classified_by_model")
    training_sessions = relationship("ModelTrainingStatus", back_populates="model")


class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_name = Column(String(100), nullable=False, unique=True)
    dataset_file_path = Column(String(500), nullable=False)
    total_images = Column(Integer, default=0)
    gram_positive_count = Column(Integer, default=0)
    gram_negative_count = Column(Integer, default=0)
    uploaded_by_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    classifications = relationship("Classification", back_populates="dataset")
    training_sessions = relationship("ModelTrainingStatus", back_populates="dataset")


class ModelTrainingStatus(Base):
    __tablename__ = "model_training_status"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("ai_models.id"), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    status = Column(SQLEnum(ModelStatus), default=ModelStatus.IDLE)
    progress = Column(Float, default=0.0)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    error_message = Column(Text)
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, default=0)
    
    # Relationships
    model = relationship("AIModel", back_populates="training_sessions")
    dataset = relationship("Dataset", back_populates="training_sessions")


class Classification(Base):
    __tablename__ = "classifications"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    classified_by_model_id = Column(Integer, ForeignKey("ai_models.id"), nullable=False)
    classified_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    reannotated_by_user_id = Column(Integer, ForeignKey("users.id"))
    
    # File info
    image_file_name = Column(String(255), nullable=False)
    image_path = Column(String(500), nullable=False)
    
    # AI Results
    classification_gram = Column(SQLEnum(GramClassification), nullable=False)
    classification_bentuk = Column(SQLEnum(ShapeClassification), nullable=False)
    confidence_score = Column(Float, nullable=False)
    classified_at = Column(DateTime, default=datetime.utcnow)
    
    # Doctor Validation
    validation_gram = Column(SQLEnum(GramClassification))
    validation_bentuk = Column(SQLEnum(ShapeClassification))
    catatan_dokter = Column(Text)
    reannotated_at = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_classifications_pending', 'patient_id', postgresql_where=(validation_gram == None)),
    )
    
    # Relationships
    patient = relationship("Patient", back_populates="classifications")
    dataset = relationship("Dataset", back_populates="classifications")
    classified_by_model = relationship("AIModel", back_populates="classifications")


class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String(50), nullable=False)
    table_name = Column(String(50), nullable=False)
    record_id = Column(Integer, nullable=False)
    old_value = Column(JSONB)
    new_value = Column(JSONB)
    ip_address = Column(String(50))
    user_agent = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token_hash = Column(String(255), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    is_revoked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
```

---

## 🔧 Environment Variables

Create `.env` file:

```env
# Database
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/bacteria_classification

# JWT
SECRET_KEY=your-super-secret-key-change-this-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# API
API_HOST=0.0.0.0
API_PORT=8000

# File Storage
UPLOAD_DIR=./uploads/images
MODEL_DIR=./models
```

---

## 🔄 Database Migration dengan Alembic

### 1. Initialize Alembic
```powershell
alembic init alembic
```

### 2. Configure Alembic

Edit `alembic/env.py`:

```python
from app.db_connection import Base
from app.db_models import User, Patient, AIModel, Dataset, Classification, ModelTrainingStatus, AuditLog, Session

target_metadata = Base.metadata

# Add your database URL
from app.db_connection import DATABASE_URL
config.set_main_option("sqlalchemy.url", DATABASE_URL)
```

### 3. Create Migration
```powershell
# Auto-generate migration from models
alembic revision --autogenerate -m "Initial schema"

# Apply migration
alembic upgrade head
```

### 4. Future Migrations
```powershell
# When you modify models
alembic revision --autogenerate -m "Add new column"
alembic upgrade head

# Rollback
alembic downgrade -1
```

---

## 🧪 Testing Connection

Create `test_db.py`:

```python
from app.db_connection import engine, SessionLocal
from app.db_models import User, AIModel
from sqlalchemy import text

def test_connection():
    # Test raw connection
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print(f"✅ Database connected: {result.fetchone()}")
    
    # Test ORM
    db = SessionLocal()
    try:
        users = db.query(User).all()
        print(f"✅ Found {len(users)} users")
        
        for user in users:
            print(f"  - {user.username} ({user.role})")
        
        active_model = db.query(AIModel).filter(AIModel.is_active == True).first()
        if active_model:
            print(f"✅ Active model: {active_model.model_name} v{active_model.version}")
    finally:
        db.close()

if __name__ == "__main__":
    test_connection()
```

Run test:
```powershell
python test_db.py
```

---

## 📊 Monitoring & Performance

### Check Database Size
```sql
SELECT 
    pg_size_pretty(pg_database_size('bacteria_classification')) as db_size;
```

### Check Table Sizes
```sql
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### Check Index Usage
```sql
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as scans,
    pg_size_pretty(pg_relation_size(indexrelid)) as size
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

### Optimize Performance
```sql
-- Analyze tables for query planner
ANALYZE;

-- Vacuum to reclaim space
VACUUM ANALYZE;

-- Reindex if needed
REINDEX DATABASE bacteria_classification;
```

---

## 🚀 Production Checklist

- [ ] Change SECRET_KEY in .env
- [ ] Set proper PostgreSQL password
- [ ] Enable SSL connection
- [ ] Configure connection pooling (use `pool_size=20`)
- [ ] Disable SQLAlchemy echo (set to False)
- [ ] Setup database backups (pg_dump)
- [ ] Configure log rotation
- [ ] Monitor slow queries (pg_stat_statements)
- [ ] Setup read replicas if needed
- [ ] Use connection pooler (PgBouncer)

---

## 🔄 Backup & Restore

### Backup
```powershell
# Full backup
pg_dump -U postgres -d bacteria_classification > backup_$(Get-Date -Format "yyyyMMdd_HHmmss").sql

# Backup with compression
pg_dump -U postgres -d bacteria_classification | gzip > backup.sql.gz
```

### Restore
```powershell
# From SQL file
psql -U postgres -d bacteria_classification < backup.sql

# From compressed
gunzip -c backup.sql.gz | psql -U postgres -d bacteria_classification
```

---

## 🐳 Docker Compose (Recommended)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:14-alpine
    container_name: bacteria-db
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: your_password
      POSTGRES_DB: bacteria_classification
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database_schema.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped
  
  pgadmin:
    image: dpage/pgadmin4
    container_name: pgadmin
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@bacteria.com
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "5050:80"
    depends_on:
      - postgres
    restart: unless-stopped

volumes:
  postgres_data:
```

Run:
```powershell
docker-compose up -d
```

Access pgAdmin: http://localhost:5050

---

## 📚 Next Steps

1. ✅ Install PostgreSQL (Docker recommended)
2. ✅ Run database_schema.sql
3. ✅ Install SQLAlchemy dependencies
4. ⏳ Create SQLAlchemy models (I'll generate for you)
5. ⏳ Migrate FastAPI endpoints to use SQLAlchemy
6. ⏳ Test CRUD operations
7. ⏳ Setup Alembic migrations

Butuh saya buatkan file-file di atas? 🚀
