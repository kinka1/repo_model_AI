# 🗄️ ARSITEKTUR DATABASE - BACTERIA CLASSIFICATION SYSTEM

## 📊 Entity Relationship Diagram (ERD)

```
┌─────────────────────┐
│      users          │
├─────────────────────┤
│ PK  id              │
│     username        │──┐
│     email           │  │
│     hashed_password │  │
│     full_name       │  │
│     role            │  │ (Admin/Analis/Dokter)
│     is_active       │  │
│     created_at      │  │
│     updated_at      │  │
│     last_login      │  │
└─────────────────────┘  │
         ▲               │
         │               │
         │ created_by    │ classified_by
         │               │
         │               │
         │               │
┌────────┴────────────┐  │     ┌─────────────────────┐
│    patients         │  │     │   ai_models         │
├─────────────────────┤  │     ├─────────────────────┤
│ PK  id              │  │     │ PK  id              │
│ UK  id_pasien       │  │     │ UK  (name, version) │
│     nama_lengkap    │  │     │     model_name      │
│     jenis_kelamin   │  │     │     model_type      │
│     tanggal_lahir   │  │     │     version         │
│     alamat          │  │     │     model_file_path │
│     no_telepon      │  │     │     accuracy        │
│ FK  created_by_id   │──┘     │     precision_score │
│     created_at      │        │     recall_score    │
│     updated_at      │        │     f1_score        │
└─────────────────────┘        │     is_active       │◄──┐ UNIQUE constraint
         │                     │ FK  created_by_id   │   │ (hanya 1 aktif)
         │                     │     created_at      │   │
         │                     │     updated_at      │   │
         │                     └─────────────────────┘   │
         │                              ▲                │
         │                              │                │
         │                              │ model_id       │
         │                              │                │
         │                     ┌────────┴────────────┐   │
         │                     │  model_training_    │   │
         │                     │     status          │   │
         │                     ├─────────────────────┤   │
         │                     │ PK  id              │   │
         │                     │ FK  model_id        │───┘
         │                     │ FK  dataset_id      │
         │                     │     status          │ (TRAINING/COMPLETED)
         │                     │     progress        │ (0.0 - 1.0)
         │                     │     start_time      │
         │                     │     end_time        │
         │                     │     error_message   │
         │                     │     current_epoch   │
         │                     │     total_epochs    │
         │                     └─────────────────────┘
         │                              ▲
         │                              │
         │                     ┌────────┴────────────┐
         │                     │     datasets        │
         │                     ├─────────────────────┤
         │                     │ PK  id              │
         │                     │     dataset_name    │
         │                     │     dataset_path    │
         │                     │     total_images    │
         │                     │     gram_pos_count  │
         │                     │     gram_neg_count  │
         │                     │ FK  uploaded_by_id  │
         │                     │     created_at      │
         │                     └─────────────────────┘
         │                              │
         │                              │
         │ patient_id          dataset_id
         │                              │
         ▼                              │
┌─────────────────────────────────────────┐
│       classifications                   │
├─────────────────────────────────────────┤
│ PK  id                                  │
│ FK  patient_id          ────────────────┼──► patients.id (CASCADE)
│ FK  dataset_id          ────────────────┘
│ FK  classified_by_model_id ─────────────┼──► ai_models.id
│ FK  classified_by_user_id ──────────────┼──► users.id (Analis)
│ FK  reannotated_by_user_id ─────────────┼──► users.id (Dokter)
│                                         │
│ === FILE INFO ===                       │
│     image_file_name                     │
│     image_path                          │
│                                         │
│ === AI RESULTS ===                      │
│     classification_gram    (Positif/Negatif)
│     classification_bentuk  (Batang/Kokus)
│     confidence_score       (0.0 - 1.0)
│     classified_at                       │
│                                         │
│ === DOCTOR VALIDATION ===               │
│     validation_gram        (NULL until validated)
│     validation_bentuk      (NULL until validated)
│     catatan_dokter                      │
│     reannotated_at                      │
│                                         │
│     created_at                          │
│     updated_at                          │
└─────────────────────────────────────────┘
         │
         │ (Optional Audit)
         ▼
┌─────────────────────┐
│   audit_logs        │
├─────────────────────┤
│ PK  id              │
│ FK  user_id         │
│     action          │
│     table_name      │
│     record_id       │
│     old_value       │ (JSONB)
│     new_value       │ (JSONB)
│     ip_address      │
│     user_agent      │
│     created_at      │
└─────────────────────┘

┌─────────────────────┐
│   sessions          │  (Optional - untuk token management)
├─────────────────────┤
│ PK  id              │
│ FK  user_id         │
│ UK  token_hash      │
│     expires_at      │
│     is_revoked      │
│     created_at      │
└─────────────────────┘
```

## 🔑 Relasi Utama

### 1. **users → patients**
- 1 user (Admin/Analis) bisa create banyak patients
- **Relationship:** 1:N (One-to-Many)

### 2. **patients → classifications**
- 1 patient bisa punya banyak classifications (gambar bakteri)
- **Relationship:** 1:N (One-to-Many)
- **Cascade DELETE:** Jika patient dihapus, semua classifications ikut terhapus

### 3. **ai_models → classifications**
- 1 ai_model bisa classify banyak images
- **Relationship:** 1:N (One-to-Many)

### 4. **users (Analis) → classifications**
- 1 Analis bisa upload banyak images
- **Relationship:** 1:N (One-to-Many)

### 5. **users (Dokter) → classifications**
- 1 Dokter bisa validate banyak images
- **Relationship:** 1:N (One-to-Many)

### 6. **ai_models → model_training_status**
- 1 model bisa punya banyak training sessions
- **Relationship:** 1:N (One-to-Many)

### 7. **datasets → model_training_status**
- 1 dataset bisa dipakai untuk training banyak models
- **Relationship:** 1:N (One-to-Many)

---

## 📋 Table Details

### **users** (8 rows typical)
- **Purpose:** Multi-role authentication (Admin, Analis, Dokter)
- **Key Constraint:** username & email UNIQUE
- **Security:** bcrypt hashed passwords

### **patients** (100-1000+ rows)
- **Purpose:** Data pasien yang diperiksa
- **Key Constraint:** id_pasien UNIQUE (P001, P002, etc)
- **Business Rule:** Bisa di-create oleh Admin atau Analis

### **ai_models** (5-20 rows)
- **Purpose:** Versioning & management model AI
- **Key Constraint:** UNIQUE (model_name, version)
- **Special:** Hanya 1 model boleh is_active=TRUE (UNIQUE INDEX)
- **File:** Menyimpan path ke file .pth

### **datasets** (10-50 rows)
- **Purpose:** Dataset untuk training/retraining
- **Statistics:** Total images, class distribution
- **File:** Path ke dataset file

### **classifications** (10,000+ rows) - CORE TABLE
- **Purpose:** Hasil klasifikasi AI + Validasi Dokter
- **Workflow:**
  1. Analis upload → AI classify → save result
  2. Dokter validate → update validation_gram, validation_bentuk
- **Status:** validation_gram NULL = pending, NOT NULL = validated

### **model_training_status** (10-100 rows)
- **Purpose:** Track progress training model
- **Real-time:** progress field untuk progress bar
- **Status:** TRAINING → COMPLETED/FAILED

### **audit_logs** (1000+ rows)
- **Purpose:** Activity logging semua action
- **Security:** Track who did what when
- **JSONB:** Menyimpan old/new values untuk rollback

### **sessions** (100-1000 rows)
- **Purpose:** Token management & revocation
- **Security:** Bisa revoke token untuk logout

---

## 🔍 Important Indexes

```sql
-- Performance kritik untuk query
idx_users_username              -- Fast login
idx_patients_id_pasien          -- Search patient
idx_classifications_pending     -- Get pending validations (Dokter)
idx_classifications_patient     -- Get patient history
idx_ai_models_active            -- Get active model
idx_one_active_model            -- Constraint: hanya 1 aktif
```

---

## 📊 Database Size Estimation

| Table | Rows (1 year) | Size Estimate |
|-------|---------------|---------------|
| users | 50 | 10 KB |
| patients | 5,000 | 500 KB |
| classifications | 50,000 | 20 MB |
| ai_models | 20 | 5 KB |
| datasets | 50 | 10 KB |
| model_training_status | 100 | 20 KB |
| audit_logs | 100,000 | 50 MB |
| **TOTAL** | | **~70 MB** |

**Plus File Storage:**
- Images: 50,000 × 500 KB = **25 GB**
- Models: 20 × 300 MB = **6 GB**
- **Total Storage: ~31 GB**

---

## 🚀 Technology Stack

### Database
- **PostgreSQL 14+** (Recommended)
  - ✅ JSONB support untuk audit_logs
  - ✅ Powerful indexes
  - ✅ Triggers & Views
  - ✅ Concurrent writes

### Alternative
- **MySQL 8.0+**
  - ✅ Good performance
  - ⚠️ Limited JSON support
  - ⚠️ Less powerful indexes

### ORM
- **SQLAlchemy 2.0** (Python)
  - Async support
  - Migration with Alembic
  - Type hints

---

## 📝 Next Steps

1. **Install PostgreSQL**
   ```bash
   # Windows (with chocolatey)
   choco install postgresql
   
   # Or download: https://www.postgresql.org/download/
   ```

2. **Create Database**
   ```sql
   CREATE DATABASE bacteria_classification;
   ```

3. **Run Schema**
   ```bash
   psql -U postgres -d bacteria_classification -f database_schema.sql
   ```

4. **Install SQLAlchemy**
   ```bash
   pip install sqlalchemy psycopg2-binary alembic
   ```

5. **Configure Connection**
   ```python
   DATABASE_URL = "postgresql://postgres:password@localhost/bacteria_classification"
   ```

Apakah Anda ingin saya buatkan:
- **A)** SQLAlchemy models (ORM) untuk semua table
- **B)** Database connection & CRUD operations
- **C)** Migration scripts dengan Alembic

? 🎯
