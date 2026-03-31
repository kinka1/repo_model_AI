# 🚀 QUICK START GUIDE - Database TA KITA

## 📋 Prerequisites

- ✅ Python 3.8+
- ✅ PostgreSQL 14+ **OR** Docker
- ✅ Git

---

## ⚡ Option A: Quick Start with Docker (RECOMMENDED)

### 1. Start PostgreSQL Container
```powershell
# Start database
docker-compose up -d postgres

# Wait for database to be ready (10-15 seconds)
docker-compose logs -f postgres
```

### 2. Install Python Dependencies
```powershell
pip install -r requirements_db.txt
```

### 3. Create .env File
```powershell
# Copy example and edit
Copy-Item .env.example .env

# Edit .env with your settings (default values work for Docker)
```

### 4. Test Database Connection
```powershell
python test_db.py
```

### 5. Start FastAPI Server
```powershell
# Existing command
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Access Services
- **API Swagger UI**: http://localhost:8000/docs
- **pgAdmin**: http://localhost:5050
  - Email: `admin@takita.com`
  - Password: `admin`

---

## 📦 Option B: Manual PostgreSQL Installation

### 1. Install PostgreSQL

#### Windows (Chocolatey)
```powershell
choco install postgresql14 -y
```

#### Windows (Installer)
Download from: https://www.postgresql.org/download/windows/

### 2. Create Database
```powershell
# Connect to PostgreSQL
psql -U postgres

# In psql console:
CREATE DATABASE "TA KITA";
\q
```

### 3. Run Database Schema
```powershell
psql -U postgres -d "TA KITA" -f database_schema.sql
```

### 4. Verify Tables
```powershell
psql -U postgres -d "TA KITA" -c "\dt"
```

### 5. Install Python Dependencies
```powershell
pip install -r requirements_db.txt
```

### 6. Configure Environment
```powershell
Copy-Item .env.example .env

# Edit DATABASE_URL in .env if needed:
# DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/TA KITA
```

### 7. Test Connection
```powershell
python test_db.py
```

---

## 🔧 Migrate Existing API to PostgreSQL

### Step 1: Update FastAPI Dependencies

In your `app/main.py`, add database dependency:

```python
from app.db_connection import get_db
from sqlalchemy.orm import Session
from fastapi import Depends

# Add to endpoints:
@app.get("/users")
def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users
```

### Step 2: Replace In-Memory Database

**Before (app/auth.py):**
```python
from app.database import db

user = db.get_user_by_username(username)
```

**After:**
```python
from app.db_models import User

def authenticate_user(username: str, password: str, db: Session):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user
```

### Step 3: Update Dependencies

Replace `get_current_user` in `app/auth.py`:

```python
from sqlalchemy.orm import Session

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # Query from database
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    
    return user
```

---

## 🧪 Testing Checklist

After setup, verify:

- [ ] `python test_db.py` passes all tests
- [ ] Login endpoint works: `POST /auth/login`
- [ ] 3 users exist: admin, analis1, dokter1
- [ ] 1 active AI model exists
- [ ] Database views return data

---

## 📊 Database Management

### View All Users
```powershell
psql -U postgres -d "TA KITA" -c "SELECT username, role, email FROM users;"
```

### Check Active Model
```powershell
psql -U postgres -d "TA KITA" -c "SELECT model_name, version, accuracy FROM ai_models WHERE is_active = true;"
```

### View Statistics
```powershell
psql -U postgres -d "TA KITA" -c "SELECT * FROM analyst_dashboard_stats;"
```

### Backup Database
```powershell
pg_dump -U postgres -d "TA KITA" > backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').sql
```

---

## 🐛 Troubleshooting

### Error: "database 'TA KITA' does not exist"
```powershell
psql -U postgres -c 'CREATE DATABASE "TA KITA";'
```

### Error: "password authentication failed"
Edit `.env` with correct password:
```
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/TA KITA
```

### Error: "module 'psycopg2' not found"
```powershell
pip install psycopg2-binary
```

### Error: Docker container won't start
```powershell
# Check logs
docker-compose logs postgres

# Restart
docker-compose down
docker-compose up -d
```

### Error: Port 5432 already in use
Stop existing PostgreSQL service or change port in `docker-compose.yml`:
```yaml
ports:
  - "5433:5432"  # Use 5433 on host
```

---

## 📚 Next Steps

1. ✅ **Database Running** - PostgreSQL with TA KITA database
2. ✅ **Tables Created** - 8 tables with seed data
3. ✅ **Test Passed** - Connection verified
4. ⏳ **Migrate Endpoints** - Replace in-memory database with PostgreSQL
5. ⏳ **Implement CRUD** - User management, patient management
6. ⏳ **Add File Upload** - Image storage for classifications
7. ⏳ **Deploy** - Production deployment guide

---

## 🆘 Need Help?

1. Check `DATABASE_SETUP_GUIDE.md` for detailed documentation
2. Review `DATABASE_ARCHITECTURE.md` for schema design
3. Run `python test_db.py` for diagnostics
4. Check Docker logs: `docker-compose logs -f`

---

## ✅ Success Indicators

You're ready to proceed if:

```
✅ docker-compose ps shows 'ta-kita-db' as 'Up'
✅ python test_db.py shows "ALL TESTS PASSED"
✅ Can login to pgAdmin at localhost:5050
✅ FastAPI server starts without errors
```

Good luck! 🚀
