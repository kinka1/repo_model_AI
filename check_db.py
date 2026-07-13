from app.database import SessionLocal
from app.models import Classification, Specimen
import os

db = SessionLocal()
try:
    print("--- Specimens ---")
    specimens = db.query(Specimen).order_by(Specimen.id.desc()).limit(5).all()
    for s in specimens:
        print(f"ID: {s.id}, file_path: {s.file_path}, exists: {os.path.exists(s.file_path) if s.file_path else False}")
    
    print("\n--- Classifications ---")
    crops = db.query(Classification).order_by(Classification.id.desc()).limit(10).all()
    for c in crops:
        print(f"ID: {c.id}, Specimen: {c.specimen_id}, image_path: {c.image_path}, exists: {os.path.exists(c.image_path) if c.image_path else False}")
finally:
    db.close()
