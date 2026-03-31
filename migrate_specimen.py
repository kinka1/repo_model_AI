from sqlalchemy import text
from app.database import engine

with engine.connect() as conn:
    try:
        conn.execute(text("ALTER TABLE specimens ADD COLUMN total_detected INTEGER DEFAULT 0;"))
        conn.commit()
        print("Success: added total_detected to specimens")
    except Exception as e:
        print("Info:", str(e))
