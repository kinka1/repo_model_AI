"""
Migration: Widen specimens.accession_number from VARCHAR(50) to VARCHAR(255).

Run:
    python migrate_accession_length.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.database import SessionLocal
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError, OperationalError


def migrate():
    print("[MIGRATE] Widening specimens.accession_number from VARCHAR(50) to VARCHAR(255) ...")

    db = SessionLocal()
    try:
        # PostgreSQL
        db.execute(
            text("""
                ALTER TABLE specimens
                ALTER COLUMN accession_number TYPE VARCHAR(255)
            """)
        )
        db.commit()
        print("[MIGRATE] Column altered successfully.")
    except (ProgrammingError, OperationalError) as e:
        db.rollback()
        print(f"[MIGRATE] Could not alter column: {e}")
    finally:
        db.close()

    print("[MIGRATE] Done.")


if __name__ == "__main__":
    migrate()
