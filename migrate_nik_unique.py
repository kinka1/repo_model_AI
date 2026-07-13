"""
Migration: Add UNIQUE constraint to patients.nik column.

Run this once to clean up any duplicate NIKs and apply the constraint:
    python migrate_nik_unique.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.database import engine, SessionLocal
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError, OperationalError


def migrate():
    print("[MIGRATE] Checking patients.nik for duplicates ...")

    db = SessionLocal()
    try:
        # 1. Find duplicate NIKs
        dups = db.execute(
            text("""
                SELECT nik, COUNT(*) AS cnt, array_agg(id) AS ids
                FROM patients
                WHERE nik IS NOT NULL AND nik != ''
                GROUP BY nik
                HAVING COUNT(*) > 1
            """)
        ).fetchall()

        if dups:
            print(f"[MIGRATE] Found {len(dups)} NIK(s) with duplicates:")
            for row in dups:
                ids = list(row.ids)
                keep_id = ids[0]
                remove_ids = ids[1:]
                print(f"  NIK={row.nik} count={row.cnt} — keeping id={keep_id}, merging/removing ids={remove_ids}")

                # Re-link specimens & classifications to the kept patient
                for rid in remove_ids:
                    db.execute(
                        text("UPDATE specimens SET patient_id = :keep WHERE patient_id = :remove"),
                        {"keep": keep_id, "remove": rid},
                    )
                    db.execute(
                        text("UPDATE classifications SET patient_id = :keep WHERE patient_id = :remove"),
                        {"keep": keep_id, "remove": rid},
                    )
                    db.execute(
                        text("DELETE FROM patients WHERE id = :remove"),
                        {"remove": rid},
                    )
                    print(f"    Merged patient id={rid} into id={keep_id} and deleted.")
            db.commit()
            print("[MIGRATE] Duplicates resolved.")
        else:
            print("[MIGRATE] No duplicate NIKs found.")

        # 2. Add unique constraint (PostgreSQL)
        print("[MIGRATE] Adding UNIQUE constraint on patients.nik ...")
        try:
            db.execute(
                text("""
                    ALTER TABLE patients
                    ADD CONSTRAINT uq_patients_nik UNIQUE (nik)
                """)
            )
            db.commit()
            print("[MIGRATE] UNIQUE constraint added successfully.")
        except (ProgrammingError, OperationalError) as e:
            db.rollback()
            err_str = str(e)
            if "already exists" in err_str.lower():
                print("[MIGRATE] Constraint already exists — skipping.")
            else:
                print(f"[MIGRATE] Could not add constraint (may already exist): {e}")

    finally:
        db.close()

    print("[MIGRATE] Done.")


if __name__ == "__main__":
    migrate()
