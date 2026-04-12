from sqlalchemy import text
from app.database import engine

with engine.connect() as conn:
    try:
        # Specimen migration
        conn.execute(text("ALTER TABLE specimens ADD COLUMN IF NOT EXISTS total_detected INTEGER DEFAULT 0;"))

        # Patient migration for patient_date
        conn.execute(text("""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_name = 'patients' AND column_name = 'waktu_masuk'
                ) AND NOT EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_name = 'patients' AND column_name = 'patient_date'
                ) THEN
                    EXECUTE 'ALTER TABLE patients RENAME COLUMN waktu_masuk TO patient_date';
                END IF;
            END $$;
        """))
        conn.execute(text("ALTER INDEX IF EXISTS idx_patients_waktu_masuk RENAME TO idx_patients_patient_date;"))
        conn.execute(text("ALTER TABLE patients ADD COLUMN IF NOT EXISTS patient_date TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP;"))
        conn.execute(text("UPDATE patients SET patient_date = COALESCE(patient_date, created_at, CURRENT_TIMESTAMP);"))
        conn.execute(text("ALTER TABLE patients ALTER COLUMN patient_date SET NOT NULL;"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_patients_patient_date ON patients (patient_date);"))

        conn.commit()
        print("Success: migrations applied (specimens.total_detected, patients.patient_date)")
    except Exception as e:
        print("Info:", str(e))
