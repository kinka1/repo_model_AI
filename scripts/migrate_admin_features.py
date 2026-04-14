import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


def ensure_inference_time_column(conn):
    column_exists = conn.execute(
        text(
            """
            SELECT 1 FROM information_schema.columns
            WHERE table_name='ai_models' AND column_name='inference_time_s'
            """
        )
    ).fetchone()

    if column_exists:
        print("[OK] Column ai_models.inference_time_s already exists")
        return

    print("[MIGRATION] Adding ai_models.inference_time_s column ...")
    conn.execute(text("ALTER TABLE ai_models ADD COLUMN inference_time_s DOUBLE PRECISION"))
    print("[DONE] Column added")


def ensure_retrain_config_table(conn):
    print("[CHECK] Ensuring model_retrain_config table exists ...")
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS model_retrain_config (
                id SERIAL PRIMARY KEY,
                auto_retrain_enabled BOOLEAN NOT NULL DEFAULT FALSE,
                trigger_count INTEGER NOT NULL DEFAULT 500,
                validated_data_since_last_train INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
    )

    count = conn.execute(text("SELECT COUNT(*) FROM model_retrain_config")).scalar()
    if count == 0:
        print("[INIT] Inserting default retrain configuration row ...")
        conn.execute(
            text(
                "INSERT INTO model_retrain_config (auto_retrain_enabled, trigger_count, validated_data_since_last_train) VALUES (FALSE, 500, 0)"
            )
        )
    else:
        print("[OK] Existing retrain configuration rows found")


def main():
    load_dotenv()
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:admin@localhost:5432/TA KITA")
    engine = create_engine(database_url)

    with engine.begin() as conn:
        ensure_inference_time_column(conn)
        ensure_retrain_config_table(conn)


if __name__ == "__main__":
    main()
