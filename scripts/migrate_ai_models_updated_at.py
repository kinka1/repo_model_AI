import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


def main() -> None:
    load_dotenv()
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:admin@localhost:5432/TA KITA")
    engine = create_engine(database_url)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                ALTER TABLE ai_models
                ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
                """
            )
        )
        print("ai_models.updated_at is ready")


if __name__ == "__main__":
    main()
