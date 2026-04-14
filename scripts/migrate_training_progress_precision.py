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
                ALTER TABLE model_training_status
                ALTER COLUMN progress TYPE NUMERIC(5,2)
                """
            )
        )
        print("Updated model_training_status.progress to NUMERIC(5,2)")


if __name__ == "__main__":
    main()
