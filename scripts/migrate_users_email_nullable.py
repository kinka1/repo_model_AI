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
                ALTER TABLE users
                ALTER COLUMN email DROP NOT NULL
                """
            )
        )
        print("users.email is now nullable")


if __name__ == "__main__":
    main()
