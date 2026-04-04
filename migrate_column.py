import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:admin@localhost:5432/TA KITA")

def migrate():
    engine = create_engine(DATABASE_URL)
    with engine.connect() as connection:
        print("Adding specimen_id column to classifications table...")
        try:
            connection.execute(text("ALTER TABLE classifications ADD COLUMN specimen_id INTEGER REFERENCES specimens(id) ON DELETE CASCADE;"))
            connection.commit()
            print("Successfully added specimen_id column.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    migrate()
