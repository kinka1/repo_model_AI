import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def migrate():
    engine = create_engine(DATABASE_URL)
    with engine.connect() as connection:
        print("Adding inference_time_s column to ai_models table...")
        try:
            connection.execute(text("ALTER TABLE ai_models ADD COLUMN inference_time_s FLOAT;"))
            connection.commit()
            print("Successfully added inference_time_s column.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    migrate()
