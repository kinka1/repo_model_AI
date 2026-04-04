import os
import re
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:admin@localhost:5432/TA KITA")

def fix_links():
    engine = create_engine(DATABASE_URL)
    with engine.connect() as connection:
        print("Fetching existing classifications to fix specimen links...")
        result = connection.execute(text("SELECT id, image_file_name FROM classifications WHERE specimen_id IS NULL;"))
        rows = result.fetchall()
        
        updated_count = 0
        for row_id, filename in rows:
            # Format: crop_{specimen_id}_{idx}_{uuid}.jpg
            match = re.match(r"crop_(\d+)_", filename)
            if match:
                specimen_id = int(match.group(1))
                connection.execute(
                    text("UPDATE classifications SET specimen_id = :sid WHERE id = :cid"),
                    {"sid": specimen_id, "cid": row_id}
                )
                updated_count += 1
        
        connection.commit()
        print(f"Successfully linked {updated_count} classification records to their specimens.")

if __name__ == "__main__":
    fix_links()
