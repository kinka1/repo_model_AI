import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:admin@localhost:5432/TA KITA")

def drop_constraints():
    engine = create_engine(DATABASE_URL)
    with engine.connect() as connection:
        print("Dropping strict check constraints from classifications table...")
        constraints = [
            "classifications_validation_gram_check",
            "classifications_validation_bentuk_check",
            "classifications_classification_gram_check",
            "classifications_classification_bentuk_check"
        ]
        
        for con in constraints:
            try:
                connection.execute(text(f"ALTER TABLE classifications DROP CONSTRAINT IF EXISTS \"{con}\";"))
                print(f"Dropped {con} (if it existed).")
            except Exception as e:
                print(f"Could not drop {con}: {e}")
        
        connection.commit()
        print("Constraint removal complete.")

if __name__ == "__main__":
    drop_constraints()
