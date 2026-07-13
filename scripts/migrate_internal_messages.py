"""
Migration script untuk membuat tabel internal_messages.
Tabel ini digunakan untuk menyimpan thread pesan antara Analis dan Dokter
terkait proses revisi spesimen.

Usage:
    python scripts/migrate_internal_messages.py
"""
import os
import sys

# Add project root to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from dotenv import load_dotenv


def ensure_internal_messages_table(conn):
    """Create internal_messages table and indexes if they don't exist."""
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS internal_messages (
            id SERIAL PRIMARY KEY,
            specimen_id INTEGER NOT NULL REFERENCES specimens(id) ON DELETE CASCADE,
            sender_id INTEGER NOT NULL REFERENCES users(id),
            message_text TEXT NOT NULL,
            created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
    """))
    # Create index for fast lookup by specimen_id
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_internal_messages_specimen_id
        ON internal_messages(specimen_id)
    """))
    print("[OK] internal_messages table ready")


def main():
    load_dotenv()
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:8702584@localhost:5432/ta_kita",
    )
    engine = create_engine(database_url)
    with engine.begin() as conn:
        ensure_internal_messages_table(conn)
        print("[OK] Migration completed successfully")


if __name__ == "__main__":
    main()
