"""
Migration: Fix model_training_status.progress column.
Changes from NUMERIC(3,2) to FLOAT to support 0-100 percentage values.
"""
import os
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.environ.get("DATABASE_URL", "")
if not DB_URL:
    print("DATABASE_URL not set in .env")
    exit(1)

import psycopg2

conn = psycopg2.connect(DB_URL)
conn.autocommit = True
cur = conn.cursor()

# Check current type
cur.execute("""
    SELECT data_type, numeric_precision, numeric_scale 
    FROM information_schema.columns 
    WHERE table_name = 'model_training_status' AND column_name = 'progress'
""")
row = cur.fetchone()
if row:
    print(f"Current: type={row[0]}, precision={row[1]}, scale={row[2]}")
    cur.execute("ALTER TABLE model_training_status ALTER COLUMN progress TYPE FLOAT")
    print("Fixed: progress column changed to FLOAT")
else:
    print("Column not found")

cur.close()
conn.close()
print("Done")
