import sqlite3

DB_PATH = 'projectaudit.db'

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

try:
    cursor.execute("ALTER TABLE projects ADD COLUMN similarity_flag TEXT DEFAULT 'UNIQUE';")
    print("✅ similarity_flag column added successfully.")
except sqlite3.OperationalError as e:
    if 'duplicate column name' in str(e):
        print("ℹ️ similarity_flag column already exists.")
    else:
        print(f"❌ Error: {e}")
finally:
    conn.commit()
    cursor.close()
    conn.close()
