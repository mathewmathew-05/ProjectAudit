import sqlite3

db_path = "projectaudit.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Add updated_at column if missing
try:
    cursor.execute("ALTER TABLE projects ADD COLUMN updated_at TEXT;")
    print("Added 'updated_at' column.")
except sqlite3.OperationalError as e:
    print("'updated_at' column may already exist or error:", e)

# Add faculty_comment column if missing
try:
    cursor.execute("ALTER TABLE projects ADD COLUMN faculty_comment TEXT;")
    print("Added 'faculty_comment' column.")
except sqlite3.OperationalError as e:
    print("'faculty_comment' column may already exist or error:", e)

conn.commit()
conn.close()
print("Done.")
