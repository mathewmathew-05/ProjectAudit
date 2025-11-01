import sqlite3

DB_PATH = 'projectaudit.db'
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

print('Tables and row counts:')
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
for (t,) in cur.fetchall():
    try:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        cnt = cur.fetchone()[0]
    except Exception as e:
        cnt = f'ERROR: {e}'
    print(f" - {t}: {cnt}")

print('\nSample users (up to 10):')
try:
    cur.execute('SELECT id, name, email, role FROM users LIMIT 10')
    rows = cur.fetchall()
    if rows:
        for r in rows:
            print(r)
    else:
        print(' (no users)')
except Exception as e:
    print(' users table error:', e)

conn.close()
