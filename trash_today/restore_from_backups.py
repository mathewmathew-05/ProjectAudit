import sqlite3
import csv
import os
import shutil

DB_PATH = 'projectaudit.db'
BACKUP_DIR = 'backups'

# Find expected backup files
proj_backup = None
sim_backup = None
for fname in os.listdir(BACKUP_DIR):
    if fname.startswith('projects_backup_') and fname.endswith('.csv'):
        proj_backup = os.path.join(BACKUP_DIR, fname)
    if fname.startswith('project_similarity_backup_') and fname.endswith('.csv'):
        sim_backup = os.path.join(BACKUP_DIR, fname)

if not proj_backup and not sim_backup:
    print('No backups found to restore.')
    raise SystemExit(1)

# Make a DB snapshot first
snapshot = DB_PATH + '.snapshot'
shutil.copyfile(DB_PATH, snapshot)
print(f'Created DB snapshot: {snapshot}')

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

def restore_table_from_csv(table, csv_path):
    print(f'Restoring {table} from {csv_path}')
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        cols = next(reader)
        rows = list(reader)
    if not rows:
        print(f'No rows found in {csv_path}. Skipping.')
        return
    placeholders = ','.join(['?'] * len(cols))
    insert_sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
    try:
        cur.executemany(insert_sql, rows)
        conn.commit()
        print(f'Inserted {len(rows)} rows into {table}.')
    except Exception as e:
        print(f'Error inserting into {table}: {e}')
        conn.rollback()
        raise

try:
    if proj_backup:
        restore_table_from_csv('projects', proj_backup)
    if sim_backup:
        restore_table_from_csv('project_similarity', sim_backup)
finally:
    cur.close()
    conn.close()

print('Restore complete. If anything went wrong, the snapshot file contains pre-restore DB.')
