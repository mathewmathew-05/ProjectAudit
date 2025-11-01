from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import sqlite3
import uuid
import datetime
import logging
import os

try:
    import torch
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('all-MiniLM-L6-v2')
    SIMILARITY_ENABLED = True
except Exception:
    SIMILARITY_ENABLED = False
    model = None
    util = None

app = Flask(__name__)
CORS(app)
DUPLICATE_THRESHOLD = 92.0
HIGH_SIMILARITY_THRESHOLD = 78.0
MEDIUM_SIMILARITY_THRESHOLD = 65.0
DB_PATH = "projectaudit.db"

@app.route('/api/debug_db', methods=['GET'])
def debug_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # List tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        # For each table, get columns and a few rows
        data = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [col[1] for col in cursor.fetchall()]
            cursor.execute(f"SELECT * FROM {table} LIMIT 5")
            rows = cursor.fetchall()
            data[table] = {'columns': columns, 'rows': rows}
        conn.close()
        return jsonify({'tables': tables, 'data': data})
    except Exception as e:
        return jsonify({'error': str(e)})


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            domain TEXT,
            description TEXT,
            assignedFacultyEmail TEXT NOT NULL,
            assignedFacultyName TEXT NOT NULL,
            submittedBy TEXT NOT NULL,
            submittedByName TEXT NOT NULL,
            submittedOn TEXT,
            status TEXT,
            similarity_percentage REAL DEFAULT 0,
            similarity_flag TEXT DEFAULT 'UNIQUE',
            faculty_comment TEXT,
            updated_at TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS project_similarity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id_1 TEXT NOT NULL,
            project_id_2 TEXT NOT NULL,
            similarity REAL NOT NULL,
            UNIQUE(project_id_1, project_id_2)
        )
    ''')
    conn.commit()
    cursor.close()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def fetch_one(query, params=None):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute(query, params or [])
            result = cursor.fetchone()
            return dict(result) if result else None
        except sqlite3.Error as err:
            logging.error(f"Error fetching one row: {err}")
            return None
        finally:
            cursor.close()
            conn.close()
    return None

def fetch_all(query, params=None):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute(query, params or [])
            results = cursor.fetchall()
            return [dict(row) for row in results]
        except sqlite3.Error as err:
            logging.error(f"Error fetching all rows: {err}")
            return []
        finally:
            cursor.close()
            conn.close()
    return []

def execute_query(query, params=None):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute(query, params or [])
            conn.commit()
            return True
        except sqlite3.Error as err:
            logging.error(f"Error executing query: {err}")
            conn.rollback()
            return False
        finally:
            cursor.close()
            conn.close()
    return False

def get_user_by_email_db(email):
    query = "SELECT id, name, email, password, role FROM users WHERE email = ?"
    return fetch_one(query, (email.strip().lower(),))

def get_project_by_id_db(project_id):
    query = "SELECT * FROM projects WHERE id = ?"
    return fetch_one(query, (project_id,))

def get_all_faculty_db():
    query = "SELECT id, name, email, role FROM users WHERE role = 'faculty'"
    return fetch_all(query)

def calculate_basic_similarity(new_description, existing_descriptions):
    if not existing_descriptions:
        return 0.0
    new_words = set(new_description.lower().split())
    max_similarity = 0.0
    for existing_desc in existing_descriptions:
        existing_words = set(existing_desc.lower().split())
        if len(new_words) == 0 or len(existing_words) == 0:
            continue
        intersection = new_words.intersection(existing_words)
        union = new_words.union(existing_words)
        similarity = (len(intersection) / len(union)) * 100 if len(union) > 0 else 0
        max_similarity = max(max_similarity, similarity)
    return max_similarity

def calculate_semantic_similarity(new_project, existing_projects):
    if not SIMILARITY_ENABLED or not existing_projects:
        existing_descriptions = [proj['description'] for proj in existing_projects if proj.get('description')]
        new_text = f"{new_project['title']} {new_project['description']}"
        return calculate_basic_similarity(new_text, existing_descriptions), None
    try:
        new_text = f"{new_project['title']} {new_project['description']}"
        existing_texts = [f"{proj['title']} {proj['description']}" for proj in existing_projects]
        new_embedding = model.encode(new_text, convert_to_tensor=True)
        existing_embeddings = model.encode(existing_texts, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(new_embedding, existing_embeddings)[0]
        max_score, max_idx = torch.max(cosine_scores, dim=0)
        max_score = max_score.item() * 100
        if max_score > 0 and max_idx.item() < len(existing_projects):
            most_similar_proj = existing_projects[max_idx.item()]
            return max_score, most_similar_proj
        return max_score, None
    except Exception as e:
        logging.error(f"Error in semantic similarity calculation: {e}")
        existing_descriptions = [proj['description'] for proj in existing_projects if proj.get('description')]
        new_text = f"{new_project['title']} {new_project['description']}"
        return calculate_basic_similarity(new_text, existing_descriptions), None

HTML_TEMPLATE = open('templates/index.html', 'r', encoding='utf-8').read() if os.path.exists('templates/index.html') else """
<!DOCTYPE html>
<html><head><title>ProjectAudit - Service Starting</title></head>
<body>
<h1>ProjectAudit Backend is Running</h1>
<p>Please ensure the frontend HTML file is properly loaded.</p>
<p>Database Status: Connected</p>
<p>AI Similarity: {}</p>
</body></html>
""".format("Enabled" if SIMILARITY_ENABLED else "Disabled - Using Basic Similarity")


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data received'}), 400
        name = data.get('name', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        role = data.get('role', '')
        if not all([name, email, password, role]):
            return jsonify({'success': False, 'message': 'All fields are required.'}), 400
        if get_user_by_email_db(email):
            return jsonify({'success': False, 'message': 'Email already registered.'}), 409
        if role not in ['student', 'faculty']:
            return jsonify({'success': False, 'message': 'Invalid role specified.'}), 400
        user_id = str(uuid.uuid4())
        query = "INSERT INTO users (id, name, email, password, role) VALUES (?, ?, ?, ?, ?)"
        if execute_query(query, (user_id, name, email, password, role)):
            return jsonify({'success': True, 'message': 'Registration successful!'}), 201
        else:
            return jsonify({'success': False, 'message': 'Registration failed - database error.'}), 500
    except Exception as e:
        logging.error(f"Registration error: {e}")
        return jsonify({'success': False, 'message': 'Registration failed - server error.'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data received'}), 400
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        role = data.get('role', '')
        user = get_user_by_email_db(email)
        if not user or user['password'] != password or user['role'] != role:
            return jsonify({'success': False, 'message': 'Invalid credentials or role mismatch.'}), 401
        return jsonify({
            'success': True,
            'message': 'Login successful!',
            'user_id': user['id'],
            'name': user['name'],
            'email': user['email'],
            'role': user['role']
        })
    except Exception as e:
        logging.error(f"Login error: {e}")
        return jsonify({'success': False, 'message': 'Login failed - server error.'}), 500

@app.route('/api/projects', methods=['POST'])
def submit_project():
    def update_project_similarity(new_project_id, new_title, new_description, faculty_email):
        other_projects_query = "SELECT id, title, description FROM projects WHERE assignedFacultyEmail = ? AND id != ?"
        other_projects = fetch_all(other_projects_query, (faculty_email, new_project_id))
        for proj in other_projects:
            sim, _ = calculate_semantic_similarity({'title': new_title, 'description': new_description}, [proj])
            execute_query("REPLACE INTO project_similarity (project_id_1, project_id_2, similarity) VALUES (?, ?, ?)", (new_project_id, proj['id'], sim))
            execute_query("REPLACE INTO project_similarity (project_id_1, project_id_2, similarity) VALUES (?, ?, ?)", (proj['id'], new_project_id, sim))

    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': 'No data received'}), 400

    title = data.get('title', '').strip()
    domain = data.get('domain', '')
    description = data.get('description', '').strip()
    assigned_faculty_email = data.get('assignedFacultyEmail', '').strip().lower()
    submitted_by_email = data.get('submittedByEmail', '').strip().lower()
    submitted_by_name = data.get('submittedByName', '').strip()

    if not all([title, domain, description, assigned_faculty_email, submitted_by_email, submitted_by_name]):
        return jsonify({'success': False, 'message': 'Missing required project fields.'}), 400

    faculty = get_user_by_email_db(assigned_faculty_email)
    if not faculty or faculty['role'] != 'faculty':
        return jsonify({'success': False, 'message': 'Assigned faculty not found or invalid.'}), 400

    submitting_student = get_user_by_email_db(submitted_by_email)
    if not submitting_student or submitting_student['role'] != 'student':
        return jsonify({'success': False, 'message': 'Submitting user not found or is not a student.'}), 400

    existing_projects_query = "SELECT title, description FROM projects WHERE submittedBy != ?"
    existing_projects = fetch_all(existing_projects_query, (submitted_by_email,))
    new_project = {'title': title, 'description': description}
    similarity_percentage, most_similar_proj = calculate_semantic_similarity(new_project, existing_projects)

    # Determine similarity flag
    if similarity_percentage >= DUPLICATE_THRESHOLD:
        similarity_flag = 'DUPLICATE'
    elif similarity_percentage >= HIGH_SIMILARITY_THRESHOLD:
        similarity_flag = 'HIGH_SIMILARITY'
    elif similarity_percentage >= MEDIUM_SIMILARITY_THRESHOLD:
        similarity_flag = 'MEDIUM_SIMILARITY'
    else:
        similarity_flag = 'UNIQUE'

    project_id = str(uuid.uuid4())
    submitted_on = datetime.datetime.now().isoformat()
    insert_query = """
    INSERT INTO projects (id, title, domain, description, assignedFacultyEmail, assignedFacultyName,
                         submittedBy, submittedByName, submittedOn, status, similarity_percentage, similarity_flag)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    success = execute_query(insert_query, (
        project_id, title, domain, description, assigned_faculty_email,
        faculty['name'], submitted_by_email, submitted_by_name,
        submitted_on, 'pending', similarity_percentage, similarity_flag
    ))

    if success:
        update_project_similarity(project_id, title, description, assigned_faculty_email)
        response = {
            'success': True,
            'message': 'Project submitted successfully!',
            'project': {
                'id': project_id,
                'title': title,
                'similarity_percentage': round(similarity_percentage, 2),
                'similarity_flag': similarity_flag
            }
        }
        if similarity_flag == 'DUPLICATE':
            response['similarity_warning'] = f"⚠️ POTENTIAL DUPLICATE: Your project is {similarity_percentage:.1f}% similar to an existing project."
            if most_similar_proj:
                response['similarity_warning'] += f" Similar to '{most_similar_proj.get('title', 'Unknown Project')}'"
        elif similarity_flag == 'HIGH_SIMILARITY':
            response['similarity_warning'] = f"📋 HIGH SIMILARITY: Your project is {similarity_percentage:.1f}% similar to existing content."
        return jsonify(response), 201
    else:
        return jsonify({'success': False, 'message': 'Project submission failed - database error.'}), 500

@app.route('/api/projects/student', methods=['GET'])
def get_student_projects():
    try:
        student_email = request.args.get('email')
        if not student_email:
            return jsonify({'success': False, 'message': 'Student email is required.'}), 400
        clean_email = student_email.strip().lower()
        logging.info(f"[DEBUG] Student project query for email: '{clean_email}'")
        query = '''SELECT id, title, domain, description, assignedFacultyEmail, assignedFacultyName, 
                          submittedBy, submittedByName, submittedOn, updated_at, status, similarity_percentage, 
                          similarity_flag, faculty_comment 
                   FROM projects 
                   WHERE LOWER(submittedBy) = LOWER(?) 
                   ORDER BY submittedOn DESC'''
        student_projects = fetch_all(query, (clean_email,))
        logging.info(f"[DEBUG] Found {len(student_projects)} projects for student: '{clean_email}'")
        if not student_projects:
            logging.warning(f"No projects found for student: {clean_email}")
        return jsonify(student_projects)
    except Exception as e:
        logging.error(f"Error fetching student projects: {e}")
        return jsonify([])

@app.route('/api/projects/faculty', methods=['GET'])
def get_faculty_projects():
    try:
        faculty_email = request.args.get('email')
        if not faculty_email:
            return jsonify({'success': False, 'message': 'Faculty email is required.'}), 400
        clean_email = faculty_email.strip().lower()
        logging.info(f"[DEBUG] Faculty project query for email: '{clean_email}'")
        query = "SELECT * FROM projects WHERE LOWER(assignedFacultyEmail) = LOWER(?) ORDER BY similarity_percentage DESC, submittedOn DESC"
        faculty_projects = fetch_all(query, (clean_email,))
        logging.info(f"[DEBUG] Found {len(faculty_projects)} projects for faculty: '{clean_email}'")
        return jsonify(faculty_projects)
    except Exception as e:
        logging.error(f"Error fetching faculty projects: {e}")
        return jsonify([])

@app.route('/api/projects/<project_id>/status', methods=['PUT'])
def update_project_status(project_id):
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data received'}), 400
        new_status = data.get('status')
        faculty_comment = data.get('faculty_comment')
        project = get_project_by_id_db(project_id)
        if not project:
            logging.error(f"Project not found for status update: {project_id}")
            return jsonify({'success': False, 'message': 'Project not found.'}), 404
        if new_status not in ['approved', 'rejected', 'pending']:
            logging.error(f"Invalid status attempted: {new_status}")
            return jsonify({'success': False, 'message': 'Invalid status.'}), 400
        if new_status == 'rejected' and not faculty_comment:
            faculty_comment = "Rejected by faculty"
        update_query = """
        UPDATE projects SET status = ?, faculty_comment = ?, updated_at = ?
        WHERE id = ?
        """
        try:
            result = execute_query(update_query, (new_status, faculty_comment, datetime.datetime.now().isoformat(), project_id))
            if result:
                return jsonify({'success': True, 'message': 'Project status updated.'}), 200
            else:
                logging.error(f"Failed to update project status for {project_id} (query returned False)")
                return jsonify({'success': False, 'message': 'Failed to update project status.'}), 500
        except Exception as e:
            logging.error(f"Exception during project status update: {e}")
            return jsonify({'success': False, 'message': f'Exception: {e}'}), 500
    except Exception as e:
        logging.error(f"Status update error: {e}")
        return jsonify({'success': False, 'message': 'Status update failed - server error.'}), 500

@app.route('/api/projects/<project_id>', methods=['PUT'])
def resubmit_project(project_id):
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data received'}), 400
        title = data.get('title', '').strip()
        description = data.get('description', '').strip()
        project = get_project_by_id_db(project_id)
        if not project:
            return jsonify({'success': False, 'message': 'Project not found.'}), 404
        existing_projects_query = "SELECT title, description FROM projects WHERE id != ? AND submittedBy != ?"
        existing_projects = fetch_all(existing_projects_query, (project_id, project['submittedBy']))
        new_project = {'title': title, 'description': description}
        similarity_percentage, _ = calculate_semantic_similarity(new_project, existing_projects)
        if similarity_percentage >= DUPLICATE_THRESHOLD:
            similarity_flag = 'DUPLICATE'
        elif similarity_percentage >= HIGH_SIMILARITY_THRESHOLD:
            similarity_flag = 'HIGH_SIMILARITY'
        elif similarity_percentage >= MEDIUM_SIMILARITY_THRESHOLD:
            similarity_flag = 'MEDIUM_SIMILARITY'
        else:
            similarity_flag = 'UNIQUE'
        update_query = """
        UPDATE projects SET title = ?, description = ?, status = 'pending',
        faculty_comment = NULL, similarity_percentage = ?, similarity_flag = ?, updated_at = ?, submittedOn = ?
        WHERE id = ?
        """
        now_iso = datetime.datetime.now().isoformat()
        if execute_query(update_query, (
            title, description, similarity_percentage, similarity_flag,
            now_iso, now_iso, project_id
        )):
            return jsonify({
                'success': True, 
                'message': 'Project updated and resubmitted successfully.',
                'similarity_percentage': round(similarity_percentage, 2),
                'similarity_flag': similarity_flag
            }), 200
        else:
            return jsonify({'success': False, 'message': 'Failed to resubmit project.'}), 500
    except Exception as e:
        logging.error(f"Resubmit error: {e}")
        return jsonify({'success': False, 'message': 'Resubmit failed - server error.'}), 500

@app.route('/api/projects/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    try:
        project = get_project_by_id_db(project_id)
        if not project:
            return jsonify({'success': False, 'message': 'Project not found.'}), 404
        query = "DELETE FROM projects WHERE id = ?"
        if execute_query(query, (project_id,)):
            return jsonify({'success': True, 'message': 'Project deleted successfully.'}), 200
        else:
            return jsonify({'success': False, 'message': 'Failed to delete project.'}), 500
    except Exception as e:
        logging.error(f"Delete error: {e}")
        return jsonify({'success': False, 'message': 'Delete failed - server error.'}), 500

@app.route('/api/faculty_list', methods=['GET'])
def get_faculty_list():
    try:
        faculty_users = get_all_faculty_db()
        return jsonify(faculty_users)
    except Exception as e:
        logging.error(f"Faculty list error: {e}")
        return jsonify([])


# --- Similarity Analysis Endpoint ---
@app.route('/api/similarity_analysis', methods=['GET'])
def similarity_analysis():
    try:
        faculty_email = request.args.get('email')
        if not faculty_email:
            return jsonify({'success': False, 'message': 'Faculty email is required.'}), 400
        # Get all projects for this faculty
        query = "SELECT id, title, description, submittedByName, submittedBy, similarity_percentage, similarity_flag, status FROM projects WHERE assignedFacultyEmail = ?"
        projects = fetch_all(query, (faculty_email.strip().lower(),))
        # Find duplicate and high similarity pairs
        duplicate_pairs = []
        high_similarity_pairs = []
        for i, p1 in enumerate(projects):
            for j, p2 in enumerate(projects):
                if i < j:
                    sim = 0
                    try:
                        sim = max(p1.get('similarity_percentage', 0), p2.get('similarity_percentage', 0))
                    except Exception:
                        pass
                    if sim >= DUPLICATE_THRESHOLD:
                        duplicate_pairs.append({
                            'project1': {'title': p1['title'], 'student': p1['submittedByName'], 'status': p1['status']},
                            'project2': {'title': p2['title'], 'student': p2['submittedByName'], 'status': p2['status']},
                            'similarity_score': sim
                        })
                    elif sim >= HIGH_SIMILARITY_THRESHOLD:
                        high_similarity_pairs.append({
                            'project1': {'title': p1['title'], 'student': p1['submittedByName'], 'status': p1['status']},
                            'project2': {'title': p2['title'], 'student': p2['submittedByName'], 'status': p2['status']},
                            'similarity_score': sim
                        })
        return jsonify({
            'total_duplicates': len(duplicate_pairs),
            'total_high_similarity': len(high_similarity_pairs),
            'analysis_timestamp': datetime.datetime.now().isoformat(),
            'duplicate_pairs': duplicate_pairs,
            'high_similarity_pairs': high_similarity_pairs
        })
    except Exception as e:
        logging.error(f"Similarity analysis error: {e}")
        return jsonify({'success': False, 'message': 'Similarity analysis failed.'}), 500

# --- Faculty Stats Endpoint ---
@app.route('/api/faculty_stats', methods=['GET'])
def faculty_stats():
    try:
        faculty_email = request.args.get('email')
        if not faculty_email:
            return jsonify({'success': False, 'message': 'Faculty email is required.'}), 400
        query = "SELECT status, similarity_percentage FROM projects WHERE assignedFacultyEmail = ?"
        projects = fetch_all(query, (faculty_email.strip().lower(),))
        total = len(projects)
        pending = sum(1 for p in projects if p['status'] == 'pending')
        approved = sum(1 for p in projects if p['status'] == 'approved')
        rejected = sum(1 for p in projects if p['status'] == 'rejected')
        duplicates = sum(1 for p in projects if p.get('similarity_percentage', 0) >= DUPLICATE_THRESHOLD)
        avg_similarity = round(sum(p.get('similarity_percentage', 0) for p in projects) / total, 2) if total else 0
        return jsonify({
            'total': total,
            'pending': pending,
            'approved': approved,
            'rejected': rejected,
            'duplicates': duplicates,
            'avg_similarity': avg_similarity
        })
    except Exception as e:
        logging.error(f"Faculty stats error: {e}")
        return jsonify({'success': False, 'message': 'Faculty stats failed.'}), 500

if __name__ == '__main__':
    print("🚀 Starting ProjectAudit Server...")
    print(f"📊 Database: {DB_PATH}")
    print(f"🤖 AI Similarity: {'Enabled' if SIMILARITY_ENABLED else 'Disabled (using basic similarity)'}")
    print(f"📈 Similarity Thresholds: Duplicate≥{DUPLICATE_THRESHOLD}%, High≥{HIGH_SIMILARITY_THRESHOLD}%")
    print("✅ Server ready!")
    init_db()
    app.run(debug=True, port=5000, host='0.0.0.0')
