"""
Database models and initialization for Digital Inspector
"""
import sqlite3
from datetime import datetime
import os

DATABASE_PATH = 'digital_inspector.db'

def init_db():
    """Initialize database with tables"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            full_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Documents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            pages INTEGER DEFAULT 1,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processing_status TEXT DEFAULT 'pending',
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Detections table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            page_number INTEGER DEFAULT 1,
            detection_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            bbox_x1 REAL,
            bbox_y1 REAL,
            bbox_x2 REAL,
            bbox_y2 REAL,
            qr_data TEXT,
            qr_valid BOOLEAN,
            annotated_image_path TEXT,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
    ''')
    
    # Sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_token TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Insert demo users
    demo_users = [
        ('inspector', 'demo123', 'Digital Inspector'),
        ('admin', 'admin123', 'System Administrator'),
        ('user', 'user123', 'Regular User')
    ]
    
    for username, password, full_name in demo_users:
        try:
            cursor.execute(
                'INSERT INTO users (username, password, full_name) VALUES (?, ?, ?)',
                (username, password, full_name)
            )
        except sqlite3.IntegrityError:
            pass  # User already exists
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully")

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def verify_user(username, password):
    """Verify user credentials"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT * FROM users WHERE username = ? AND password = ?',
        (username, password)
    )
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None

def save_document(user_id, filename, original_filename, file_path, file_size, pages=1):
    """Save document to database"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO documents (user_id, filename, original_filename, file_path, file_size, pages)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, filename, original_filename, file_path, file_size, pages))
    document_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return document_id

def save_detection(document_id, page_number, detection_type, confidence, bbox, 
                   qr_data=None, qr_valid=None, annotated_image_path=None):
    """Save detection result to database"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO detections 
        (document_id, page_number, detection_type, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
         qr_data, qr_valid, annotated_image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (document_id, page_number, detection_type, confidence, 
          bbox[0], bbox[1], bbox[2], bbox[3], qr_data, qr_valid, annotated_image_path))
    conn.commit()
    conn.close()

def update_document_status(document_id, status):
    """Update document processing status"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE documents SET processing_status = ? WHERE id = ?',
        (status, document_id)
    )
    conn.commit()
    conn.close()

def get_user_documents(user_id, limit=50):
    """Get all documents for a user"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT d.*, 
               COUNT(det.id) as total_detections,
               SUM(CASE WHEN det.detection_type IN ('signature', 'Signature') THEN 1 ELSE 0 END) as signatures_count,
               SUM(CASE WHEN det.detection_type IN ('stamp', 'Stamp') THEN 1 ELSE 0 END) as stamps_count,
               SUM(CASE WHEN det.detection_type LIKE '%qr%' OR det.detection_type LIKE '%QR%' THEN 1 ELSE 0 END) as qr_count
        FROM documents d
        LEFT JOIN detections det ON d.id = det.document_id
        WHERE d.user_id = ?
        GROUP BY d.id
        ORDER BY d.upload_date DESC
        LIMIT ?
    ''', (user_id, limit))
    documents = cursor.fetchall()
    conn.close()
    return [dict(doc) for doc in documents]

def get_document_by_id(document_id):
    """Get document by ID"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM documents WHERE id = ?', (document_id,))
    document = cursor.fetchone()
    conn.close()
    return dict(document) if document else None

def get_document_detections(document_id):
    """Get all detections for a document"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM detections 
        WHERE document_id = ? 
        ORDER BY page_number, confidence DESC
    ''', (document_id,))
    detections = cursor.fetchall()
    conn.close()
    return [dict(det) for det in detections]

def get_dashboard_stats(user_id):
    """Get dashboard statistics"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Total documents
    cursor.execute('SELECT COUNT(*) as count FROM documents WHERE user_id = ?', (user_id,))
    total_docs = cursor.fetchone()['count']
    
    # Total detections
    cursor.execute('''
        SELECT COUNT(*) as count FROM detections det
        JOIN documents doc ON det.document_id = doc.id
        WHERE doc.user_id = ?
    ''', (user_id,))
    total_detections = cursor.fetchone()['count']
    
    # Detections by type
    cursor.execute('''
        SELECT detection_type, COUNT(*) as count 
        FROM detections det
        JOIN documents doc ON det.document_id = doc.id
        WHERE doc.user_id = ?
        GROUP BY detection_type
    ''', (user_id,))
    by_type = cursor.fetchall()
    
    # Average confidence
    cursor.execute('''
        SELECT AVG(confidence) as avg_conf FROM detections det
        JOIN documents doc ON det.document_id = doc.id
        WHERE doc.user_id = ?
    ''', (user_id,))
    avg_conf = cursor.fetchone()['avg_conf'] or 0
    
    conn.close()
    
    return {
        'total_documents': total_docs,
        'total_detections': total_detections,
        'by_type': [dict(row) for row in by_type],
        'average_confidence': avg_conf
    }

if __name__ == '__main__':
    init_db()
    print("Demo users created:")
    print("  inspector / demo123")
    print("  admin / admin123")
    print("  user / user123")

