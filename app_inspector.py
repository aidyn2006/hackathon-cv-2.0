"""
Digital Inspector - Flask Application
Complete document inspection system with YOLOv8 detection
"""
import os
import io
import uuid
import base64
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, request, render_template, jsonify, redirect, url_for, session, send_file
from werkzeug.utils import secure_filename
from pdf2image import convert_from_bytes
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from pyzbar import pyzbar
import logging

from preprocessing import preprocess_for_yolo
from postprocessing import post_process_detections, calculate_detection_quality_score
import database as db

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inspector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'digital-inspector-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ANNOTATED_FOLDER'] = 'annotated'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ANNOTATED_FOLDER'], exist_ok=True)

model_path = 'best.pt'
if os.path.exists(model_path):
    model = YOLO(model_path)
    import torch
    if torch.cuda.is_available():
        model.to('cuda')
        logger.info(f"✅ YOLO with CUDA GPU")
    elif torch.backends.mps.is_available():
        model.to('mps')
        logger.info(f"✅ YOLO with MPS (Apple Silicon)")
    else:
        logger.info(f"✅ YOLO on CPU")
else:
    model = None
    logger.warning(f"❌ Model not found!")

db.init_db()

@app.context_processor
def utility_processor():
    def get_image_base64(image_path):
        """Convert image file to base64 for embedding in HTML"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
        except:
            return ""
    return dict(get_image_base64=get_image_base64)

def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = db.verify_user(username, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['full_name'] = user['full_name']
            logger.info(f"User logged in: {username}")
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout"""
    username = session.get('username', 'Unknown')
    session.clear()
    logger.info(f"User logged out: {username}")
    return redirect(url_for('login'))

@app.route('/')
@login_required
def dashboard():
    """Dashboard with statistics"""
    user_id = session['user_id']
    
    stats = db.get_dashboard_stats(user_id)
    
    documents = db.get_user_documents(user_id, limit=10)
    
    return render_template('dashboard.html', 
                         stats=stats, 
                         documents=documents,
                         username=session.get('full_name', session.get('username')))

@app.route('/documents')
@login_required
def documents_list():
    """List all documents"""
    user_id = session['user_id']
    
    detection_type = request.args.get('type', 'all')
    min_confidence = float(request.args.get('min_conf', 0.0))
    
    documents = db.get_user_documents(user_id, limit=100)
    
    return render_template('documents.html',
                         documents=documents,
                         detection_type=detection_type,
                         min_confidence=min_confidence)

@app.route('/upload')
@login_required
def upload_page():
    """Upload page"""
    return render_template('upload.html')

@app.route('/reports')
@login_required
def reports():
    """Reports page"""
    user_id = session['user_id']
    stats = db.get_dashboard_stats(user_id)
    documents = db.get_user_documents(user_id)
    
    return render_template('reports.html', 
                         stats=stats,
                         documents=documents)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def detect_qr_codes(image):
    """Detect QR codes using pyzbar"""
    qr_codes = []
    decoded_objects = pyzbar.decode(image)
    
    for obj in decoded_objects:
        qr_codes.append({
            'type': obj.type,
            'data': obj.data.decode('utf-8'),
            'rect': obj.rect
        })
    
    return qr_codes

@app.route('/api/upload', methods=['POST'])
@login_required
def upload_document():
    """Upload and process document"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        user_id = session['user_id']
        original_filename = secure_filename(file.filename)
        
        file_ext = original_filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{file_ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(file_path)
        file_size = os.path.getsize(file_path)
        
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        
        if file_ext == 'pdf':
            images = convert_from_bytes(file_bytes, dpi=250)
            pages = len(images)
        else:
            images = [Image.open(io.BytesIO(file_bytes))]
            pages = 1
        
        document_id = db.save_document(
            user_id, unique_filename, original_filename, file_path, file_size, pages
        )
        
        logger.info(f"Processing document {document_id}: {original_filename}")
        
        results = []
        for page_num, pil_image in enumerate(images):
            page_results = process_page(pil_image, document_id, page_num + 1)
            results.append(page_results)
        
        db.update_document_status(document_id, 'completed')
        
        logger.info(f"Document {document_id} processed successfully")
        
        return jsonify({
            'success': True,
            'document_id': document_id,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        if 'document_id' in locals():
            db.update_document_status(document_id, 'failed')
        return jsonify({'error': str(e)}), 500

def process_page(pil_image, document_id, page_number):
    """Process single page with YOLO detection"""
    img_array = np.array(pil_image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    preprocessed_img = preprocess_for_yolo(img_cv, target_size=640, fast_mode=True)
    
    yolo_results = []
    if model is not None:
        yolo_detections = model(preprocessed_img, 
                               conf=0.20,
                               imgsz=640,
                               iou=0.5,
                               verbose=False,
                               half=True)
        
        for detection in yolo_detections:
            boxes = detection.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = model.names[cls]
                
                yolo_results.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
    
    qr_codes = detect_qr_codes(preprocessed_img)
    
    yolo_results, qr_codes = post_process_detections(yolo_results, preprocessed_img, qr_codes)
    
    annotated_img = draw_annotations(preprocessed_img.copy(), yolo_results, qr_codes)
    
    annotated_filename = f"doc_{document_id}_page_{page_number}.png"
    annotated_path = os.path.join(app.config['ANNOTATED_FOLDER'], annotated_filename)
    cv2.imwrite(annotated_path, annotated_img)
    
    original_filename = f"doc_{document_id}_page_{page_number}_original.png"
    original_path = os.path.join(app.config['ANNOTATED_FOLDER'], original_filename)
    cv2.imwrite(original_path, preprocessed_img)
    
    for det in yolo_results:
        db.save_detection(
            document_id, page_number, det['class'], det['confidence'],
            det['bbox'], annotated_image_path=annotated_path
        )
    
    for qr in qr_codes:
        rect = qr['rect']
        bbox = [rect.left, rect.top, rect.left + rect.width, rect.top + rect.height]
        db.save_detection(
            document_id, page_number, 'qr-code', qr.get('confidence', 1.0),
            bbox, qr_data=qr.get('data', ''), qr_valid=qr.get('valid', True),
            annotated_image_path=annotated_path
        )
    
    _, annotated_buffer = cv2.imencode('.png', annotated_img)
    annotated_base64 = base64.b64encode(annotated_buffer).decode('utf-8')
    
    _, original_buffer = cv2.imencode('.png', preprocessed_img)
    original_base64 = base64.b64encode(original_buffer).decode('utf-8')
    
    return {
        'page': page_number,
        'image': annotated_base64,
        'original_image': original_base64,
        'detections': yolo_results,
        'qr_codes': [{'data': qr.get('data', ''), 'valid': qr.get('valid', True)} for qr in qr_codes]
    }

def draw_annotations(image, detections, qr_codes):
    """Draw bounding boxes on image"""
    img = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        class_name = det['class']
        conf = det['confidence']
        
        if 'signature' in class_name.lower() or 'подпись' in class_name.lower():
            color = (0, 255, 0)
        elif 'stamp' in class_name.lower() or 'штамп' in class_name.lower():
            color = (144, 238, 144)
        else:
            color = (255, 255, 255)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        label = f"{class_name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (label_width, label_height), _ = cv2.getTextSize(label, font, 0.6, 2)
        
        if y1 - label_height - 10 > 0:
            label_y = y1 - 8
            cv2.rectangle(img, (x1, y1 - label_height - 10), 
                         (x1 + label_width + 10, y1), color, -1)
        else:
            label_y = y1 + label_height + 2
            cv2.rectangle(img, (x1, y1), 
                         (x1 + label_width + 10, y1 + label_height + 10), color, -1)
        
        cv2.putText(img, label, (x1 + 5, label_y), font, 0.6, (0, 0, 0), 2)
    
    for qr in qr_codes:
        rect = qr['rect']
        x1, y1 = rect.left, rect.top
        x2, y2 = rect.left + rect.width, rect.top + rect.height
        
        color = (0, 255, 255) if qr.get('valid', True) else (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        label = f"QR: {qr.get('data', '')[:20]}..."
        font = cv2.FONT_HERSHEY_SIMPLEX
        (label_width, label_height), _ = cv2.getTextSize(label, font, 0.5, 2)
        
        if y1 - label_height - 10 > 0:
            label_y = y1 - 8
            cv2.rectangle(img, (x1, y1 - label_height - 10),
                         (x1 + label_width + 10, y1), color, -1)
        else:
            label_y = y1 + label_height + 2
            cv2.rectangle(img, (x1, y1),
                         (x1 + label_width + 10, y1 + label_height + 10), color, -1)
        
        cv2.putText(img, label, (x1 + 5, label_y), font, 0.5, (0, 0, 0), 2)
    
    return img

@app.route('/document/<int:document_id>')
@login_required
def document_details(document_id):
    """Show document details with detections"""
    document = db.get_document_by_id(document_id)
    if not document or document['user_id'] != session['user_id']:
        return "Document not found", 404
    
    detections = db.get_document_detections(document_id)
    
    pages = {}
    for det in detections:
        page = det['page_number']
        if page not in pages:
            pages[page] = {'detections': [], 'qr_codes': []}
        
        if 'qr' in det['detection_type'].lower():
            pages[page]['qr_codes'].append(det)
        else:
            pages[page]['detections'].append(det)
    
    return render_template('document_details.html',
                         document=document,
                         pages=pages,
                         detections=detections)

@app.route('/document/<int:document_id>/download')
@login_required
def download_document(document_id):
    """Download original document"""
    document = db.get_document_by_id(document_id)
    if not document or document['user_id'] != session['user_id']:
        return "Document not found", 404
    
    return send_file(document['file_path'], 
                    as_attachment=True,
                    download_name=document['original_filename'])

@app.route('/document/<int:document_id>/annotated/<int:page>')
@login_required
def download_annotated(document_id, page):
    """Download annotated page"""
    document = db.get_document_by_id(document_id)
    if not document or document['user_id'] != session['user_id']:
        return "Document not found", 404
    
    annotated_path = os.path.join(app.config['ANNOTATED_FOLDER'], 
                                  f"doc_{document_id}_page_{page}.png")
    
    if not os.path.exists(annotated_path):
        return "Annotated image not found", 404
    
    return send_file(annotated_path, as_attachment=True,
                    download_name=f"{document['original_filename']}_page_{page}_annotated.png")

@app.route('/api/stats')
@login_required
def api_stats():
    """Get user statistics"""
    user_id = session['user_id']
    stats = db.get_dashboard_stats(user_id)
    return jsonify(stats)

@app.route('/api/documents')
@login_required
def api_documents():
    """Get user documents"""
    user_id = session['user_id']
    documents = db.get_user_documents(user_id)
    return jsonify({'documents': documents})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)