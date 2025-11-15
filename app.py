import os
import io
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
from pdf2image import convert_from_bytes
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import base64
from pyzbar import pyzbar
import logging

# Импортируем наши модули улучшений
from preprocessing import preprocess_for_yolo
from postprocessing import post_process_detections, calculate_detection_quality_score

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Создаем папку для загрузок если её нет
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Загружаем модель с оптимальными настройками
model_path = 'best.pt'
if os.path.exists(model_path):
    model = YOLO(model_path)
    logger.info(f"Model loaded successfully from {model_path}")
else:
    model = None
    logger.warning(f"Model file {model_path} not found!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def detect_qr_codes(image):
    """Детекция QR-кодов с помощью pyzbar"""
    qr_codes = []
    decoded_objects = pyzbar.decode(image)
    
    for obj in decoded_objects:
        qr_codes.append({
            'type': obj.type,
            'data': obj.data.decode('utf-8'),
            'rect': obj.rect
        })
    
    return qr_codes

def process_pdf(pdf_bytes):
    """Обработка PDF: конвертация в изображения и детекция с улучшениями"""
    try:
        logger.info("Starting PDF processing")
        
        # Конвертируем PDF в изображения с оптимальным DPI (200 для скорости)
        images = convert_from_bytes(pdf_bytes, dpi=200)
        results = []
        
        for page_num, pil_image in enumerate(images):
            logger.info(f"Processing page {page_num + 1}/{len(images)}")
            
            # Конвертируем PIL в numpy array для OpenCV
            img_array = np.array(pil_image)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # ========== PREPROCESSING ==========
            # Применяем быстрый препроцессинг (deskew, CLAHE)
            logger.info(f"Applying FAST preprocessing to page {page_num + 1}")
            preprocessed_img = preprocess_for_yolo(img_cv, target_size=640, fast_mode=True)
            
            # Детекция с помощью YOLO модели на улучшенном изображении
            yolo_results = []
            if model is not None:
                logger.info(f"Running YOLO detection on page {page_num + 1}")
                # Используем оптимальный размер для баланса скорости и точности
                yolo_detections = model(preprocessed_img, 
                                       conf=0.25,      # Более низкий порог, пост-обработка отфильтрует
                                       imgsz=640,      # Быстрый размер (640 вместо 1280)
                                       iou=0.6,        # NMS threshold
                                       verbose=False)  # Отключаем verbose для скорости
                
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
                
                logger.info(f"Initial detections: {len(yolo_results)}")
            
            # Детекция QR-кодов на улучшенном изображении
            qr_codes = detect_qr_codes(preprocessed_img)
            logger.info(f"Found {len(qr_codes)} QR codes")
            
            # ========== POST-PROCESSING ==========
            # Применяем улучшенный post-processing
            logger.info(f"Applying post-processing to page {page_num + 1}")
            yolo_results, qr_codes = post_process_detections(
                yolo_results, 
                preprocessed_img, 
                qr_codes
            )
            
            # Вычисляем качество детекций
            quality_score = calculate_detection_quality_score(yolo_results)
            logger.info(f"Detection quality score: {quality_score:.3f}")
            
            # Рисуем результаты на улучшенном изображении
            annotated_img = preprocessed_img.copy()
            
            # Рисуем YOLO детекции с улучшенной визуализацией
            for det in yolo_results:
                x1, y1, x2, y2 = map(int, det['bbox'])
                class_name = det['class']
                conf = det['confidence']
                
                # Цвета для разных классов (BGR формат)
                if 'подпись' in class_name.lower() or 'signature' in class_name.lower():
                    color = (0, 255, 0)  # Зеленый для подписей
                    thickness = 3
                elif 'штамп' in class_name.lower() or 'stamp' in class_name.lower():
                    color = (255, 0, 0)  # Синий для штампов
                    thickness = 3
                else:
                    color = (0, 0, 255)  # Красный для остального
                    thickness = 2
                
                # Рисуем прямоугольник
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness)
                
                # Создаем фон для текста для лучшей читаемости
                label = f"{class_name} {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_img, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated_img, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Рисуем QR-коды с индикацией валидности
            for qr in qr_codes:
                rect = qr['rect']
                
                # Цвет зависит от валидности
                if qr.get('valid', True):
                    qr_color = (255, 255, 0)  # Желтый для валидных
                    label_prefix = "✓ QR"
                else:
                    qr_color = (0, 165, 255)  # Оранжевый для невалидных
                    label_prefix = "✗ QR"
                
                cv2.rectangle(annotated_img, (rect.left, rect.top), 
                            (rect.left + rect.width, rect.top + rect.height), 
                            qr_color, 3)
                
                label = f"{label_prefix}: {qr['data'][:20]}..."
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_img, (rect.left, rect.top - label_size[1] - 10),
                             (rect.left + label_size[0], rect.top), qr_color, -1)
                cv2.putText(annotated_img, label, (rect.left, rect.top - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Конвертируем обратно в PIL для отправки
            annotated_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
            
            # Конвертируем в base64 для отправки в браузер
            buffered = io.BytesIO()
            annotated_pil.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Подготавливаем QR коды для JSON
            qr_codes_json = []
            for qr in qr_codes:
                qr_codes_json.append({
                    'type': qr.get('type', 'QRCODE'),
                    'data': qr.get('data', ''),
                    'valid': qr.get('valid', True),
                    'confidence': qr.get('confidence', 1.0),
                    'rect': {
                        'left': qr['rect'].left,
                        'top': qr['rect'].top,
                        'width': qr['rect'].width,
                        'height': qr['rect'].height
                    }
                })
            
            results.append({
                'page': page_num + 1,
                'image': img_str,
                'yolo_detections': yolo_results,
                'qr_codes': qr_codes_json,
                'quality_score': quality_score,
                'stats': {
                    'total_detections': len(yolo_results),
                    'total_qr_codes': len(qr_codes),
                    'high_confidence_detections': sum(1 for det in yolo_results if det['confidence'] > 0.7)
                }
            })
            
            logger.info(f"Page {page_num + 1} processed successfully: "
                       f"{len(yolo_results)} detections, {len(qr_codes)} QR codes")
        
        logger.info(f"PDF processing completed: {len(results)} pages processed")
        return results
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        raise Exception(f"Ошибка обработки PDF: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logger.warning("Upload attempt without file")
        return jsonify({'error': 'Файл не найден'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        logger.warning("Upload attempt with empty filename")
        return jsonify({'error': 'Файл не выбран'}), 400
    
    if file and allowed_file(file.filename):
        try:
            logger.info(f"Processing uploaded file: {file.filename}")
            
            # Читаем PDF в память
            pdf_bytes = file.read()
            logger.info(f"File size: {len(pdf_bytes)} bytes")
            
            # Обрабатываем PDF с улучшениями
            results = process_pdf(pdf_bytes)
            
            # Считаем общую статистику
            total_detections = sum(r['stats']['total_detections'] for r in results)
            total_qr = sum(r['stats']['total_qr_codes'] for r in results)
            avg_quality = np.mean([r['quality_score'] for r in results])
            
            logger.info(f"Processing completed successfully: "
                       f"{len(results)} pages, {total_detections} detections, "
                       f"{total_qr} QR codes, avg quality: {avg_quality:.3f}")
            
            return jsonify({
                'success': True,
                'results': results,
                'summary': {
                    'total_pages': len(results),
                    'total_detections': total_detections,
                    'total_qr_codes': total_qr,
                    'average_quality': float(avg_quality)
                }
            })
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    logger.warning(f"Unsupported file format: {file.filename}")
    return jsonify({'error': 'Неподдерживаемый формат файла'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

