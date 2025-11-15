"""
Модуль post-processing для улучшения результатов детекции YOLO.
Включает: фильтрацию bbox, NMS, merge близких боксов, валидацию QR.
"""

import cv2
import numpy as np
from pyzbar import pyzbar
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_iou(box1, box2):
    """
    Вычисление Intersection over Union (IoU) между двумя боксами.
    
    Args:
        box1, box2: [x1, y1, x2, y2]
    
    Returns:
        IoU значение (0-1)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Вычисляем координаты пересечения
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # Площадь пересечения
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height
    
    # Площади боксов
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # IoU
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    
    return inter_area / union_area


def filter_low_confidence_detections(detections, min_confidence=0.3):
    """
    Удаление детекций с низкой уверенностью.
    Критично важно: всё ниже 0.3 - шум.
    
    Args:
        detections: список словарей с детекциями
        min_confidence: минимальный порог confidence
    
    Returns:
        Отфильтрованный список детекций
    """
    filtered = [det for det in detections if det['confidence'] >= min_confidence]
    
    removed_count = len(detections) - len(filtered)
    if removed_count > 0:
        logger.info(f"Filtered out {removed_count} low-confidence detections (< {min_confidence})")
    
    return filtered


def non_max_suppression(detections, iou_threshold=0.6):
    """
    Применение Non-Maximum Suppression для удаления дублирующихся детекций.
    
    Args:
        detections: список словарей с детекциями
        iou_threshold: порог IoU для считания боксов перекрывающимися
    
    Returns:
        Список детекций после NMS
    """
    if len(detections) == 0:
        return []
    
    # Группируем детекции по классам
    classes = {}
    for det in detections:
        class_name = det['class']
        if class_name not in classes:
            classes[class_name] = []
        classes[class_name].append(det)
    
    # Применяем NMS к каждому классу отдельно
    final_detections = []
    
    for class_name, class_detections in classes.items():
        # Сортируем по confidence (по убыванию)
        class_detections = sorted(class_detections, 
                                 key=lambda x: x['confidence'], 
                                 reverse=True)
        
        keep = []
        while len(class_detections) > 0:
            # Берем детекцию с наибольшей уверенностью
            best = class_detections[0]
            keep.append(best)
            class_detections = class_detections[1:]
            
            # Удаляем все сильно перекрывающиеся детекции
            filtered = []
            for det in class_detections:
                iou = calculate_iou(best['bbox'], det['bbox'])
                if iou < iou_threshold:
                    filtered.append(det)
            
            class_detections = filtered
        
        final_detections.extend(keep)
        logger.info(f"NMS for class '{class_name}': kept {len(keep)} detections")
    
    return final_detections


def calculate_distance(box1, box2):
    """
    Вычисление расстояния между центрами двух боксов.
    """
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    
    return np.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)


def merge_nearby_signatures(detections, distance_threshold=50):
    """
    Объединение близко расположенных детекций подписей.
    Часто одна подпись детектируется несколько раз рядом.
    
    Args:
        detections: список детекций
        distance_threshold: максимальное расстояние для объединения
    
    Returns:
        Список с объединенными детекциями
    """
    # Разделяем на подписи и остальное
    signatures = [det for det in detections 
                  if 'подпись' in det['class'].lower() or 
                     'signature' in det['class'].lower()]
    others = [det for det in detections 
              if 'подпись' not in det['class'].lower() and 
                 'signature' not in det['class'].lower()]
    
    if len(signatures) <= 1:
        return detections
    
    # Объединяем близкие подписи
    merged = []
    used = set()
    
    for i, sig1 in enumerate(signatures):
        if i in used:
            continue
        
        # Находим все близкие подписи
        cluster = [sig1]
        cluster_indices = [i]
        
        for j, sig2 in enumerate(signatures):
            if j <= i or j in used:
                continue
            
            distance = calculate_distance(sig1['bbox'], sig2['bbox'])
            if distance < distance_threshold:
                cluster.append(sig2)
                cluster_indices.append(j)
        
        # Если найдены близкие подписи, объединяем их
        if len(cluster) > 1:
            # Берем бокс с максимальной уверенностью как основу
            best = max(cluster, key=lambda x: x['confidence'])
            
            # Расширяем бокс, чтобы включить все близкие
            x1 = min(det['bbox'][0] for det in cluster)
            y1 = min(det['bbox'][1] for det in cluster)
            x2 = max(det['bbox'][2] for det in cluster)
            y2 = max(det['bbox'][3] for det in cluster)
            
            # Усредняем confidence
            avg_conf = np.mean([det['confidence'] for det in cluster])
            
            merged.append({
                'class': best['class'],
                'confidence': avg_conf,
                'bbox': [x1, y1, x2, y2]
            })
            
            used.update(cluster_indices)
            logger.info(f"Merged {len(cluster)} nearby signatures into one")
        else:
            merged.append(sig1)
            used.add(i)
    
    # Добавляем обратно остальные детекции
    final_detections = merged + others
    
    return final_detections


def validate_qr_codes(image, qr_detections):
    """
    Валидация QR-кодов - проверка, что они действительно декодируются.
    Если QR не декодируется, добавляем penalty (понижаем confidence).
    
    Args:
        image: numpy array изображения
        qr_detections: список детекций QR-кодов
    
    Returns:
        Список валидированных QR-кодов с обновленными confidence
    """
    validated = []
    
    for qr in qr_detections:
        try:
            # Извлекаем область QR-кода
            rect = qr['rect']
            x1 = rect.left
            y1 = rect.top
            x2 = rect.left + rect.width
            y2 = rect.top + rect.height
            
            # Добавляем небольшой padding
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            qr_region = image[y1:y2, x1:x2]
            
            # Пробуем декодировать
            decoded = pyzbar.decode(qr_region)
            
            if len(decoded) > 0:
                # QR успешно декодируется
                qr['valid'] = True
                qr['confidence'] = 1.0
                logger.info(f"QR code validated: {qr['data'][:30]}...")
            else:
                # QR не декодируется - понижаем confidence
                qr['valid'] = False
                qr['confidence'] = 0.5
                logger.warning(f"QR code failed validation, applying penalty")
            
            validated.append(qr)
            
        except Exception as e:
            logger.error(f"Error validating QR code: {str(e)}")
            qr['valid'] = False
            qr['confidence'] = 0.3
            validated.append(qr)
    
    return validated


def apply_class_specific_filtering(detections):
    """
    Применение специфичной для класса фильтрации.
    Разные пороги для разных типов объектов.
    """
    filtered = []
    
    class_thresholds = {
        'подпись': 0.25,    # Более низкий порог для подписей (они слабые)
        'signature': 0.25,
        'штамп': 0.35,      # Штампы обычно четче
        'stamp': 0.35,
        'qr': 0.4,          # QR должны быть уверенными
        'default': 0.3      # По умолчанию
    }
    
    for det in detections:
        class_name = det['class'].lower()
        
        # Определяем порог для класса
        threshold = class_thresholds.get('default')
        for key, value in class_thresholds.items():
            if key in class_name:
                threshold = value
                break
        
        if det['confidence'] >= threshold:
            filtered.append(det)
    
    return filtered


def post_process_detections(detections, image=None, qr_codes=None):
    """
    Полный pipeline post-processing для детекций.
    
    Args:
        detections: список YOLO детекций
        image: numpy array изображения (для валидации QR)
        qr_codes: список детектированных QR-кодов
    
    Returns:
        Обработанные детекции и QR-коды
    """
    logger.info(f"Starting post-processing with {len(detections)} initial detections")
    
    # 1. Применяем класс-специфичную фильтрацию
    processed = apply_class_specific_filtering(detections)
    logger.info(f"After class-specific filtering: {len(processed)} detections")
    
    # 2. Удаляем сомнительные детекции (< 0.3 как fallback)
    processed = filter_low_confidence_detections(processed, min_confidence=0.3)
    logger.info(f"After confidence filtering: {len(processed)} detections")
    
    # 3. Применяем NMS с порогом 0.6
    processed = non_max_suppression(processed, iou_threshold=0.6)
    logger.info(f"After NMS: {len(processed)} detections")
    
    # 4. Объединяем близкие детекции подписей
    processed = merge_nearby_signatures(processed, distance_threshold=50)
    logger.info(f"After signature merging: {len(processed)} detections")
    
    # 5. Валидация QR-кодов
    validated_qr = qr_codes
    if image is not None and qr_codes is not None:
        validated_qr = validate_qr_codes(image, qr_codes)
        logger.info(f"Validated {len(validated_qr)} QR codes")
    
    logger.info("Post-processing completed successfully")
    return processed, validated_qr


def calculate_detection_quality_score(detections):
    """
    Вычисление общего показателя качества детекций.
    Полезно для отладки и мониторинга.
    """
    if len(detections) == 0:
        return 0.0
    
    # Средняя confidence
    avg_confidence = np.mean([det['confidence'] for det in detections])
    
    # Количество высококачественных детекций
    high_quality = sum(1 for det in detections if det['confidence'] > 0.7)
    high_quality_ratio = high_quality / len(detections)
    
    # Итоговый score
    quality_score = (avg_confidence * 0.6) + (high_quality_ratio * 0.4)
    
    logger.info(f"Detection quality score: {quality_score:.3f}")
    logger.info(f"  - Average confidence: {avg_confidence:.3f}")
    logger.info(f"  - High quality ratio: {high_quality_ratio:.3f}")
    
    return quality_score


if __name__ == "__main__":
    # Тестирование модуля
    print("Post-processing module loaded successfully!")
    print("Available functions:")
    print("- filter_low_confidence_detections()")
    print("- non_max_suppression()")
    print("- merge_nearby_signatures()")
    print("- validate_qr_codes()")
    print("- post_process_detections()")
    print("- calculate_detection_quality_score()")

