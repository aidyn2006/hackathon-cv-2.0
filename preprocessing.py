"""
Модуль препроцессинга изображений для улучшения точности детекции.
Включает: auto-rotate, deskew, CLAHE, sharpening, noise reduction.
"""

import cv2
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def auto_rotate_image(image):
    """
    Автоматическая ротация изображения на основе детекции текста.
    Использует морфологический анализ для определения правильной ориентации.
    """
    try:
        img = image.copy()
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return img
        
        coords = np.vstack([cnt for cnt in contours])
        rect = cv2.minAreaRect(coords)
        angle = rect[2]
        
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
        
        if abs(angle) > 0.5:
            logger.info(f"Auto-rotating image by {angle:.2f} degrees")
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, 
                                borderMode=cv2.BORDER_REPLICATE)
        
        return img
    except Exception as e:
        logger.warning(f"Auto-rotate failed: {str(e)}, returning original image")
        return image

def deskew_image(image):
    """
    Исправление наклона изображения (deskewing).
    Критически важно для точной детекции подписей и штампов.
    """
    try:
        img = image.copy()
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        coords = np.column_stack(np.where(thresh > 0))
        
        if len(coords) == 0:
            return img
        
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Ограничиваем максимальный угол поворота до 10 градусов
        # Это предотвращает неправильное определение ориентации и поворот на 90 градусов
        if abs(angle) > 10:
            logger.info(f"Skipping deskew: angle {angle:.2f} degrees is too large (max 10)")
            return img
        
        if abs(angle) > 0.1:
            logger.info(f"Deskewing image by {angle:.2f} degrees")
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
        
        return img
    except Exception as e:
        logger.warning(f"Deskew failed: {str(e)}, returning original image")
        return image

def apply_clahe_contrast(image):
    """
    Применение CLAHE (Contrast Limited Adaptive Histogram Equalization)
    для улучшения контраста. Особенно эффективно для слабых подписей.
    """
    try:
        img = image.copy()
        
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
        else:
            l = img
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
        l_clahe = clahe.apply(l)
        
        alpha = 1.3
        beta = 10
        l_contrasted = cv2.convertScaleAbs(l_clahe, alpha=alpha, beta=beta)
        
        if len(img.shape) == 3:
            lab_merged = cv2.merge([l_contrasted, a, b])
            result = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)
        else:
            result = l_contrasted
        
        logger.info("Applied CLAHE and contrast enhancement")
        return result
    except Exception as e:
        logger.warning(f"CLAHE failed: {str(e)}, returning original image")
        return image

def sharpen_for_signature(image):
    """
    Применение sharpening специально настроенного для подписей.
    Усиливает края и детали, критично для детекции подписей.
    """
    try:
        img = image.copy()
        
        kernel_sharpening = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        
        sharpened = cv2.filter2D(img, -1, kernel_sharpening)
        
        gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
        
        result = cv2.addWeighted(sharpened, 0.6, unsharp_mask, 0.4, 0)
        
        logger.info("Applied signature sharpening")
        return result
    except Exception as e:
        logger.warning(f"Sharpening failed: {str(e)}, returning original image")
        return image

def remove_noise(image, fast_mode=True):
    """
    Удаление шума и артефактов с изображения.
    Использует комбинацию фильтров для мягкого удаления шума
    без потери важных деталей.
    """
    try:
        img = image.copy()
        
        if fast_mode:
            result = cv2.GaussianBlur(img, (3, 3), 0)
            logger.info("Applied fast noise reduction")
        else:
            if len(img.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(
                    img, None, h=10, hColor=10,
                    templateWindowSize=7, searchWindowSize=21
                )
            else:
                denoised = cv2.fastNlMeansDenoising(
                    img, None, h=10,
                    templateWindowSize=7, searchWindowSize=21
                )
            result = cv2.bilateralFilter(denoised, 5, 50, 50)
            logger.info("Applied full noise reduction")
        
        return result
    except Exception as e:
        logger.warning(f"Noise reduction failed: {str(e)}, returning original image")
        return image

def preprocess_image_pipeline(image, fast_mode=True):
    """
    Полный pipeline препроцессинга изображения.
    
    Args:
        image: numpy array изображения (BGR формат)
        fast_mode: если True, использует быстрые операции
    
    Returns:
        Обработанное изображение (numpy array)
    """
    logger.info(f"Starting image preprocessing pipeline (fast_mode={fast_mode})")
    
    processed = image.copy()
    
    if fast_mode:
        logger.info("Using FAST mode - essential operations only")
        
        processed = remove_noise(processed, fast_mode=True)
        
        processed = deskew_image(processed)
        
        processed = apply_clahe_contrast(processed)
        
    else:
        logger.info("Using FULL mode - all operations")
        
        processed = remove_noise(processed, fast_mode=False)
        
        processed = auto_rotate_image(processed)
        
        processed = deskew_image(processed)
        
        processed = apply_clahe_contrast(processed)
        
        processed = sharpen_for_signature(processed)
    
    logger.info("Preprocessing pipeline completed")
    return processed

def preprocess_for_yolo(image, target_size=640, fast_mode=True):
    """
    Препроцессинг специально для YOLO детекции.
    Применяет улучшения и подготавливает изображение оптимального размера.
    
    Args:
        image: numpy array изображения
        target_size: целевой размер для YOLO (640 для скорости, 1280 для точности)
        fast_mode: быстрый режим обработки (рекомендуется True)
    
    Returns:
        Обработанное изображение
    """
    processed = preprocess_image_pipeline(image, fast_mode=fast_mode)
    
    h, w = processed.shape[:2]
    
    if max(h, w) < target_size:
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        processed = cv2.resize(processed, (new_w, new_h), 
                             interpolation=cv2.INTER_CUBIC)
        logger.info(f"Upscaled image to {new_w}x{new_h}")
    
    return processed

if __name__ == "__main__":
    print("Preprocessing module loaded successfully!")
    print("Available functions:")
    print("- auto_rotate_image()")
    print("- deskew_image()")
    print("- apply_clahe_contrast()")
    print("- sharpen_for_signature()")
    print("- remove_noise()")
    print("- preprocess_image_pipeline()")
    print("- preprocess_for_yolo()")