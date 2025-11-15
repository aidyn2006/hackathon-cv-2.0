"""
Скрипт для обучения YOLO модели с оптимальными параметрами.
Настроен для максимальной точности детекции подписей, штампов и QR-кодов.
"""

import os
import yaml
from ultralytics import YOLO
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_augmentation_config():
    """
    Создание конфигурации с агрессивными аугментациями.
    Критично для реалистичной работы на документах в разных условиях.
    """
    augmentation_config = {
        # Базовые геометрические трансформации
        'hsv_h': 0.015,        # Изменение цветового тона
        'hsv_s': 0.7,          # Изменение насыщенности
        'hsv_v': 0.4,          # Изменение яркости
        
        # Геометрические аугментации
        'degrees': 10.0,       # Случайное вращение до ±10 градусов
        'translate': 0.1,      # Случайный сдвиг до 10%
        'scale': 0.5,          # Масштабирование от 50% до 150%
        'shear': 2.0,          # Сдвиг (shearing)
        'perspective': 0.0003, # Перспективные искажения
        
        # Аугментации изображения
        'flipud': 0.0,         # Вертикальное отражение (выключено для документов)
        'fliplr': 0.5,         # Горизонтальное отражение (50% шанс)
        'mosaic': 1.0,         # Mosaic augmentation
        'mixup': 0.15,         # MixUp augmentation
        
        # Аугментации качества
        'blur': 0.01,          # Размытие (имитация плохого скана)
        'noise': 0.02,         # Добавление шума
        'brightness': 0.4,     # Изменение яркости
        'contrast': 0.4,       # Изменение контраста
    }
    
    return augmentation_config


def train_model(
    data_yaml_path,
    base_model='yolov8x.pt',  # Используем самую большую модель для максимальной точности
    epochs=100,
    imgsz=1280,
    batch_size=-1,  # Auto batch size
    device='auto',
    project='runs/detect',
    name='signature_stamp_detector',
    patience=20,
    save_period=10
):
    """
    Обучение YOLO модели с оптимальными параметрами.
    
    Args:
        data_yaml_path: путь к data.yaml с датасетом
        base_model: базовая модель для transfer learning
        epochs: количество эпох (рекомендуется 80-120)
        imgsz: размер изображения (1024-1536, критично важно!)
        batch_size: размер батча (-1 для автоопределения)
        device: устройство для обучения
        project: папка проекта
        name: имя эксперимента
        patience: early stopping patience
        save_period: частота сохранения чекпоинтов
    """
    
    logger.info("=" * 80)
    logger.info("Starting YOLO training with optimized parameters")
    logger.info("=" * 80)
    
    # Проверяем наличие data.yaml
    if not os.path.exists(data_yaml_path):
        logger.error(f"Data config not found: {data_yaml_path}")
        logger.info("Please create a data.yaml file with your dataset configuration")
        return None
    
    # Загружаем базовую модель
    logger.info(f"Loading base model: {base_model}")
    model = YOLO(base_model)
    
    # Получаем конфигурацию аугментаций
    aug_config = create_augmentation_config()
    
    # Параметры обучения
    train_params = {
        'data': data_yaml_path,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'device': device,
        'project': project,
        'name': name,
        'exist_ok': True,
        
        # Optimizer настройки (AdamW для лучшей сходимости)
        'optimizer': 'AdamW',
        'lr0': 0.001,          # Initial learning rate
        'lrf': 0.01,           # Final learning rate (lr0 * lrf)
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # Freezing слоев (важно для transfer learning)
        'freeze': 10,          # Замораживаем первые 10 слоев на старте
        
        # Аугментации
        **aug_config,
        
        # Улучшения тренировки
        'patience': patience,
        'save': True,
        'save_period': save_period,
        'cache': True,         # Кэширование изображений в RAM
        'workers': 8,
        'cos_lr': True,        # Cosine learning rate scheduler
        'close_mosaic': 10,    # Отключаем mosaic за 10 эпох до конца
        
        # Валидация
        'val': True,
        'plots': True,
        'verbose': True,
        
        # Дополнительные параметры для детекции мелких объектов
        'amp': True,           # Automatic Mixed Precision
        'multi_scale': True,   # Multi-scale training
    }
    
    logger.info("Training parameters:")
    logger.info(f"  - Image size: {imgsz}")
    logger.info(f"  - Epochs: {epochs}")
    logger.info(f"  - Optimizer: AdamW")
    logger.info(f"  - Frozen layers: 10")
    logger.info(f"  - Base model: {base_model}")
    logger.info("  - Augmentations: ENABLED (aggressive)")
    
    # Запускаем обучение
    logger.info("\nStarting training...")
    logger.info("=" * 80)
    
    try:
        results = model.train(**train_params)
        
        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)
        
        # Информация о результатах
        best_model_path = Path(project) / name / 'weights' / 'best.pt'
        logger.info(f"Best model saved to: {best_model_path}")
        
        return results
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return None


def create_sample_data_yaml():
    """
    Создает пример data.yaml файла для обучения.
    """
    sample_yaml = {
        'path': '../datasets/document_detection',  # Путь к датасету
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        
        # Классы
        'names': {
            0: 'подпись',
            1: 'штамп',
            2: 'qr_code'
        }
    }
    
    # Сохраняем пример
    example_path = 'data_example.yaml'
    with open(example_path, 'w', encoding='utf-8') as f:
        yaml.dump(sample_yaml, f, allow_unicode=True, sort_keys=False)
    
    logger.info(f"Example data.yaml created: {example_path}")
    logger.info("Please modify it according to your dataset structure")


def validate_model(model_path, data_yaml_path, imgsz=1280):
    """
    Валидация обученной модели.
    """
    logger.info("Starting model validation...")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return None
    
    model = YOLO(model_path)
    
    # Запускаем валидацию
    results = model.val(
        data=data_yaml_path,
        imgsz=imgsz,
        batch=1,
        conf=0.25,
        iou=0.6,
        device='auto',
        plots=True,
        save_json=True
    )
    
    logger.info("Validation results:")
    logger.info(f"  - mAP@0.5: {results.box.map50:.4f}")
    logger.info(f"  - mAP@0.5:0.95: {results.box.map:.4f}")
    logger.info(f"  - Precision: {results.box.mp:.4f}")
    logger.info(f"  - Recall: {results.box.mr:.4f}")
    
    return results


def export_model(model_path, formats=['onnx', 'torchscript']):
    """
    Экспорт модели в различные форматы для деплоя.
    """
    logger.info(f"Exporting model to formats: {formats}")
    
    model = YOLO(model_path)
    
    for fmt in formats:
        try:
            exported = model.export(format=fmt, imgsz=1280)
            logger.info(f"  - Successfully exported to {fmt}: {exported}")
        except Exception as e:
            logger.error(f"  - Failed to export to {fmt}: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLO model for document detection')
    parser.add_argument('--data', type=str, default='data.yaml', 
                       help='Path to data.yaml file')
    parser.add_argument('--model', type=str, default='yolov8x.pt',
                       help='Base model to use (yolov8n/s/m/l/x)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (80-120 recommended)')
    parser.add_argument('--imgsz', type=int, default=1280,
                       help='Image size (1024-1536 recommended)')
    parser.add_argument('--batch', type=int, default=-1,
                       help='Batch size (-1 for auto)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cpu/0/1/...)')
    parser.add_argument('--name', type=str, default='signature_stamp_detector',
                       help='Experiment name')
    parser.add_argument('--create-example', action='store_true',
                       help='Create example data.yaml file')
    parser.add_argument('--validate', type=str, default=None,
                       help='Path to model for validation')
    parser.add_argument('--export', type=str, default=None,
                       help='Path to model for export')
    
    args = parser.parse_args()
    
    # Создаем пример конфигурации если нужно
    if args.create_example:
        create_sample_data_yaml()
        exit(0)
    
    # Валидация модели
    if args.validate:
        validate_model(args.validate, args.data, args.imgsz)
        exit(0)
    
    # Экспорт модели
    if args.export:
        export_model(args.export)
        exit(0)
    
    # Обучение модели
    logger.info("Starting training with parameters:")
    logger.info(f"  Data: {args.data}")
    logger.info(f"  Base model: {args.model}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Image size: {args.imgsz}")
    logger.info(f"  Batch: {args.batch}")
    logger.info(f"  Device: {args.device}")
    logger.info("")
    
    results = train_model(
        data_yaml_path=args.data,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        name=args.name
    )
    
    if results:
        logger.info("\n" + "=" * 80)
        logger.info("Training pipeline completed successfully!")
        logger.info("=" * 80)
        logger.info("\nNext steps:")
        logger.info("1. Check the training results in runs/detect/")
        logger.info("2. Validate the model: python train_yolo.py --validate runs/detect/best.pt")
        logger.info("3. Export the model: python train_yolo.py --export runs/detect/best.pt")
        logger.info("4. Replace best.pt in the project root with your trained model")
    else:
        logger.error("Training failed. Please check the logs for details.")

