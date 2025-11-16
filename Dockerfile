FROM python:3.11-slim-bookworm

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libzbar0 \
    poppler-utils \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Установка Python зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Копирование файлов приложения
COPY app_inspector.py .
COPY preprocessing.py .
COPY postprocessing.py .
COPY database.py .
COPY templates/ ./templates/
COPY static/ ./static/

# Копирование моделей
COPY best.pt .
COPY best-4.pt .

# Создание необходимых директорий
RUN mkdir -p uploads annotated && \
    chmod -R 777 uploads annotated

# Инициализация базы данных (если не существует)
RUN python -c "import database; database.init_db()" || true

EXPOSE 5002

ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5002/login || exit 1

CMD ["python", "app_inspector.py"]

