#!/bin/bash

# Останавливаем все старые процессы
pkill -f "python.*app.py" 2>/dev/null

# Переходим в директорию проекта
cd "$(dirname "$0")"

# Добавляем poppler в PATH
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"

# Активируем виртуальное окружение
source venv/bin/activate

# Проверяем что poppler доступен
if ! command -v pdftoppm &> /dev/null; then
    echo "ERROR: poppler not found in PATH"
    echo "Please install: brew install poppler"
    exit 1
fi

echo "✓ Poppler found: $(which pdftoppm)"
echo "✓ Starting Flask application..."
echo "✓ URL: http://localhost:5001"
echo ""

# Запускаем приложение
python app.py

