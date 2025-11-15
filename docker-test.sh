#!/bin/bash

echo "🧪 Проверка Docker конфигурации..."
echo "=================================="
echo ""

echo "1️⃣ Проверка Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен!"
    exit 1
fi
echo "✅ Docker установлен"

echo ""
echo "2️⃣ Проверка docker-compose..."
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose не установлен!"
    exit 1
fi
echo "✅ docker-compose установлен"

echo ""
echo "3️⃣ Проверка файлов..."

files=("Dockerfile" "docker-compose.yml" "requirements.txt" "app_inspector.py" "database.py" "preprocessing.py" "postprocessing.py")

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file не найден!"
        exit 1
    fi
done

echo ""
echo "4️⃣ Проверка YOLO модели..."
if [ -f "best.pt" ]; then
    size=$(du -h best.pt | cut -f1)
    echo "✅ best.pt найден ($size)"
else
    echo "⚠️  best.pt не найден - детекция не будет работать!"
fi

echo ""
echo "5️⃣ Проверка папок..."
mkdir -p uploads annotated
echo "✅ uploads/ и annotated/ готовы"

echo ""
echo "=================================="
echo "✅ Все проверки пройдены!"
echo "=================================="
echo ""
echo "Готово к запуску:"
echo "  ./docker-start.sh"
echo ""
echo "или"
echo ""
echo "  docker-compose up -d"
echo ""

