#!/bin/bash
# Скрипт для запуска Digital Inspector через Docker Compose

set -e

echo "🚀 Digital Inspector - Запуск через Docker Compose"
echo "=================================================="

# Переход в директорию проекта
cd "$(dirname "$0")"

# Проверка наличия Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен!"
    echo "Установите Docker: https://www.docker.com/get-started"
    exit 1
fi

# Проверка наличия Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose не установлен!"
    exit 1
fi

# Проверка наличия моделей
if [ ! -f "best.pt" ] || [ ! -f "best-4.pt" ]; then
    echo "❌ Файлы моделей не найдены!"
    echo "Необходимы файлы: best.pt и best-4.pt"
    exit 1
fi

echo ""
echo "📦 Сборка Docker образа..."
docker-compose build

echo ""
echo "🚀 Запуск контейнера..."
docker-compose up -d

echo ""
echo "⏳ Ожидание запуска приложения..."
sleep 5

echo ""
echo "✅ Приложение запущено!"
echo ""
echo "📍 Доступно по адресам:"
echo "   - http://localhost:5002"
echo "   - http://127.0.0.1:5002"
echo ""
echo "👤 Логин по умолчанию:"
echo "   Username: admin"
echo "   Password: admin123"
echo ""
echo "📊 Команды для управления:"
echo "   docker-compose logs -f           # Просмотр логов"
echo "   docker-compose ps                # Статус контейнера"
echo "   docker-compose stop              # Остановка"
echo "   docker-compose restart           # Перезапуск"
echo "   docker-compose down              # Остановка и удаление"
echo ""
echo "📝 Логи также доступны в файле: inspector.log"
echo ""
echo "🎉 Готово к работе!"
