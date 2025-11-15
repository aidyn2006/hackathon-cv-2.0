#!/bin/bash

echo "🚀 Starting Digital Inspector..."
echo "================================"

if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose не установлен!"
    echo "Установите: brew install docker-compose"
    exit 1
fi

if [ ! -f "best.pt" ]; then
    echo "⚠️  Предупреждение: файл best.pt не найден!"
    echo "Приложение запустится, но детекция не будет работать."
fi

echo "📦 Останавливаем старые контейнеры..."
docker-compose down 2>/dev/null

echo "🔨 Собираем Docker образ..."
docker-compose build

echo "🚀 Запускаем приложение..."
docker-compose up -d

echo ""
echo "================================"
echo "✅ Digital Inspector запущен!"
echo "================================"
echo ""
echo "🌐 URL: http://localhost:5002"
echo "👤 Логин: inspector"
echo "🔑 Пароль: demo123"
echo ""
echo "📊 Проверить статус:"
echo "   docker-compose ps"
echo ""
echo "📝 Посмотреть логи:"
echo "   docker-compose logs -f"
echo ""
echo "🛑 Остановить:"
echo "   docker-compose down"
echo ""
echo "================================"

