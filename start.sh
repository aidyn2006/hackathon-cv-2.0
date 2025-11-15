#!/bin/bash

# Переходим в директорию проекта
cd "$(dirname "$0")"

# Устанавливаем PYTHONPATH
export PYTHONPATH="/Users/aidyn/Downloads/hacknu/venv/lib/python3.13/site-packages:$PYTHONPATH"

# Запускаем приложение
exec python3 app.py

