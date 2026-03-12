#!/bin/bash
# /root/ukrsell_v4/setup_store.sh
set -e

PROJECT_ROOT="/root/ukrsell_v4"
STORE_NAME=$1
DOMAIN="ukrsellbot.com"

if [ -z "$STORE_NAME" ]; then
    echo "❌ Ошибка: Укажите название магазина (например: ./setup_store.sh phonestore)"
    exit 1
fi

echo "--- 🛠 Настройка магазина: $STORE_NAME ---"

# Установка пути, чтобы Python видел папку core
export PYTHONPATH=$PROJECT_ROOT

# 1. Запуск подготовки данных (Очистка + ID Map + FAISS)
echo "Step 1: Preparing Data and Building Index..."
python3 $PROJECT_ROOT/scripts/prepare_data.py $STORE_NAME

# 2. Автоматическая регистрация Webhook
echo "Step 2: Registering Telegram Webhook..."

CONFIG_FILE="$PROJECT_ROOT/stores/$STORE_NAME/config.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "⚠️  Предупреждение: config.json не найден в $CONFIG_FILE. Пропускаю установку Webhook."
else
    # Извлекаем токен через python, так как он уже гарантированно есть в системе
    BOT_TOKEN=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['bot_token'])")
    
    if [ -z "$BOT_TOKEN" ]; then
        echo "❌ Ошибка: bot_token не найден в $CONFIG_FILE"
    else
        # Формируем URL для Единого Шлюза v4
        WEBHOOK_URL="https://$DOMAIN/api/v4/$STORE_NAME"
        
        echo "🔗 Установка Webhook для $STORE_NAME на: $WEBHOOK_URL"
        
        RESPONSE=$(curl -s -X POST "https://api.telegram.org/bot$BOT_TOKEN/setWebhook" \
             -d "url=$WEBHOOK_URL")
        
        if [[ $RESPONSE == *"\"ok\":true"* ]]; then
            echo "✅ Webhook успешно привязан к шлюзу v4."
        else
            echo "❌ Ошибка при регистрации Webhook: $RESPONSE"
        fi
    fi
fi

echo "--- ✅ Магазин $STORE_NAME готов к работе! ---"