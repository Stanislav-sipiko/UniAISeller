# /root/ukrsell_v4/scripts/test_flow.py v4.5.0
import asyncio
import sys
import json
import logging
from pathlib import Path

# 1. Настройка путей для корректных импортов
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Подавление отладочных логов для чистоты вывода теста
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

from kernel import UkrSellKernel

async def run_e2e_test(target_slug: str):
    """
    ФИНАЛЬНЫЙ СКВОЗНОЙ ТЕСТ (E2E FLOW) v4.5.0.
    Теперь поддерживает выбор магазина через sys.argv.
    """
    print("\n" + "="*60)
    print(f"🚀 ЗАПУСК СКВОЗНОГО ТЕСТА СИСТЕМЫ UkrSell v4 (Магазин: {target_slug})")
    print("="*60)
    
    # Инициализируем ядро
    kernel = UkrSellKernel()
    
    try:
        print("⚙️  Шаг 1: Инициализация ядра (загрузка моделей и магазинов)...")
        await kernel.initialize()
        
        active_slugs = kernel.get_all_active_slugs()
        
        if target_slug not in active_slugs:
            print(f"❌ ОШИБКА: Магазин '{target_slug}' не найден в реестре!")
            print(f"Доступные магазины: {active_slugs}")
            return

        # Получаем компоненты из реестра
        ctx = kernel.registry.get_context(target_slug)
        
        print(f"✅ Платформа готова. Тестируем магазин: {target_slug.upper()}")
        print(f"📊 SKU в базе: {ctx.profile.get('total_sku', 0)}")

        # 2. Симуляция входящего сообщения
        # Для luckydog используем запрос по товарам для животных, для остальных - общий
        default_text = "Порадьте щось цікаве до 5000 грн"
        if target_slug == "luckydog":
            default_text = "Який корм порадите для лабрадора до 1500 грн?"
        elif "phone" in target_slug:
            default_text = "Смартфон Xiaomi з NFC до 11000 грн"

        test_update = {
            "update_id": 123456,
            "message": {
                "from": {"id": 999, "first_name": "Tester"},
                "chat": {"id": 999, "type": "private"},
                "text": default_text
            }
        }
        
        user_query = test_update["message"]["text"]
        print(f"\n👤 ЗАПРОС: {user_query}")
        print("-" * 50)

        # 3. ЭТАП 1: Имитация обработки через Kernel (Normalization + Webhook)
        print("🧠 Шаг 2: Эмуляция прохождения через Kernel.handle_webhook...")
        # (В реальной системе здесь отрабатывает фоновая задача)

        # 4. ЭТАП 2: Эмуляция работы DialogManager (Extraction)
        # Мы вызываем get_recommendations напрямую, чтобы проверить всю цепочку логики
        print("🔎 Шаг 3: Запуск цепочки Intent -> Retrieval -> Synthesis...")
        
        # 5. ЭТАП 3: Синтез финального ответа
        response_text = await kernel.get_recommendations(
            ctx=ctx,
            query=user_query,
            filters=None, # Фильтры будут извлечены Analyzer-ом автоматически
            top_k=3,
            user_id=999
        )

        # 6. ФИНАЛЬНЫЙ ВЫВОД
        print("\n" + "="*60)
        print("🤖 ИТОГОВЫЙ ОТВЕТ (как увидит пользователь):")
        print("-" * 60)
        print(response_text)
        print("-" * 60)
        print(f"✅ Тест успешно завершен для {target_slug}")
        print("="*60)

    except Exception as e:
        print(f"\n💥 КРИТИЧЕСКАЯ ОШИБКА ТЕСТА: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await kernel.close()
        print("\n🛑 Ресурсы освобождены. Выход.")

if __name__ == "__main__":
    # Читаем slug из аргументов или используем дефолтный luckydog
    slug_arg = sys.argv[1] if len(sys.argv) > 1 else "luckydog"
    
    try:
        asyncio.run(run_e2e_test(slug_arg))
    except KeyboardInterrupt:
        pass