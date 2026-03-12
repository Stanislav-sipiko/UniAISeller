# /root/ukrsell_v4/scripts/test_intelligence_chain.py v5.16.4
import asyncio
import os
import json
import logging
import sys

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.store_context import StoreContext
from core.analyzer import Analyzer
from core.llm_selector import LLMSelector

# Настройка логгера для теста
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Test_Chain")

async def setup_test_data(test_path):
    """Создает окружение магазина, имитирующее проект №3 (luckydog)."""
    base = os.path.join(test_path, "stores/luckydog")
    os.makedirs(base, exist_ok=True)
    
    # 1. Данные товаров
    id_map = {
        "p1": {"name": "Xiaomi Redmi Note 13", "brand": "xiaomi", "price": 8500.0, "category": "смартфони"},
        "p2": {"name": "Samsung Galaxy S24", "brand": "samsung", "price": 32000.0, "category": "смартфони"}
    }
    with open(os.path.join(base, "id_map.json"), "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)
        
    # 2. Профиль магазина (Критично для Analyzer v5.16.0)
    profile = {
        "store_type": "electronics",
        "total_sku": 2,
        "expertise_fields": ["смартфони", "телефони", "гаджети"],
        "main_category": "Електроніка",
        "price_analytics": {"median": 20000.0}
    }
    with open(os.path.join(base, "store_profile.json"), "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    # 3. Конфиг
    config = {"currency": "грн", "language": "Ukrainian", "use_llm_persona": True}
    with open(os.path.join(base, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return base

async def run_full_test():
    print("\n🚀 ЗАПУСК ИНТЕГРАЦИОННОГО ТЕСТА UKRSELL V4.8 (INTEL CHAIN)\n" + "="*60)
    
    test_store_path = await setup_test_data("/tmp/ukrsell_test_v4")
    
    # Инициализация селектора и принудительная активация моделей через refresh()
    llm_selector = LLMSelector()
    print("🔍 Синхронизация LLM стеков (refresh)...")
    await llm_selector.refresh()
    
    status = llm_selector.get_status()
    print(f"📡 Модели: LIGHT: {status.get('light')} | FAST: {status.get('fast')} | HEAVY: {status.get('heavy')}")
    
    if status.get('fast') == "OFFLINE" or status.get('heavy') == "OFFLINE":
        logger.error("Критическая ошибка: LLM стеки остались OFFLINE после refresh().")
        await llm_selector.close()
        return

    # Инициализация контекста
    ctx = StoreContext(base_path=test_store_path, db_engine=None, llm_selector=llm_selector)
    analyzer = Analyzer(ctx)

    # --- ТЕСТ 1: Извлечение интента ---
    print("\n[1] Тест Extract Intent (Fast Model)...")
    query = "Купити білий Xiaomi до 10000 грн"
    try:
        intent = await analyzer.extract_intent(query)
        print(f"✅ Target: {intent.get('target')}")
        print(f"✅ Properties: {intent.get('properties')}")
    except Exception as e:
        print(f"❌ Ошибка в Extract Intent: {e}")

    # --- ТЕСТ 2: Semantic Guard (Оффтоп) ---
    print("\n[2] Тест Semantic Guard (Heavy Model Logic)...")
    offtopic_query = "У вас є корм для собак?"
    entities_wrong = {"category": "корм для тварин", "action": "SEARCH"}
    
    try:
        resp_guard = await analyzer.synthesize_response(
            search_results={"status": "SEMANTIC_REJECT", "products": []}, 
            entities=entities_wrong, 
            user_query=offtopic_query
        )
        print(f"🤖 Статус: {resp_guard['status']}")
        print(f"🤖 Модель: {resp_guard.get('model_used')}")
        print(f"🤖 Ответ: {resp_guard['text']}")
    except Exception as e:
        print(f"❌ Ошибка в Semantic Guard: {e}")

    # --- ТЕСТ 3: Успешный Grounding (Синтез по товарам) ---
    print("\n[3] Тест Response Synthesis (Grounding)...")
    search_results = {
        "status": "SUCCESS",
        "products": [
            {"product": {"name": "Xiaomi Redmi Note 13", "price": 8500, "link": "https://p.ua/1", "description": "Global Version"}}
        ]
    }
    entities_ok = {"category": "смартфони", "action": "SEARCH"}
    
    try:
        resp_ok = await analyzer.synthesize_response(
            search_results=search_results, 
            entities=entities_ok, 
            user_query="Покажи дешевий Сяомі"
        )
        print(f"🤖 Статус: {resp_ok['status']}")
        print(f"🤖 Модель: {resp_ok.get('model_used')}")
        print(f"🤖 Текст ответа:\n{resp_ok['text']}")
    except Exception as e:
        print(f"❌ Ошибка в Grounding: {e}")

    # Закрываем сессии LLM
    await llm_selector.close()
    print("\n" + "="*60 + "\n✅ ТЕСТ ЗАВЕРШЕН")

if __name__ == "__main__":
    try:
        asyncio.run(run_full_test())
    except Exception as e:
        logger.error(f"Тест провалился с системной ошибкой: {e}")