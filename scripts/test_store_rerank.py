# /root/ukrsell_v4/scripts/test_store_rerank.py
import sys
import os

# Гарантируем корректные импорты из корня проекта
sys.path.append("/root/ukrsell_v4")

from core.llm_selector import LLMSelector
from core.store_context import StoreContext
from core.logger import logger, log_event

class MockDB:
    """Имитация базы данных смартфонов для проверки реранкинга."""
    def vector_search(self, query, limit=15):
        # Имитируем типичную выдачу векторного поиска, где есть "шум"
        return [
            {
                "name": "Смартфон Nothing Phone (2a) 12/256GB White",
                "brand": "Nothing",
                "price": 14500,
                "category": "смартфоны",
                "attributes": {"ram": "12GB", "storage": "256GB", "screen": "OLED"}
            },
            {
                "name": "Смартфон Google Pixel 7a 8/128GB Sea",
                "brand": "Google",
                "price": 16200,
                "category": "смартфоны",
                "attributes": {"ram": "8GB", "storage": "128GB", "camera": "64MP"}
            },
            {
                "name": "Планшет Apple iPad Air M2 128GB Blue",
                "brand": "Apple",
                "price": 28000,
                "category": "планшеты",
                "attributes": {"chip": "M2", "storage": "128GB", "display": "Liquid Retina"}
            }
        ]

def run_diagnostic():
    print("🚀 ЗАПУСК ТЕСТА SMART RERANKING v2.2 (STRICT DOMAIN)")
    print("="*70)

    selector = LLMSelector()
    db = MockDB()
    
    # Инициализация контекста. 
    # ВАЖНО: 'аксессуары' отсутствуют в списке, чтобы проверить SEARCH_ABORTED
    store = StoreContext(db, selector)
    store.allowed_categories = ["смартфоны", "планшеты"] 

    # Список тестов, нацеленных на проверку ТВОИХ кейсов
    test_queries = [
        "Зарядка 45W для Nothing Phone",                # Должен быть BLOCKED (Accessory)
        "Чехол прозрачный на Pixel 7a",                # Должен быть BLOCKED (Accessory)
        "Смартфон с 12ГБ оперативы до 15000 грн",      # Должен быть SUCCESS (Nothing Phone)
        "Планшет на чипе M2",                          # Должен быть SUCCESS (iPad)
        "PC кабель Type-C",                            # Должен быть BLOCKED (Accessory)
        "Купить iPhone 15 Pro"                         # Должен быть EMPTY (Т.к. в MockDB его нет)
    ]

    for q in test_queries:
        print(f"\n🔍 ТЕСТИРУЕМ ЗАПРОС: {q}")
        results = store.search_products(q)
        
        if not results:
            # Проверяем логи, чтобы понять причину
            print("❌ СТАТУС: Результаты отфильтрованы (Hallucination Guard / No Match)")
        else:
            print(f"✅ СТАТУС: Найдено релевантных позиций: {len(results)}")
            for i, res in enumerate(results):
                marker = " [!] LOW CONFIDENCE" if res.get('low_confidence') else ""
                print(f"   {i+1}. {res['name']} | {res['price']} UAH | Score: {res['rerank_score']:.2f}{marker}")

    print("\n" + "="*70)
    print("Диагностика завершена. Проверь logs/events.log для детального JSON-анализа.")

if __name__ == "__main__":
    run_diagnostic()