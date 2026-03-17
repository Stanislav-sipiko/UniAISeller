# /root/ukrsell_v4/scripts/smoke_test.py v1.0.1
import sys
import os

# Гарантируем, что корень проекта находится в python path для работы импортов core.*
sys.path.append(os.getcwd())

# Явный импорт функций из intelligence.py для предотвращения NameError
try:
    from core.intelligence import (
        safe_extract_json, 
        entity_filter, 
        merge_followup
    )
except ImportError as e:
    print(f"КРИТИЧЕСКАЯ ОШИБКА ИМПОРТА: {e}")
    print("Убедитесь, что вы запускаете скрипт из корня проекта: python3 scripts/smoke_test.py")
    sys.exit(1)

# 1. Тестовые данные (имитация структуры из вашего JSON)
mock_products = [
    {
        "product": {
            "id": "1", 
            "name": "Кігтеріз поперечний", 
            "attributes": {"subtype": "Кігтеріз", "animal": "dog"}
        }
    },
    {
        "product": {
            "id": "2", 
            "name": "Шампунь AnimAll", 
            "attributes": {"subtype": "Шампунь", "animal": "dog"}
        }
    },
    {
        "product": {
            "id": "3", 
            "name": "Светр для котів", 
            "attributes": {"subtype": "Светр", "animal": "cat"}
        }
    }
]

def run_test():
    print("=== Запуск Smoke Test для intelligence.py ===\n")

    # ТЕСТ 1: Извлечение сущностей (Парсинг JSON от LLM)
    raw_response = '{"action": "SEARCH", "entities": {"subtype": "Шампунь", "animal": "dog"}}'
    intent = safe_extract_json(raw_response)
    
    test_1_res = intent.get('entities', {}).get('subtype') == 'Шампунь'
    print(f"[TEST 1] Парсинг JSON: {'OK' if test_1_res else 'FAIL'}")

    # ТЕСТ 2: Фильтрация (Entity Filter)
    # Проверяем, находит ли фильтр конкретный товар по атрибутам
    filtered = entity_filter(mock_products, intent)
    
    test_2_res = len(filtered) == 1 and "Шампунь" in filtered[0]['product']['name']
    print(f"[TEST 2] Фильтрация по атрибутам: {'OK' if test_2_res else 'FAIL'}")

    # ТЕСТ 3: Контекст (Merge Followup)
    # Имитируем ситуацию: сначала искали Шампунь (dog), затем уточнили "для котів"
    prev_intent = intent
    new_raw = '{"action": "SEARCH", "entities": {"animal": "cat"}}'
    new_intent = safe_extract_json(new_raw)
    
    # Слияние должно сохранить 'subtype': 'Шампунь' из прошлого шага и добавить 'animal': 'cat'
    merged = merge_followup(prev_intent, new_intent)
    
    entities = merged.get('entities', {})
    has_subtype = entities.get('subtype') == 'Шампунь'
    has_animal = entities.get('animal') == 'cat'
    
    print(f"[TEST 3] Слияние контекста: {'OK' if (has_subtype and has_animal) else 'FAIL'}")

    print("\n=== Проверка завершена ===")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"ОШИБКА ПРИ ВЫПОЛНЕНИИ ТЕСТА: {e}")
        # Выводим traceback для отладки в CI/CD
        import traceback
        traceback.print_exc()
        sys.exit(1)