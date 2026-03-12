import json
import os
from pathlib import Path

def generate_blob(item):
    """
    Генерирует поисковый индекс на основе нормализованных данных.
    Включает: Животные (теги), Категория, Бренд, Подтип, Назначение, Заголовок.
    """
    # Расширенный поиск по животным (синонимы для чата)
    animals_map = {
        "dog": "собака собаки для собак собачий dog dogs",
        "cat": "кіт коти кішка кішки для котів для кішок cat cats",
        "rabbit": "кролик кролики для кроликів",
        "rodent": "гризун гризуни",
        "bird": "птах птахи папуга",
        "fish": "риба риби акваріум",
        "reptile": "рептилія рептилії"
    }
    
    selected_animals = item.get("animal", [])
    animal_tags = " ".join([animals_map.get(a, a) for a in selected_animals])
    
    category = item.get("category", "")
    brand = item.get("brand") or ""
    subtype = item.get("attributes", {}).get("subtype", "")
    purpose = item.get("attributes", {}).get("purpose", "")
    title = item.get("title", "")

    # Формируем финальную строку
    parts = [animal_tags, category, brand, subtype, purpose, title]
    raw_blob = " ".join(filter(None, parts))
    
    # Очистка от лишних пробелов и перевод в нижний регистр
    return " ".join(raw_blob.split()).lower()

def process_final_json(file_path):
    path = Path(file_path)
    if not path.exists():
        print(f"CRITICAL ERROR: File {file_path} not found.")
        return

    with open(path, 'r', encoding='utf-8') as f:
        try:
            products = json.load(f)
        except json.JSONDecodeError as e:
            print(f"JSON ERROR: {e}")
            return

    final_data = []
    
    for item in products:
        # 1. Strict Trim product_id (исправление ошибок Sonnet из ноутсов)
        if "product_id" in item:
            item["product_id"] = str(item["product_id"]).strip()
        
        # 2. Brand Normalization (Collar/WAUDOG/null)
        brand = item.get("brand")
        if brand:
            brand_lower = brand.lower()
            if "collar" in brand_lower:
                item["brand"] = "Collar"
            elif "waudog" in brand_lower:
                item["brand"] = "WAUDOG"
            elif "animall" in brand_lower:
                item["brand"] = "AnimAll"

        # 3. Inject Search Blob
        item["search_blob"] = generate_blob(item)
        
        final_data.append(item)

    # Сохранение финального результата
    output_path = path.parent / "normalized_products_final.json"
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(final_data, f_out, ensure_ascii=False, indent=2)

    print(f"SUCCESS: {len(final_data)} products processed.")
    print(f"FINAL FILE: {output_path}")

if __name__ == "__main__":
    SOURCE_FILE = "/root/ukrsell_v4/stores/luckydog/normalized_products.json"
    process_final_json(SOURCE_FILE)