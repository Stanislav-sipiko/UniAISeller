# /root/ukrsell_v4/scripts/generate_metadata.py
import json
import os
import sys
from pathlib import Path
import google.generativeai as genai

# Настройка путей
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from core.config import get_random_gemini_key, STORES_DIR
from core.logger import logger, log_event

def generate_metadata(store_slug: str):
    store_path = Path(STORES_DIR) / store_slug
    products_path = store_path / "products.json"
    
    if not products_path.exists():
        print(f"❌ Файл {products_path} не найден!")
        return

    # 1. Сбор статистики для промпта
    try:
        with open(products_path, 'r', encoding='utf-8') as f:
            products = json.load(f)
    except Exception as e:
        print(f"❌ Ошибка чтения products.json: {e}")
        return
    
    all_brands = set()
    all_attributes = set()
    sample_products = []

    for i, p in enumerate(products):
        if p.get('brand'): all_brands.add(p['brand'])
        attrs = p.get('attributes', {})
        if isinstance(attrs, dict):
            all_attributes.update(attrs.keys())
        
        # Концентрированная выборка товаров
        if i % (max(1, len(products) // 5)) == 0 and len(sample_products) < 5:
            sample_products.append({
                "name": p.get('name'),
                "price": p.get('price'),
                "attrs": attrs
            })

    # 2. Конфигурация Gemini
    # СТРОГОЕ ИСПОЛЬЗОВАНИЕ GEMINI-2.5-FLASH
    MODEL_ID = "gemini-2.5-flash" 
    
    genai.configure(api_key=get_random_gemini_key())
    model = genai.GenerativeModel(MODEL_ID)

    prompt = f"""
    Ты — эксперт-аналитик маркетплейса. Создай конфигурацию для ИИ-продавца магазина "{store_slug}".
    
    ДАННЫЕ:
    - Бренды: {', '.join(list(all_brands)[:20])}
    - Атрибуты: {', '.join(list(all_attributes)[:30])}
    - Примеры: {json.dumps(sample_products, ensure_ascii=False)}

    ЗАДАЧА:
    Верни JSON (БЕЗ разметки ```json):
    {{
        "expert_profile": "Инструкция по тону и акцентам общения.",
        "key_attributes": ["5 важнейших ключей из списка выше"],
        "comparison_logic": "Приоритеты при сравнении.",
        "shopping_advice": ["Совет 1", "Совет 2", "Совет 3"]
    }}
    """

    print(f"⏳ Запрос к {MODEL_ID} для магазина {store_slug}...")
    try:
        # Прямой вызов без лишних оберток
        response = model.generate_content(prompt)
        
        # Очистка текста от возможных Markdown-тегов, если модель их добавит
        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
            
        metadata = json.loads(raw_text)
        
        # Технические метаданные
        metadata["last_updated_ts"] = os.path.getmtime(products_path)
        metadata["total_items"] = len(products)
        metadata["model_version"] = MODEL_ID

        # 3. Сохранение
        output_path = store_path / "metadata.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        
        print(f"✅ Metadata успешно создана: {output_path}")
        log_event("METADATA_GEN", {"store": store_slug, "status": "success", "model": MODEL_ID})

    except Exception as e:
        print(f"💥 Ошибка генерации: {type(e).__name__} - {e}")
        logger.exception("Metadata generation failed")

if __name__ == "__main__":
    slug = sys.argv[1] if len(sys.argv) > 1 else "phonestore"
    generate_metadata(slug)