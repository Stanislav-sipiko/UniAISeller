# /root/ukrsell_v4/scripts/generate_metadata_schema.py v1.0.0
import json
import sys
import random
import asyncio
import re
from pathlib import Path
from collections import defaultdict

# Настройка путей для импорта ядра
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from core.llm_selector import LLMSelector
from core.logger import logger

async def ask_llm_for_schema(selector: LLMSelector, samples: list) -> list:
    """Отправляет выборку товаров в LLM для определения ключей фильтрации."""
    items_text = ""
    for i, p in enumerate(samples):
        # Собираем все имеющиеся ключи из атрибутов для контекста
        attr_keys = [a['key'] for a in p.get('attributes', [])]
        items_text += f"Item {i+1}: {p['title']} | Existing Keys: {', '.join(attr_keys)}\n"

    prompt = f"""
    Analyze these products from a store and identify 3-5 universal attribute keys for a filtering system.
    These keys must exist in the product 'attributes' list or be easily derivable.
    Examples: 'Тварина', 'Вид виробу', 'Виробник' or 'Brand', 'Model'.
    
    Products:
    {items_text}
    
    Return ONLY a JSON object with a 'keys' list.
    Example: {{"keys": ["Key1", "Key2"]}}
    """
    
    logger.info(f"📡 LLM_SCHEMA: Analyzing {len(samples)} samples to define store structure...")
    
    try:
        client, model_name = await selector.get_fast()
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a data architect. Output ONLY valid JSON. No prose."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content
        match = re.search(r'(\{[\s\S]*?\})', content)
        if match:
            data = json.loads(match.group(1))
            return data.get("keys", [])
    except Exception as e:
        logger.error(f"❌ LLM_SCHEMA_ERROR: {str(e)}")
    
    return []

async def generate_schema(slug: str):
    base_path = Path(f"/root/ukrsell_v4/stores/{slug}")
    products_path = base_path / "products.json"
    schema_path = base_path / "schema.json"

    if not products_path.exists():
        logger.error(f"SCHEMA_GEN: File not found {products_path}")
        return

    with open(products_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    # 1. Группировка по категориям для стратифицированной выборки
    categories = defaultdict(list)
    for p in products:
        # Ищем категорию в атрибутах (Вид виробу или Категория)
        cat = "General"
        for attr in p.get("attributes", []):
            if attr["key"] in ["Вид виробу", "Категорія", "Тип"]:
                cat = attr["value"]
                break
        categories[cat].append(p)

    logger.info(f"📊 SCHEMA_GEN: Found {len(categories)} categories in '{slug}'")

    # 2. Стратифицированная выборка (Алгоритм: не более 10 товаров суммарно)
    limit = 10
    cat_list = list(categories.keys())
    samples = []

    if cat_list:
        per_cat = max(1, limit // len(cat_list))
        for cat in cat_list:
            items = categories[cat]
            samples.extend(random.sample(items, min(len(items), per_cat)))

    # Добор до 10, если вышло меньше
    if len(samples) < limit and len(products) > len(samples):
        remaining = [p for p in products if p not in samples]
        samples.extend(random.sample(remaining, min(len(remaining), limit - len(samples))))

    samples = samples[:limit]
    
    # 3. Работа с LLM
    selector = LLMSelector()
    await selector.ensure_ready()
    
    keys = await ask_llm_for_schema(selector, samples)
    
    if keys:
        schema_data = {"keys": keys, "generated_at": "2026-03-07"}
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema_data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ SCHEMA_GEN: Success! Keys defined: {keys}")
    else:
        logger.error("❌ SCHEMA_GEN: Failed to define keys.")

    await selector.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 generate_metadata_schema.py <slug>")
        sys.exit(1)
    
    asyncio.run(generate_schema(sys.argv[1]))