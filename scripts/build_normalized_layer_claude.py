# /root/ukrsell_v4/scripts/build_normalized_layer.py v2.8
import os
import json
import asyncio
import re
import sys
import time
import html
from pathlib import Path
from collections import defaultdict

# Настройка путей
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from core.llm_selector import LLMSelector
from core.logger import logger
from core.translator import TextTranslator

# Константы для валидации
ALLOWED_ANIMALS = {"dog", "cat", "rabbit", "rodent", "bird", "fish", "reptile"}
ALLOWED_CATEGORIES = {
    "Grooming", "Feeding & Watering", "Apparel", 
    "Collars & Leashes", "Toys", "Health", "Beds & Furniture"
}
MAX_BATCHES = 3  # ОГРАНИЧЕНИЕ ДЛЯ ТЕСТА

class StatsCollector:
    def __init__(self):
        self.metrics = defaultdict(lambda: {"success": 0, "total": 0, "times": []})

    def add(self, model_name: str, duration: float, is_success: bool):
        m = self.metrics[model_name]
        m["total"] += 1
        if is_success:
            m["success"] += 1
            m["times"].append(duration)

    def print_report(self):
        print("\n" + "="*60)
        print(f"{'📊 ОТЧЕТ ПО НОРМАЛИЗАЦИИ (v2.8 POST-PROCESS)':^60}")
        print("="*60)
        for name, data in self.metrics.items():
            eff = (data["success"] / data["total"] * 100) if data["total"] > 0 else 0
            avg = (sum(data["times"]) / len(data["times"])) if data["times"] else 0
            print(f"{name[:35]:<35} | {eff:>5.0f}% | {avg:>6.2f}s")

def robust_json_repair(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"```json|```", "", text)
    text = re.sub(r'(?<=: ")(.*?)(?=",?\s*\n)', lambda m: m.group(1).replace('"', '\\"'), text)
    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1: return ""
    return text[start:end+1].strip()

# Мультиязычные синонимы категорий: EN → [UA, RU, смешанные]
CATEGORY_SYNONYMS = {
    "Feeding & Watering": ["їжа", "корм", "годування", "поїлка", "миска",
                           "ласощі", "лакомства", "food", "treat", "treats", "feeding"],
    "Food":               ["їжа", "корм", "годування", "ласощі", "лакомства",
                           "food", "treat", "treats", "кормление"],
    "Grooming":           ["груминг", "догляд", "чистка", "шампунь", "щітка",
                           "grooming", "уход", "гигиена"],
    "Apparel":            ["одяг", "одежда", "костюм", "куртка", "комбінезон",
                           "взуття", "apparel", "clothes", "clothing"],
    "Collars & Leashes":  ["нашийник", "ошейник", "повідець", "поводок", "шлея",
                           "шлейка", "collar", "leash", "harness"],
    "Toys":               ["іграшка", "игрушка", "toy", "toys", "гра"],
    "Health":             ["здоров'я", "здоровье", "ліки", "вітаміни",
                           "vitamins", "health"],
    "Beds & Furniture":   ["лежак", "лежанка", "будиночок", "будка", "кошик",
                           "bed", "house", "furniture"],
    "Travel":             ["перевезення", "перевозка", "переноска", "автокрісло",
                           "carrier", "travel", "transport"],
    "Accessories":        ["аксесуари", "аксессуары", "accessories"],
}

# Мультиязычные синонимы животных
ANIMAL_SYNONYMS = {
    "dog":     ["собака", "собаки", "собачий", "пес", "для собак", "dog", "dogs"],
    "cat":     ["кішка", "кішки", "котик", "кіт", "коти", "для кішок", "cat", "cats"],
    "rabbit":  ["кролик", "кролики", "rabbit"],
    "rodent":  ["гризун", "хом'як", "морська свинка", "rodent", "hamster"],
    "bird":    ["птах", "папуга", "bird", "parrot"],
    "fish":    ["риба", "рибка", "fish"],
    "reptile": ["рептилія", "черепаха", "reptile"],
}

def build_search_blob(title: str, brand: str, category: str, animal_list: list) -> str:
    """
    Будує розширений пошуковий блоб з мультимовними синонімами.
    Це ключовий текст за яким FAISS шукає товар.
    """
    title_clean = html.unescape(title or "")
    name_clean = re.sub(r"[^\w\s]", " ", title_clean.lower())

    parts = [name_clean, (brand or "").lower(), category.lower()]

    # Синоніми категорії (EN + UA + RU)
    cat_syns = CATEGORY_SYNONYMS.get(category, [])
    parts.extend(cat_syns)

    # Синоніми тварин
    for animal in (animal_list or []):
        animal_syns = ANIMAL_SYNONYMS.get(animal.lower(), [animal.lower()])
        parts.extend(animal_syns)

    # Дедуплікація з збереженням порядку
    final_tokens = []
    seen = set()
    for segment in " ".join(parts).split():
        if segment not in seen and len(segment) > 1:
            final_tokens.append(segment)
            seen.add(segment)
    return " ".join(final_tokens)

async def post_process_data(collection: list, translator: TextTranslator):
    """Финальная 'прическа' данных: перевод и исправление пустот."""
    print(f"🧹 Запуск Post-Processing для {len(collection)} товаров...")
    cache = {}

    for item in collection:
        attrs = item.get("attributes", {})
        new_attrs = {}
        
        for k, v in attrs.items():
            if isinstance(v, str) and v.strip():
                val_clean = v.strip()
                # Если уже на укр (есть кириллица), просто нормализуем регистр
                if re.search('[а-яА-ЯёЁіІїЇєЄ]', val_clean):
                    new_attrs[k] = val_clean.capitalize()
                else:
                    # Переводим через Google с кэшированием
                    if val_clean.lower() not in cache:
                        translated = await translator.translate(val_clean, target_lang='uk')
                        cache[val_clean.lower()] = translated.capitalize()
                    new_attrs[k] = cache[val_clean.lower()]
            else:
                new_attrs[k] = v
        
        item["attributes"] = new_attrs
        
        # Исправление заголовков (регистр)
        if item.get("title"):
            item["title"] = item["title"].strip()
            
    print("✅ Post-Processing завершен.")
    return collection

async def normalize_store(slug: str):
    base_path = Path(f"/root/ukrsell_v4/stores/{slug}")
    raw_path = base_path / "products.json"
    out_path = base_path / "products_normalized.json"

    if not raw_path.exists():
        logger.error(f"❌ Файл продуктов не найден: {raw_path}")
        return

    with open(raw_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    # Загрузка прогресса (Resume Logic)
    normalized_collection = []
    processed_ids = set()
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                normalized_collection = json.load(f)
                processed_ids = {str(p["product_id"]) for p in normalized_collection}
                print(f"ℹ️ Загружено из кэша: {len(processed_ids)} товаров.")
        except: pass

    selector = LLMSelector()
    translator = TextTranslator()
    stats = StatsCollector()
    await selector.ensure_ready()

    model_pool = [
        {"type": "openrouter", "model": "deepseek/deepseek-chat", "label": "deepseek_v3"},
        {"type": "openrouter", "model": "qwen/qwen-2.5-72b-instruct", "label": "qwen_fallback"}
    ]
    
    batch_size = 5
    batches_count = 0

    prompt_template = """Normalize pet products to JSON. Return ONLY a JSON array.
    ALLOWED ANIMALS: {animals}
    ALLOWED CATEGORIES: {categories}
    SCHEMA: {{"animal": [], "category": "", "brand": "", "attributes": {{"subtype": "", "purpose": ""}}}}
    DATA: {batch_json}"""

    for i in range(0, len(products), batch_size):
        if MAX_BATCHES and batches_count >= MAX_BATCHES:
            break

        current_batch = products[i:i + batch_size]
        batch_ids = {str(p.get("product_id")) for p in current_batch}
        
        if batch_ids.issubset(processed_ids):
            continue

        llm_input = [{"id": str(p.get("product_id")), "title": p.get("title"), 
                      "attrs": {a['key']: a['value'] for a in p.get("attributes", [])[:5]}} 
                     for p in current_batch]

        success_batch = False
        for model_entry in model_pool:
            client, model_name = selector.prepare_entry(model_entry, tier_label="PROD")
            if not client: continue

            start_t = time.time()
            try:
                resp = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt_template.format(
                        animals=", ".join(ALLOWED_ANIMALS),
                        categories=", ".join(ALLOWED_CATEGORIES),
                        batch_json=json.dumps(llm_input, ensure_ascii=False)
                    )}],
                    temperature=0.0
                )
                
                raw_text = resp.choices[0].message.content
                repaired = robust_json_repair(raw_text)
                batch_data = json.loads(repaired)

                for idx, norm_item in enumerate(batch_data):
                    orig_p = current_batch[idx]
                    p_id = str(orig_p.get("product_id"))
                    if p_id in processed_ids: continue

                    anim = [a for a in norm_item.get("animal", []) if a in ALLOWED_ANIMALS] or ["dog"]
                    cat = norm_item.get("category") if norm_item.get("category") in ALLOWED_CATEGORIES else "Grooming"

                    normalized_collection.append({
                        "product_id": orig_p.get("product_id"),
                        "title": html.unescape(orig_p.get("title")),
                        "price": orig_p.get("price"),
                        "availability": orig_p.get("availability", "InStock"),
                        "seller": "Lucky Dog",
                        "animal": anim,
                        "category": cat,
                        "attributes": norm_item.get("attributes", {}),
                        "image_url": orig_p.get("image_url"),
                        "product_url": orig_p.get("product_url"),
                        "search_blob": build_search_blob(orig_p.get("title"), norm_item.get("brand"), cat, anim)
                    })
                    processed_ids.add(p_id)

                # Сохраняем черновик
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(normalized_collection, f, ensure_ascii=False, indent=2)

                stats.add(model_name, time.time() - start_t, True)
                success_batch = True
                batches_count += 1
                print(f"✅ Батч {batches_count} обработан.")
                break

            except Exception as e:
                stats.add(model_name, time.time() - start_t, False)
                logger.warning(f"⚠️ Ошибка {model_name}: {str(e)[:50]}")

    # ФИНАЛЬНАЯ ПРИЧЕСКА
    normalized_collection = await post_process_data(normalized_collection, translator)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(normalized_collection, f, ensure_ascii=False, indent=2)
    
    await selector.close()
    stats.print_report()

if __name__ == "__main__":
    asyncio.run(normalize_store("luckydog"))