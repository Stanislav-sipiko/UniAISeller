# /root/ukrsell_v4/core/intelligence.py v7.4.6
import numpy as np
import re
import json
from core.logger import logger, log_event

# Перевод украинских/русских названий животных в английские
# (нужно для сравнения с EN-значениями поля animal в БД, например ["cat","dog"])
_ANIMAL_TRANSLATIONS: dict = {
    "кіт": "cat",    "кішка": "cat",  "кошеня": "cat", "котик": "cat",
    "кот": "cat",    "кошки": "cat",  "кошка": "cat",
    "собака": "dog", "пес": "dog",    "цуцик": "dog",  "цуценя": "dog",
    "щенок": "dog",  "собаки": "dog", "пса": "dog",    "псу": "dog",
}

def get_stem(text: str) -> str:
    """Извлекает основу слова (минимум 3 символа) для мягкого сравнения."""
    if not text: return ""
    word = str(text).lower().strip()
    # Убираем типичные окончания (украинский/русский/английский)
    word = re.sub(r'(и|ы|і|а|я|ов|ам|ами|ів|у|е|ом|s|es|ed|ing)$', '', word)
    return word if len(word) >= 3 else word

def safe_extract_json(text: str) -> dict:
    """Извлекает JSON из текста, игнорируя мусор вокруг."""
    try:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(text)
    except Exception as e:
        logger.error(f" [INTEL] JSON Extraction Error: {e} | Raw text: {text[:100]}...")
        return {"action": "SEARCH", "entities": {}}

def semantic_guard(products: list, threshold: float = 1.15) -> list:
    """Отсеивает по FAISS Distance (L2)."""
    before_count = len(products)
    filtered = [res for res in products if res.get("score", 999) < threshold]
    logger.debug(f" [INTEL] Semantic Guard: {before_count} -> {len(filtered)} (threshold: {threshold})")
    return filtered

def entity_filter(products: list, intent: dict, intent_mapping: dict = None,
                  category_map: dict = None) -> list:
    """
    Универсальный динамический фильтр (Агностик данных).
    Использует intent_mapping из store_profile.json для связи ключей LLM и БД.

    Исправления v7.4.7:
      - category_map резолвинг: "одяг" → ["Apparel"] до сравнения с полем category в БД.
        Без этого LLM-категория на UA/RU не совпадала с EN-значением в normalized_products.
    """
    # Ключи, которые НЕ являются атрибутами товара и не должны попадать в фильтр
    SERVICE_KEYS = {"price_limit", "action", "resolved_product", "target", "properties"}

    entities = intent.get("entities", {})
    garbage_values = {"no brand", "unknown", "none", "null", "any", "n/a", "", "не вказано", "no_brand"}

    active_filters = {
        k: v for k, v in entities.items()
        if k not in SERVICE_KEYS
        and v is not None
        and str(v).lower() not in garbage_values
    }

    if not active_filters or not products:
        logger.debug(" [INTEL] Entity Filter: No specific entities to filter by.")
        return products

    # ── Category map резолвинг ──────────────────────────────────────────
    # LLM возвращает "одяг", "куртка", "шлея" — UA/RU слова.
    # В БД category хранится на EN: "Apparel", "Walking" и т.д.
    # Резолвим через category_map ДО фильтрации.
    if category_map and "category" in active_filters:
        raw_cat = str(active_filters["category"]).lower().strip()
        # Убираем служебные ключи из category_map
        resolved = category_map.get(raw_cat)
        if resolved:
            # resolved может быть list ["Apparel"] или str "Apparel"
            if isinstance(resolved, list):
                # Берём первый — наиболее специфичный
                active_filters["category"] = resolved[0]
            else:
                active_filters["category"] = resolved
            logger.debug(f" [INTEL] Category resolved: '{raw_cat}' → '{active_filters['category']}'")
        else:
            # Не нашли в category_map — пробуем регистронезависимый поиск
            for map_key, map_val in category_map.items():
                if map_key.startswith("_"):
                    continue
                if raw_cat in map_key or map_key in raw_cat:
                    active_filters["category"] = map_val[0] if isinstance(map_val, list) else map_val
                    logger.debug(f" [INTEL] Category fuzzy-resolved: '{raw_cat}' → '{active_filters['category']}'")
                    break
            else:
                # Совсем не нашли — убираем category из фильтров, пусть FAISS сам ищет
                logger.debug(f" [INTEL] Category '{raw_cat}' not in category_map → removing filter")
                del active_filters["category"]
    # ────────────────────────────────────────────────────────────────────

    category_requested = "category" in active_filters
    mapping = intent_mapping or {k: [k] for k in active_filters.keys()}

    filtered = []
    logger.debug(f" [INTEL] Dynamic Filtering (Map active): {active_filters}")

    for res in products:
        p = res.get("data") if "data" in res else res
        match = True

        for intent_key, target_val in active_filters.items():
            possible_keys = mapping.get(intent_key, [intent_key])
            if isinstance(possible_keys, str):
                possible_keys = [possible_keys]

            found_field_match = False
            field_exists_in_prod = False

            for attr in possible_keys:
                if attr in p and p[attr]:
                    field_exists_in_prod = True
                    prod_val = str(p[attr]).lower()

                    if intent_key == "category":
                        if get_stem(target_val) in get_stem(prod_val):
                            found_field_match = True
                            break
                    else:
                        _target = str(target_val).lower()
                        _target_en = _ANIMAL_TRANSLATIONS.get(_target, _target)
                        if (_target in prod_val or get_stem(_target) in prod_val
                                or _target_en in prod_val):
                            found_field_match = True
                            break

            if field_exists_in_prod and not found_field_match:
                match = False
                break

        if match:
            filtered.append(res)

    if not filtered:
        # Всегда делаем мягкий fallback на топ-3 raw hits.
        # Confidence Engine и LLM-промпт с явным CATALOG сами определят релевантность.
        # Удалена жёсткая ветка «category_requested → return []», которая давала
        # result_count=0 → mode=NO_RESULTS даже когда товар физически есть в каталоге.
        logger.warning(
            f" [INTEL] Zero results after strict filtering "
            f"(active_filters={list(active_filters.keys())}). "
            f"Falling back to top-3 raw hits."
        )
        return products[:3]

    return filtered

# Категории, которые LLM возвращает как generic-мусор (не реальные DB-категории)
_GARBAGE_CATEGORIES = {
    "тварина", "тварини", "вид виробу", "одяг для тварин", "розмір одягу для тварин",
    "розмір", "виробник", "матеріал", "колір", "стать тварини",
    "animal", "product type", "clothing for animals", "size",
}

def merge_followup(prev_intent: dict, new_intent: dict) -> dict:
    """
    Склейка контекста диалога.
    FIX FU-05: не перезаписываем category если новое значение — мусорная generic-категория
    от LLM, а предыдущий контекст содержит реальную категорию.
    """
    merged = prev_intent.copy()
    new_ents = new_intent.get("entities", {})

    if "entities" not in merged:
        merged["entities"] = {}

    merged_ents = merged.get("entities", {}).copy()
    garbage_vals = {"no brand", "unknown", "none", "null", "any", "n/a", "", "no_brand"}

    for key, val in new_ents.items():
        if not val or str(val).lower() in garbage_vals:
            continue  # пустое — пропускаем

        # Для category: не перезаписываем хорошее значение мусорным
        if key == "category":
            prev_cat = merged_ents.get("category", "")
            new_cat_lower = str(val).lower().strip()
            if new_cat_lower in _GARBAGE_CATEGORIES:
                # Новая категория — мусор. Сохраняем предыдущую если она была
                if prev_cat:
                    logger.debug(f"[merge_followup] Ignoring garbage category '{val}', keeping '{prev_cat}'")
                    continue
                else:
                    # Предыдущей не было — пишем null, пусть FAISS сам ищет
                    merged_ents[key] = None
                    continue

        merged_ents[key] = val

    merged["entities"] = merged_ents
    merged["action"] = new_intent.get("action", "SEARCH")
    # Пробрасываем response_text для CHAT/TROLL
    if new_intent.get("response_text"):
        merged["response_text"] = new_intent["response_text"]
    return merged

def deduplicate_products(products: list, top_k: int = 5) -> list:
    """Дедупликация по имени товара."""
    seen = set()
    deduped = []
    for res in products:
        # Поддержка форматов: {"data": ...}, {"product": ...}, плоский словарь
        if isinstance(res, dict):
            p = res.get("data") or res.get("product") or res
        else:
            p = res
        name = p.get("name") or p.get("title", "")
        if not name: continue
        key = re.sub(r'[^\w]', '', str(name).lower())
        if key not in seen:
            seen.add(key)
            deduped.append(res)
        if len(deduped) >= top_k: break
    return deduped