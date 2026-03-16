# /root/ukrsell_v4/core/intelligence.py v7.6.5
import numpy as np
import re
import json
import os
from typing import List, Dict, Any, Optional, Union
from core.logger import logger, log_event

def get_stem(text: str) -> str:
    """
    Выполняет нормализацию текста и стемминг (удаление окончаний) 
    для улучшения качества сопоставления сущностей.
    """
    if not text: 
        return ""
    
    word = str(text).lower().strip()
    word = re.sub(r'[^\w\s]', '', word)
    
    # Регулярное выражение для типичных окончаний (UA/RU/EN)
    suffixes = r'(атор|ами|іця|иця|ов|ам|ів|ий|ый|ое|ая|ие|ы|і|а|я|у|е|ом|різ|лов|чик|s|es|ed|ing|ка|ок|ик|ою|є|ї|ої|их|ых|им|ым)$'
    word = re.sub(suffixes, '', word)
    
    return word if len(word) >= 3 else word

def safe_extract_json(text: str) -> dict:
    """
    Безопасно извлекает JSON из строки, обрабатывая артефакты генерации LLM
    и незакрытые скобки.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text:
        return {"action": "SEARCH", "entities": {}}
        
    text = text.strip()
    result = {"action": "SEARCH", "entities": {}}
    
    try:
        # Поиск JSON-структуры внутри текста
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            json_str = match.group(1).replace('\t', ' ')
            result = json.loads(json_str)
        else:
            result = json.loads(text)
    except Exception as e:
        # Попытка исправить незакрытый JSON (частая ошибка при обрыве генерации)
        if text.startswith("{") and not text.endswith("}"):
            try:
                fixed_text = text
                if fixed_text.count('{') > fixed_text.count('}'):
                    fixed_text += '}' * (fixed_text.count('{') - fixed_text.count('}'))
                result = json.loads(fixed_text)
            except:
                pass
        else:
            logger.warning(f"[INTEL] JSON Extraction failed: {e}")
            log_event("JSON_PARSE_ERROR", {"error": str(e), "text": text[:200]}, level="warning")

    # Нормализация ключей в секции entities
    if isinstance(result, dict) and "entities" in result:
        ents = result["entities"]
        if isinstance(ents, dict):
            mapping_fixes = {
                "brand_name": "brand",
                "product_type": "subtype",
                "type": "subtype",
                "color_name": "color"
            }
            for wrong_key, right_key in list(mapping_fixes.items()):
                if wrong_key in ents and right_key not in ents:
                    ents[right_key] = ents.pop(wrong_key)
                
    return result

def semantic_guard(products: list, threshold: float = 0.30) -> list:
    """
    Отсеивает товары с низким показателем релевантности (score).
    """
    if not products:
        return []
    
    filtered = []
    for res in products:
        if not isinstance(res, dict):
            continue
        score = res.get("final_score") or res.get("score", 0)
        if score >= threshold:
            filtered.append(res)
            
    return filtered

def entity_filter(
    products: list, 
    intent: dict, 
    intent_mapping: dict = None, 
    category_map: dict = None,
    store_hints: dict = None
) -> list:
    """
    Жесткая фильтрация результатов поиска по извлеченным сущностям.
    Поддерживает гибкое сопоставление через стемминг и синонимы ключей.
    """
    if not isinstance(intent, dict):
        return products

    entities = intent.get("entities", {})
    if not isinstance(entities, dict):
        entities = {}
    
    SERVICE_KEYS = {"price_limit", "action", "target", "properties", "excluded_ids", "taxonomy_hint", "category", "temperature", "language"}
    GARBAGE_VALUES = {"none", "null", "any", "unknown", "", "все", "любой", "товари"}

    active_filters = {}
    brand_ignore_list = []
    
    # Загрузка игнор-листа для брендов из настроек магазина
    if store_hints and "brand_ignore" in store_hints:
        brand_ignore_list = [str(b).lower() for b in store_hints["brand_ignore"]]

    # Формирование списка активных фильтров (исключая служебные ключи и пустые значения)
    for k, v in entities.items():
        if k in SERVICE_KEYS or v is None:
            continue
        val_str = str(v).lower().strip()
        
        if k == "brand" and any(b_ign in val_str for b_ign in brand_ignore_list):
            logger.debug(f"[INTEL] Brand Ignore trigger: {val_str}, skipping brand filter.")
            continue

        if val_str and val_str not in GARBAGE_VALUES:
            active_filters[k] = val_str

    if not products:
        return []

    # Маппинг полей для поиска в атрибутах товара
    mapping = intent_mapping or {
        "brand": ["brand", "manufacturer", "vendor"],
        "color": ["color", "colour", "color_ref"],
        "subtype": ["subtype", "product_type"]
    }

    filtered = []
    
    # БЕЗОПАСНОЕ ПОЛУЧЕНИЕ EXCLUDED_IDS (Fix для NoneType error)
    raw_excluded = intent.get("excluded_ids")
    if isinstance(raw_excluded, list):
        excluded_ids = set(str(i) for i in raw_excluded)
    else:
        excluded_ids = set()

    price_limit = entities.get("price_limit")

    for hit in products:
        if not isinstance(hit, dict):
            continue
            
        p = hit.get("product") or hit.get("data") or hit
        if not isinstance(p, dict):
            continue
            
        p_id = str(p.get("product_id") or p.get("id", ""))
        
        # 1. Проверка на исключенные ID (уже просмотренные товары)
        if p_id in excluded_ids:
            continue

        # 2. Проверка ценового лимита
        if price_limit:
            try:
                p_price = float(p.get("price", 0))
                if p_price > float(price_limit):
                    continue
            except:
                pass

        p_attrs = p.get("attributes") or {}
        p_title = str(p.get("name") or p.get("title", "")).lower()
        
        match_count = 0
        total_filters = len(active_filters)

        if total_filters == 0:
            filtered.append(hit)
            continue

        # 3. Сопоставление по каждому активному фильтру
        for ent_key, ent_val in active_filters.items():
            ent_stem = get_stem(ent_val)
            target_fields = mapping.get(ent_key, [ent_key])
            if isinstance(target_fields, str): 
                target_fields = [target_fields]
            
            field_match = False
            for field in target_fields:
                val_in_prod = p.get(field) or p_attrs.get(field)
                if val_in_prod:
                    # Поддержка списка значений в атрибуте (например, цвета)
                    vals = [str(x).lower() for x in (val_in_prod if isinstance(val_in_prod, list) else [val_in_prod])]
                    if any(ent_val in v or ent_stem in get_stem(v) for v in vals):
                        field_match = True
                        break
            
            # Если в атрибутах не нашли, ищем прямо в названии товара
            if not field_match:
                if ent_val in p_title or ent_stem in get_stem(p_title):
                    field_match = True
            
            if field_match:
                match_count += 1

        # Логика пропуска: полное совпадение или -1 для сложных запросов (>=3 фильтра)
        if match_count == total_filters:
            filtered.append(hit)
        elif total_filters >= 3 and match_count >= (total_filters - 1):
            filtered.append(hit)

    # Защитный механизм: если фильтры убили всё, но есть супер-релевантный топ, возвращаем его
    if not filtered and products:
        first_hit = products[0]
        if isinstance(first_hit, dict):
            top_hit_score = first_hit.get("final_score") or first_hit.get("score", 0)
            if top_hit_score > 0.85:
                return products[:2]

    return filtered

def merge_followup(prev_intent: dict, new_intent: dict, category_map: dict = None) -> dict:
    """
    Объединяет текущий интент с контекстом предыдущего запроса.
    Реализует логику сохранения фильтров при уточнении или сброс при смене категории.
    """
    if not isinstance(prev_intent, dict): return new_intent
    if not isinstance(new_intent, dict): return prev_intent

    merged = prev_intent.copy()
    new_ents = new_intent.get("entities", {})
    prev_ents = prev_intent.get("entities", {})
    
    if not isinstance(new_ents, dict): new_ents = {}
    if not isinstance(prev_ents, dict): prev_ents = {}
    
    def norm_cat(c):
        c = str(c or "").lower().strip()
        if category_map:
            for cat_id, cat_data in category_map.items():
                names = [str(cat_id).lower()]
                if isinstance(cat_data, dict):
                    names.append(str(cat_data.get("name", "")).lower())
                if c in names: return str(cat_id).lower()
        return get_stem(c)

    new_cat_norm = norm_cat(new_ents.get("category"))
    prev_cat_norm = norm_cat(prev_ents.get("category"))

    # Если категория сменилась — начинаем с чистого листа, но переносим общие фильтры
    if new_cat_norm and prev_cat_norm and new_cat_norm != prev_cat_norm:
        merged_ents = {"category": new_ents["category"]}
        for k, v in prev_ents.items():
            if k not in new_ents and k != "category":
                merged_ents[k] = v
    else:
        merged_ents = prev_ents.copy()
        
    # Наложение новых сущностей на старые
    for k, v in new_ents.items():
        if v is not None and str(v).lower() not in {"none", "null", "any", ""}:
            merged_ents[k] = v

    merged["entities"] = merged_ents
    merged["action"] = new_intent.get("action", prev_intent.get("action", "SEARCH"))
    
    # Объединение исключенных ID для предотвращения повторов
    new_excl = new_intent.get("excluded_ids", [])
    if not isinstance(new_excl, list): new_excl = []
    
    prev_excl = prev_intent.get("excluded_ids", [])
    if not isinstance(prev_excl, list): prev_excl = []
    
    merged["excluded_ids"] = list(set(str(i) for i in (new_excl + prev_excl)))

    return merged

def deduplicate_products(products: list, top_k: int = 5) -> list:
    """
    Удаляет дубликаты товаров на основе нормализованного названия.
    """
    if not products:
        return []
    seen = set()
    result = []
    for hit in products:
        if not isinstance(hit, dict):
            continue
        p = hit.get("product") or hit.get("data") or hit
        if not isinstance(p, dict):
            continue
            
        p_title = str(p.get("name") or p.get("title", ""))
        # Создаем ключ из букв и цифр для игнорирования спецсимволов и пробелов
        name_key = re.sub(r'[^\w]', '', p_title.lower())
        
        if name_key not in seen:
            seen.add(name_key)
            result.append(hit)
        
        if len(result) >= top_k:
            break
    return result