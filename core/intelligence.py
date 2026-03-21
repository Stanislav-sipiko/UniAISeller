# /root/ukrsell_v4/core/intelligence.py v7.7.4

import numpy as np
import re
import json
import os
import time
from typing import List, Dict, Any, Optional, Union
from core.logger import logger, log_event

def get_stem(text: str) -> str:
    """
    Выполняет нормализацию текста и стемминг (удаление окончаний) 
    для улучшения качества сопоставления сущностей.
    Поддерживает украинский, русский и английский языки.
    """
    if not text: 
        return ""
    
    word = str(text).lower().strip()
    # Удаление пунктуации
    word = re.sub(r'[^\w\s]', '', word)
    
    # Регулярное выражение для типичных окончаний (UA/RU/EN)
    suffixes = r'(атор|ами|іця|иця|ій|ий|ый|ое|ая|ие|ів|ам|ов|ою|є|ї|ої|их|ых|им|ым|s|es|ed|ing|ка|ок|ик|є)$'
    stemmed = re.sub(suffixes, '', word)
    # Guard: не обрезаем если результат слишком короткий
    word = stemmed if len(stemmed) >= 3 else word
    
    return word if len(word) >= 3 else word

def safe_extract_json(text: str) -> dict:
    """
    Безопасно извлекает JSON из строки, обрабатывая артефакты генерации LLM,
    Markdown-теги и незакрытые скобки.
    """
    if not text:
        return {"action": "SEARCH", "entities": {}}

    # [ZERO OMISSION LOG] Трассировка входящего текста для парсинга
    logger.debug(f"[INTEL_TRACE] safe_extract_json input: {text[:250]}...")

    # 1. Очистка от служебных тегов размышлений (CoT)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    # 2. Очистка от Markdown-оберток кода
    text = re.sub(r"```json\s*", "", text)
    text = text.replace("```", "").strip()

    result = {"action": "SEARCH", "entities": {}}
    
    try:
        # Поиск JSON-структуры внутри текста (первое вхождение { и последнее })
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            json_str = match.group(1).replace('\t', ' ')
            # Нормализация типографских кавычек и невидимых символов
            json_str = json_str.replace('\u00ab', '"').replace('\u00bb', '"')
            json_str = json_str.replace('\u201c', '"').replace('\u201d', '"')
            json_str = json_str.replace('\u2018', "'").replace('\u2019', "'")
            json_str = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', json_str)
            result = json.loads(json_str)
        else:
            result = json.loads(text)
    except Exception as e:
        # 3. Попытка исправить незакрытый JSON (обрыв генерации)
        if text.startswith("{") and not text.endswith("}"):
            try:
                fixed_text = text
                open_braces = fixed_text.count('{')
                close_braces = fixed_text.count('}')
                if open_braces > close_braces:
                    fixed_text += '}' * (open_braces - close_braces)
                result = json.loads(fixed_text)
                logger.info("[INTEL] JSON structure auto-repaired.")
            except:
                logger.error(f"[INTEL] JSON Repair failed: {text[:100]}")
        else:
            logger.warning(f"[INTEL] JSON Extraction failed: {e}")
            log_event("JSON_PARSE_ERROR", {"error": str(e), "text": text[:200]}, level="warning")

    # 4. Нормализация ключей в секции entities
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
    Отсеивает товары с низким показателем семантической релевантности (score).
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
            
    logger.info(f"🛡️ [SEMANTIC_GUARD] In: {len(products)} | Out: {len(filtered)} | Threshold: {threshold}")
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
    
    SERVICE_KEYS = {
        "price_limit", "action", "target", "properties",
        "excluded_ids", "taxonomy_hint", "category",
        "temperature", "language"
    }
    GARBAGE_VALUES = {"none", "null", "any", "unknown", "", "все", "любой", "товари"}

    active_filters = {}
    brand_ignore_list = []

    if store_hints and "brand_ignore" in store_hints:
        brand_ignore_list = [str(b).lower() for b in store_hints["brand_ignore"]]

    for k, v in entities.items():
        if k in SERVICE_KEYS or v is None:
            continue

        val_str = str(v).lower().strip()

        if k == "brand" and any(b_ign in val_str for b_ign in brand_ignore_list):
            logger.debug(f"[INTEL] Brand Filter suppressed for: {val_str}")
            continue

        if val_str and val_str not in GARBAGE_VALUES:
            active_filters[k] = val_str

    # Разворачиваем properties в active_filters через intent_mapping магазина
    props = entities.get("properties") or {}
    if isinstance(props, dict) and intent_mapping:
        for prop_key, prop_val in props.items():
            if prop_val is None:
                continue
            val_str = str(prop_val).lower().strip()
            if not val_str or val_str in GARBAGE_VALUES:
                continue
            # Ищем prop_key в intent_mapping — если есть, добавляем как фильтр
            if prop_key in intent_mapping or prop_key.lower() in {k.lower() for k in intent_mapping}:
                active_filters[prop_key] = val_str
                logger.debug(f"[INTEL] Property expanded to filter: {prop_key}={val_str}")

    if not products:
        return []

    logger.info(f"⚙️ [ENTITY_FILTER] Active filters applied: {active_filters}")

    # Маппинг берётся только из intent_mapping магазина — ядро не знает деталей
    mapping = intent_mapping or {}

    filtered = []
    raw_excluded = intent.get("excluded_ids")
    excluded_ids = set(str(i) for i in raw_excluded) if isinstance(raw_excluded, list) else set()
    price_limit = entities.get("price_limit")

    for hit in products:
        if not isinstance(hit, dict): continue
            
        p = hit.get("product") or hit.get("data") or hit
        if not isinstance(p, dict): continue
            
        p_id = str(p.get("product_id") or p.get("id", ""))
        if p_id in excluded_ids: continue

        if price_limit:
            try:
                raw_price = p.get("price", 0)
                # Очищаем строку от валюты и пробелов
                clean_price = re.sub(r'[^\d.,]', '', str(raw_price)).replace(',', '.')
                p_price = float(clean_price) if clean_price else 0.0
                if p_price > float(price_limit): continue
            except (ValueError, TypeError):
                pass

        p_attrs = p.get("attributes") or {}
        p_title = str(p.get("name") or p.get("title", "")).lower()
        
        match_count = 0
        total_filters = len(active_filters)

        if total_filters == 0:
            filtered.append(hit)
            continue

        for ent_key, ent_val in active_filters.items():
            ent_stem = get_stem(ent_val)
            target_fields = mapping.get(ent_key, [ent_key])
            if isinstance(target_fields, str): target_fields = [target_fields]
            
            field_match = False
            for field in target_fields:
                val_in_prod = p.get(field) or p_attrs.get(field)
                if val_in_prod:
                    vals = [str(x).lower() for x in (val_in_prod if isinstance(val_in_prod, list) else [val_in_prod])]
                    if any(ent_val in v or ent_stem in get_stem(v) for v in vals):
                        field_match = True
                        break
            
            if not field_match and (ent_val in p_title or ent_stem in get_stem(p_title)):
                field_match = True
            
            if field_match:
                match_count += 1

        if match_count == total_filters:
            filtered.append(hit)
        elif total_filters >= 3 and match_count >= (total_filters - 1):
            filtered.append(hit)

    if not filtered and products:
        first_hit = products[0]
        if isinstance(first_hit, dict):
            top_hit_score = first_hit.get("final_score") or first_hit.get("score", 0)
            if top_hit_score > 0.85:
                logger.info("🆘 [ENTITY_FILTER] Emergency fallback: preserving high-score results despite filters.")
                return products[:2]

    logger.info(f"📊 [ENTITY_FILTER] Final count: {len(filtered)}")
    return filtered

def merge_followup(prev_intent: dict, new_intent: dict, category_map: dict = None) -> dict:
    """
    Объединяет текущий интент с контекстом предыдущего запроса.
    Реализует логику сохранения фильтров при уточнении или сброс при смене категории.
    """
    if not isinstance(prev_intent, dict): return new_intent
    if not isinstance(new_intent, dict): return prev_intent

    merged = prev_intent.copy()
    new_ents = new_intent.get("entities") or {}
    prev_ents = prev_intent.get("entities") or {}

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

    if new_cat_norm and prev_cat_norm and new_cat_norm != prev_cat_norm:
        logger.info(f"🔄 [MERGE] Resetting filters: {prev_cat_norm} -> {new_cat_norm}")
        merged_ents = {"category": new_ents["category"]}
        for k, v in prev_ents.items():
            if k == "brand" and k not in new_ents:
                merged_ents[k] = v
    else:
        merged_ents = prev_ents.copy()
        
    for k, v in new_ents.items():
        if v is not None and str(v).lower() not in {"none", "null", "any", ""}:
            merged_ents[k] = v

    merged["entities"] = merged_ents
    prev_action = prev_intent.get("action", "SEARCH")
    new_action  = new_intent.get("action", "SEARCH")
    new_reason  = str(new_intent.get("reason", "")).lower()

    # Unsupported keywords в reason — явный сигнал что тема вне ассортимента.
    # Не переключаем на SEARCH и не наследуем старую category.
    _UNSUPPORTED_SIGNALS = {
        "vitamin", "вітамін", "витамин", "ліки", "лекарств", "корм", "їжа",
        "вакцин", "таблетк", "клітк", "вольєр", "наповнювач", "туалет",
        "ласощ", "смаколик", "not a product", "unsupported",
    }
    is_unsupported = any(sig in new_reason for sig in _UNSUPPORTED_SIGNALS)

    if is_unsupported:
        # Unsupported тема — возвращаем CONSULT как есть, без наследования category
        merged["action"]   = "CONSULT"
        merged["entities"] = new_ents  # пустые entities, не тянем лежанки
        logger.info(f"[MERGE] Unsupported topic detected in reason — keeping CONSULT, clearing category.")
    elif (prev_action == "SEARCH"
            and new_action == "CONSULT"
            and not new_ents.get("category")):
        # Ответ на уточняющий вопрос — сохраняем SEARCH с prev category
        merged["action"] = "SEARCH"
        logger.info("[MERGE] action: CONSULT→SEARCH (clarification answer, no new category)")
    else:
        merged["action"] = new_action
    
    new_excl  = new_intent.get("excluded_ids", [])
    prev_excl = prev_intent.get("excluded_ids", [])
    # dict.fromkeys сохраняет порядок + дедупликация; [-20:] — ротация по времени
    merged["excluded_ids"] = list(dict.fromkeys(
        str(i) for i in (
            (new_excl  if isinstance(new_excl,  list) else []) +
            (prev_excl if isinstance(prev_excl, list) else [])
        )
    ))[-20:]

    return merged

def deduplicate_products(products: list, top_k: int = 5) -> list:
    """
    Удаляет дубликаты товаров на основе нормализованного названия.
    """
    if not products: return []
    seen = set()
    result = []
    for hit in products:
        if not isinstance(hit, dict): continue
        p = hit.get("product") or hit.get("data") or hit
        if not isinstance(p, dict): continue
            
        p_title = str(p.get("name") or p.get("title", ""))
        name_key = re.sub(r'[^\w]', '', p_title.lower())
        
        if name_key not in seen:
            seen.add(name_key)
            result.append(hit)
        
        if len(result) >= top_k: break
            
    logger.debug(f"♻️ [DEDUPLICATION] Reduced from {len(products)} to {len(result)}")
    return result

def get_version():
    return "7.7.4"