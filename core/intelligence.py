# /root/ukrsell_v4/core/intelligence.py v7.6.2
"""
UkrSell Intelligence Module v7.6.2
- Entity Extraction, Normalization & Smart Filtering.
- Optimized for Llama-3.1-8b simplified intent outputs.
- Resilient to category synonym shifts.
"""

import numpy as np
import re
import json
from typing import List, Dict, Any, Optional, Union
from core.logger import logger, log_event

def get_stem(text: str) -> str:
    """
    Извлекает основу слова для мягкого сравнения. 
    v7.6.2: Оптимизировано для UA/RU морфологии.
    """
    if not text: 
        return ""
    
    # Приведение к нижнему регистру и очистка
    word = str(text).lower().strip()
    
    # Удаление спецсимволов
    word = re.sub(r'[^\w\s]', '', word)
    
    # Расширенный список окончаний (важен порядок: от длинных к коротким)
    # Удаляем типичные окончания прилагательных и существительных UA/RU
    suffixes = r'(атор|ами|іця|иця|ов|ам|ів|ий|ый|ое|ая|ие|ы|і|а|я|у|е|ом|різ|лов|чик|s|es|ed|ing|ка|ок|ик|ою|є|ї|ої|их|ых|им|ым)$'
    word = re.sub(suffixes, '', word)
    
    return word if len(word) >= 3 else word

def safe_extract_json(text: str) -> dict:
    """
    Извлекает JSON из мусорного текста LLM.
    v7.6.2: Добавлена нормализация ключей (llama fix).
    """
    if not text:
        return {"action": "SEARCH", "entities": {}}
        
    text = text.strip()
    result = {"action": "SEARCH", "entities": {}}
    
    try:
        # 1. Поиск блока {...}
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            json_str = match.group(1).replace('\t', ' ')
            result = json.loads(json_str)
        else:
            # Попытка прямого парсинга
            result = json.loads(text)
    except Exception as e:
        # 2. Попытка закрыть оборванный JSON
        if text.startswith("{") and not text.endswith("}"):
            try:
                result = json.loads(text + "}")
            except:
                pass
        else:
            logger.warning(f"[INTEL] JSON Extraction failed: {e}")
            log_event("JSON_PARSE_ERROR", {"error": str(e), "text": text[:200]}, level="warning")

    # 3. Латентная нормализация ключей (Llama-3.1-8b Fix)
    # Если LLM прислала "brand_name" вместо "brand" или "type" вместо "subtype"
    if "entities" in result:
        ents = result["entities"]
        mapping_fixes = {
            "brand_name": "brand",
            "product_type": "subtype",
            "type": "subtype",
            "color_name": "color",
            "pet": "animal"
        }
        for wrong_key, right_key in list(mapping_fixes.items()):
            if wrong_key in ents and right_key not in ents:
                ents[right_key] = ents.pop(wrong_key)
                
    return result

def semantic_guard(products: list, threshold: float = 0.30) -> list:
    """
    Отсеивает низкокачественные совпадения.
    """
    if not products:
        return []
    
    filtered = []
    for res in products:
        # Используем final_score (выше = лучше)
        score = res.get("final_score", 0)
        if score >= threshold:
            filtered.append(res)
            
    return filtered

def entity_filter(products: list, intent: dict, intent_mapping: dict = None, category_map: dict = None) -> list:
    """
    Универсальный фильтр товаров. 
    v7.6.2: Добавлен Double-Pass (Attribute + Title fallback) для борьбы с упрощением LLM.
    """
    entities = intent.get("entities", intent) 
    
    SERVICE_KEYS = {"price_limit", "action", "target", "properties", "excluded_ids", "taxonomy_hint", "category"}
    GARBAGE_VALUES = {"none", "null", "any", "unknown", "", "все", "любой", "товари"}

    # 1. Сбор активных фильтров
    active_filters = {}
    for k, v in entities.items():
        if k in SERVICE_KEYS or v is None:
            continue
        val_str = str(v).lower().strip()
        if val_str and val_str not in GARBAGE_VALUES:
            active_filters[k] = val_str

    if not products or not active_filters:
        return products

    # 2. Подготовка маппинга полей
    mapping = intent_mapping or {
        "brand": ["brand", "manufacturer", "vendor"],
        "color": ["color", "colour", "color_ref"],
        "animal": ["animal", "pet_type", "for_animals"],
        "subtype": ["subtype", "product_type"]
    }

    filtered = []
    excluded_ids = set(str(i) for i in (intent.get("excluded_ids") or []))

    for hit in products:
        p = hit.get("product") if isinstance(hit, dict) and "product" in hit else hit
        p_id = str(p.get("product_id") or p.get("id", ""))
        
        if p_id in excluded_ids:
            continue

        p_attrs = p.get("attributes") or {}
        p_title = str(p.get("title", "")).lower()
        
        match_count = 0
        total_filters = len(active_filters)

        for ent_key, ent_val in active_filters.items():
            ent_stem = get_stem(ent_val)
            target_fields = mapping.get(ent_key, [ent_key])
            if isinstance(target_fields, str): target_fields = [target_fields]
            
            field_match = False
            # А) Проверка в атрибутах
            for field in target_fields:
                val_in_prod = p.get(field) or p_attrs.get(field)
                if val_in_prod:
                    vals = [str(x).lower() for x in (val_in_prod if isinstance(val_in_prod, list) else [val_in_prod])]
                    if any(ent_val in v or ent_stem in get_stem(v) for v in vals):
                        field_match = True
                        break
            
            # Б) Fallback: Проверка в заголовке
            if not field_match:
                if ent_val in p_title or ent_stem in get_stem(p_title):
                    field_match = True
            
            if field_match:
                match_count += 1

        # Решение о соответствии: позволяем 1 промах, если фильтров много (Soft Matching)
        if match_count == total_filters:
            filtered.append(hit)
        elif total_filters >= 3 and match_count >= (total_filters - 1):
            filtered.append(hit)

    # 3. Emergency Fallback: если фильтры убили всё, возвращаем топ по вектору (при высоком скоре)
    if not filtered and products:
        top_hit_score = products[0].get("final_score", 0)
        if top_hit_score > 0.8:
            logger.info(f"[INTEL] Entity filters too strict for high-score hit ({top_hit_score}). Using vector top.")
            return products[:2]

    return filtered

def merge_followup(prev_intent: dict, new_intent: dict, category_map: dict = None) -> dict:
    """
    Склеивает контекст.
    v7.6.2: Добавлена проверка синонимов категорий через category_map.
    """
    merged = prev_intent.copy()
    new_ents = new_intent.get("entities", {})
    prev_ents = prev_intent.get("entities", {})
    
    # Нормализация категорий для сравнения
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

    # Если категория действительно сменилась (а не просто синоним)
    if new_cat_norm and prev_cat_norm and new_cat_norm != prev_cat_norm:
        logger.debug(f"[INTEL] Context Reset: Category changed to {new_ents.get('category')}")
        merged_ents = {"category": new_ents["category"]}
        if "animal" in prev_ents: merged_ents["animal"] = prev_ents["animal"]
    else:
        merged_ents = prev_ents.copy()
        
    # Наложение новых сущностей
    for k, v in new_ents.items():
        if v is not None and str(v).lower() not in {"none", "null", "any", ""}:
            merged_ents[k] = v

    merged["entities"] = merged_ents
    merged["action"] = new_intent.get("action", "SEARCH")
    
    # Слияние черного списка ID
    new_excl = new_intent.get("excluded_ids", [])
    prev_excl = prev_intent.get("excluded_ids", [])
    merged["excluded_ids"] = list(set(str(i) for i in (new_excl + prev_excl)))

    return merged

def deduplicate_products(products: list, top_k: int = 5) -> list:
    """
    Убирает дубликаты по нормализованному заголовку.
    """
    if not products:
        return []
    seen = set()
    result = []
    for hit in products:
        p = hit.get("product") if isinstance(hit, dict) and "product" in hit else hit
        name = re.sub(r'[^\w]', '', str(p.get("title", "")).lower())
        if name not in seen:
            seen.add(name)
            result.append(hit)
        if len(result) >= top_k:
            break
    return result