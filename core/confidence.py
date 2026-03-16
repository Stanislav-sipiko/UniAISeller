# -*- coding: utf-8 -*-
# /root/ukrsell_v4/core/confidence.py v1.8.7
from typing import Dict, Any, List, Optional, Tuple
import re
from core.logger import logger
from core.intelligence import get_stem

# ─── ВЕСОВЫЕ КОЭФФИЦИЕНТЫ (Строгий режим v1.8.7) ────────────────────────────
W_SIM      = 0.15   # Минимизируем влияние "похожих по вектору" товаров
W_CAT      = 0.20   
W_ATTR     = 0.10   
W_RESULTS  = 0.05   
W_CLARITY  = 0.05   
W_DIRECT   = 0.45   # ГЛАВНЫЙ ВЕС: Прямое попадание слов запроса в заголовок

# Пороги принятия решений
THRESHOLD_SHOW     = 0.60   # Порог показа
THRESHOLD_CLARIFY  = 0.40   # Порог уточнения

SPEC_BONUS         = 0.10   

# ─── ВСПОМОГАТЕЛЬНАЯ ЛОГИКА ──────────────────────────────────────────────────

def resolve_category_en(raw: str, category_map: Optional[Dict]) -> List[str]:
    """Переводит UA/RU/EN ключ категории в список EN-значений."""
    if not raw or not category_map:
        return []
    key = str(raw).lower().strip()
    
    result = category_map.get(key)
    if result:
        return [v.lower() for v in result] if isinstance(result, list) else [str(result).lower()]
    
    for map_key, map_val in category_map.items():
        if map_key.startswith("_"): continue
        if key in map_key or map_key in key:
            return [v.lower() for v in map_val] if isinstance(map_val, list) else [str(map_val).lower()]
    return []

def _extract_sim_score(products: List[Dict]) -> float:
    """Извлекает семантический score."""
    if not products:
        return 0.0
    scores = []
    for res in products:
        fs = res.get("final_score")
        if fs is not None:
            scores.append(float(fs))
            continue
        raw = float(res.get("score", 0.0))
        normalized = max(0.0, (raw - 0.70) / 0.30)
        scores.append(normalized)
    return round(max(scores), 4) if scores else 0.0

def _direct_match_score(user_query: str, products: List[Dict], is_substitute: bool = False) -> float:
    """Проверяет вхождение основ слов запроса в названия товаров (get_stem)."""
    if not products or not user_query:
        return 0.0

    stopwords = {
        "хочу", "треба", "потрібен", "потрібна", "потрібно", "купити", "купить", 
        "знайти", "знайди", "шукаю", "ищу", "для", "якийсь", "якусь", "щось"
    }
    
    query_stems = [
        get_stem(w) for w in user_query.lower().split() 
        if len(w) > 2 and w.lower() not in stopwords
    ]
    
    if not query_stems:
        return 0.0

    hits = 0
    checked = 0
    for res in products[:3]:
        prod = res.get("product") or res.get("data") or res
        title = str(prod.get("title") or prod.get("name") or "").lower()
        if not title: continue
        
        checked += 1
        title_stems = [get_stem(w) for w in title.split()]
        
        match_count = sum(1 for q_stem in query_stems if any(q_stem in t_stem for t_stem in title_stems))
        
        if match_count >= 1:
            hits += 1
        elif is_substitute:
            hits += 0.3 

    if checked == 0: return 0.0
    return round(hits / checked, 4)

def _attr_match_score(entities: Dict, products: List[Dict], intent_mapping: Dict = None) -> float:
    """
    Универсальный матчинг атрибутов (Agnostic v1.8.7).
    Проверяет совпадение всех извлеченных сущностей с данными товара.
    """
    if not entities or not products:
        return 0.0

    # Служебные ключи, которые не участвуют в проверке свойств товара
    SERVICE_KEYS = {"price_limit", "action", "target", "properties", "excluded_ids", "taxonomy_hint", "category", "temperature", "language"}
    GARBAGE_VALUES = {"none", "null", "any", "unknown", "", "все", "любой"}

    # Сбор активных критериев для проверки
    active_criteria = {}
    price_limit = entities.get("price_limit") or entities.get("properties", {}).get("price_limit")

    for k, v in entities.items():
        if k in SERVICE_KEYS or v is None:
            continue
        val_str = str(v).lower().strip()
        if val_str and val_str not in GARBAGE_VALUES:
            active_criteria[k] = val_str

    if not active_criteria and not price_limit:
        return 0.0

    # Динамический маппинг полей
    mapping = intent_mapping or {
        "brand": ["brand", "manufacturer", "vendor"],
        "color": ["color", "colour", "color_ref"],
        "subtype": ["subtype", "product_type"]
    }

    matched_count = 0
    test_set = products[:3]
    
    for res in test_set:
        prod = res.get("product") or res.get("data") or res
        if not isinstance(prod, dict): continue
        
        p_attrs = prod.get("attributes") or {}
        item_score = 0
        total_checks = len(active_criteria) + (1 if price_limit else 0)

        # 1. Проверка текстовых атрибутов
        for ent_key, ent_val in active_criteria.items():
            ent_stem = get_stem(ent_val)
            target_fields = mapping.get(ent_key, [ent_key])
            
            field_match = False
            for field in target_fields:
                val_in_prod = prod.get(field) or p_attrs.get(field)
                if val_in_prod:
                    vals = [str(x).lower() for x in (val_in_prod if isinstance(val_in_prod, list) else [val_in_prod])]
                    if any(ent_val in v or ent_stem in get_stem(v) for v in vals):
                        field_match = True
                        break
            if field_match:
                item_score += 1

        # 2. Проверка цены
        if price_limit:
            try:
                if float(prod.get("price", 0)) <= float(price_limit):
                    item_score += 1
            except:
                pass
        
        if total_checks > 0 and (item_score / total_checks) >= 0.5:
            matched_count += 1

    return round(matched_count / len(test_set), 4) if test_set else 0.0

def _category_score(intent_cat_en: List[str], top_categories: List[Tuple]) -> Tuple[float, float]:
    if not intent_cat_en or not top_categories:
        return (0.3 if intent_cat_en else 0.0), 0.0

    best_prior = 0.0
    for cat_name, prob in top_categories:
        cat_lower = cat_name.lower()
        for en in intent_cat_en:
            if en in cat_lower or cat_lower in en:
                best_prior = max(best_prior, float(prob))

    cat_score = 0.5 if intent_cat_en else 0.0 
    return cat_score, best_prior

# ─── ОСНОВНЫЕ ФУНКЦИИ ────────────────────────────────────────────────────────

def compute_confidence(
    sim_score: float,
    cat_score: float,
    attr_score: float,
    result_quality: float,
    clarity_score: float,
    direct_match: float,
    category_prior: float,
    specialization_bonus: float,
    brand_present: bool = False
) -> float:
    """Итоговая формула v1.8.7."""
    base = (
        W_SIM      * sim_score +
        W_CAT      * cat_score +
        W_ATTR     * attr_score +
        W_RESULTS  * result_quality +
        W_CLARITY  * clarity_score +
        W_DIRECT   * direct_match
    )
    
    prior_bonus = max(0.05 if cat_score > 0 else 0, min(category_prior * 0.4, 0.25))
    total = base + prior_bonus + specialization_bonus
    
    if brand_present and attr_score > 0.6 and direct_match > 0.4:
        total = max(total, THRESHOLD_CLARIFY + 0.1)

    return round(min(total, 1.0), 4)

def decide_mode(confidence: float, result_count: int, sim_score: float, direct_match: float) -> str:
    if result_count == 0:
        return "NO_RESULTS"
    
    is_solid_text_match = direct_match >= 0.60
    is_semantic_match = direct_match >= 0.33 and sim_score >= 0.40
    
    if confidence >= THRESHOLD_SHOW and (is_semantic_match or is_solid_text_match) or confidence > 0.85:
        return "SHOW_PRODUCTS"
    
    if confidence >= THRESHOLD_CLARIFY:
        return "ASK_CLARIFICATION"
        
    return "NO_RESULTS"

def evaluate(
    search_result: Dict[str, Any],
    intent: Dict[str, Any],
    user_query: str,
    profile: Optional[Dict] = None,
    category_map: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Точка входа v1.8.7 (Agnostic Core)."""
    products = search_result.get("products", [])
    count = len(products)
    
    ent = intent or {}
    entities = ent.get("entities", {}) if "entities" in ent else ent
    intent_cat_raw = str(entities.get("category") or ent.get("category") or "").strip()

    intent_cat_en_list = resolve_category_en(intent_cat_raw, category_map)
    intent_cat_en_str = intent_cat_en_list[0] if intent_cat_en_list else ""
    
    is_substitute = False
    if intent_cat_raw and products:
        main_stem = get_stem(intent_cat_raw)
        if main_stem:
            found_main_cat = False
            for hit in products:
                p = hit.get("product") or hit.get("data") or hit if isinstance(hit, dict) else {}
                p_title = str(p.get("name") or p.get("title", "")).lower()
                if main_stem in p_title:
                    found_main_cat = True
                    break
            
            if not found_main_cat:
                is_substitute = True

    profile = profile or {}
    cat_dist = profile.get("category_distribution", {})
    top_categories = sorted(
        [(k, v) for k, v in cat_dist.items() if isinstance(v, (int, float))],
        key=lambda x: x[1], reverse=True
    )[:5]

    sim_score    = _extract_sim_score(products)
    direct_match = _direct_match_score(user_query, products, is_substitute)
    
    # Использование универсального матчинга без хардкода ниши
    attr_score   = _attr_match_score(entities, products)
    
    result_qual  = 1.0 if count >= 3 else (0.7 if count > 0 else 0.0)
    clarity      = 1.0 if 2 < len(user_query.split()) < 10 else 0.6
    cat_score, category_prior = _category_score(intent_cat_en_list, top_categories)

    main_cat = str(profile.get("main_category", "")).lower()
    spec_bonus = SPEC_BONUS if main_cat and intent_cat_en_str and main_cat in intent_cat_en_str else 0.0

    confidence = compute_confidence(
        sim_score            = sim_score,
        cat_score            = cat_score,
        attr_score            = attr_score,
        result_quality       = result_qual,
        clarity_score        = clarity,
        direct_match         = direct_match,
        category_prior       = category_prior,
        specialization_bonus = spec_bonus,
        brand_present        = bool(entities.get("brand"))
    )

    if intent_cat_raw and direct_match < 0.2 and not is_substitute:
        logger.warning(f"[CONFIDENCE] Killer Gate Active: Category '{intent_cat_raw}' mismatch.")
        confidence *= 0.1

    mode = decide_mode(confidence, count, sim_score, direct_match)

    logger.info(f"[CONFIDENCE v1.8.7] mode={mode} score={confidence} direct={direct_match} q={user_query!r}")

    return {
        "mode":           mode,
        "confidence":      confidence,
        "sim_score":       sim_score,
        "top_categories": top_categories,
        "is_substitute":  is_substitute,
        "all_products":    products,
        "breakdown": {
            "sim":            sim_score,
            "direct_match":   direct_match,
            "cat":            cat_score,
            "attr":           attr_score,
            "result_quality": result_qual,
            "clarity":        clarity,
            "is_substitute":  is_substitute
        }
    }