"""
Confidence Engine v1.8.2

Архитектурная роль: 
Принимает решение о режиме ответа (SHOW_PRODUCTS, ASK_CLARIFICATION, NO_RESULTS).
Интегрирован с RetrievalEngine v7.5.x и Intelligence v7.6.x.

Changelog:
    v1.8.0  [ФИКС КАТЕГОРИЙНОГО ПЕССИМИЗМА] - стемминг Title Match.
    v1.8.1  Введен "Attribute Anchor": приоритет брендам (Tatra и др.).
    v1.8.2  [RESTORATION] Возвращена оригинальная структура return (all_products, top_categories).
            Исправлены ключи breakdown для полной совместимости с логированием v4.
"""

from typing import Dict, Any, List, Optional, Tuple
import re
from core.logger import logger
from core.intelligence import get_stem


# ─── ВЕСОВЫЕ КОЭФФИЦИЕНТЫ ────────────────────────────────────────────────────
W_SIM      = 0.35   # Семантическое сходство
W_CAT      = 0.15   # Категорийное соответствие
W_ATTR     = 0.15   # Атрибутный match (brand/animal/price)
W_RESULTS  = 0.10   # Количество результатов
W_CLARITY  = 0.05   # Ясность запроса
W_DIRECT   = 0.20   # Прямое совпадение ключевых слов (Title Match)

# Пороги принятия решений
THRESHOLD_SHOW     = 0.52   
THRESHOLD_CLARIFY  = 0.30   

SPEC_BONUS         = 0.07   # Бонус, если интент совпадает с нишей магазина


# ─── ВСПОМОГАТЕЛЬНАЯ ЛОГИКА ──────────────────────────────────────────────────

def resolve_category_en(raw: str, category_map: Optional[Dict]) -> List[str]:
    """
    Переводит UA/RU/EN ключ категории в список EN-значений из category_map.
    """
    if not raw or not category_map:
        return []
    key = str(raw).lower().strip()
    
    # 1. Прямой поиск
    result = category_map.get(key)
    if result:
        return [v.lower() for v in result] if isinstance(result, list) else [str(result).lower()]
    
    # 2. Поиск по вхождению
    for map_key, map_val in category_map.items():
        if map_key.startswith("_"): continue
        if key in map_key or map_key in key:
            return [v.lower() for v in map_val] if isinstance(map_val, list) else [str(map_val).lower()]
    return []


def _extract_sim_score(products: List[Dict]) -> float:
    """Извлекает семантический score. Поддержка final_score и raw score."""
    if not products:
        return 0.0
    scores = []
    for res in products:
        fs = res.get("final_score")
        if fs is not None:
            scores.append(float(fs))
            continue
        raw = float(res.get("score", 0.0))
        # Эвристическая нормализация для сырых векторов, если final_score нет
        normalized = max(0.0, (raw - 0.70) / 0.30)
        scores.append(normalized)
    return round(max(scores), 4) if scores else 0.0


def _direct_match_score(user_query: str, products: List[Dict]) -> float:
    """
    Проверяет вхождение основ слов запроса в названия товаров.
    Использует get_stem для борьбы с окончаниями.
    """
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
        
        # Считаем, сколько слов из запроса "приземлилось" в заголовке
        match_count = sum(1 for q_stem in query_stems if any(q_stem in t_stem for t_stem in title_stems))
        if match_count >= 1:
            hits += 1

    if checked == 0: return 0.0
    return round(hits / checked, 4)


def _attr_match_score(entities: Dict, products: List[Dict]) -> float:
    """Проверяет совпадение brand, animal, price_limit."""
    ent = entities or {}
    # Поддержка разных структур вложенности
    props = ent.get("properties", {}) or {}

    brand = str(ent.get("brand") or props.get("brand") or "").lower().strip()
    animal_raw = ent.get("animal") or props.get("animal") or ent.get("entities", {}).get("animal") or ""
    animal = str(animal_raw).lower().strip() if animal_raw else ""
    price_limit = ent.get("price_limit") or props.get("price_limit")

    if not (brand or animal or price_limit):
        return 0.0
    if not products:
        return 0.0

    matched = 0
    test_set = products[:3]
    for res in test_set:
        prod = res.get("product") or res.get("data") or res
        if not prod: continue

        item_match = False
        # Brand match
        if brand:
            prod_brand = str(prod.get("brand") or "").lower()
            if brand in prod_brand or prod_brand in brand:
                item_match = True

        # Animal match
        if not item_match and animal:
            prod_animals = prod.get("animal") or []
            if isinstance(prod_animals, str): prod_animals = [prod_animals]
            if any(animal in str(pa).lower() for pa in prod_animals):
                item_match = True

        # Price match
        if not item_match and price_limit:
            try:
                if float(prod.get("price", 0)) <= float(price_limit):
                    item_match = True
            except: pass

        if item_match: matched += 1

    return round(matched / len(test_set), 4) if test_set else 0.0


def _category_score(intent_cat_en: List[str], top_categories: List[Tuple]) -> Tuple[float, float]:
    """Возвращает (cat_score, category_prior)."""
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
    """Итоговая формула v1.8.2."""
    base = (
        W_SIM      * sim_score +
        W_CAT      * cat_score +
        W_ATTR     * attr_score +
        W_RESULTS  * result_quality +
        W_CLARITY  * clarity_score +
        W_DIRECT   * direct_match
    )
    
    # Prior Bonus
    prior_bonus = max(0.05 if cat_score > 0 else 0, min(category_prior * 0.4, 0.25))
    
    total = base + prior_bonus + specialization_bonus
    
    # Attribute Anchor: Если есть бренд и прямой матч в заголовке — защищаем от NO_RESULTS
    if brand_present and attr_score > 0.4 and direct_match > 0.3:
        total = max(total, THRESHOLD_CLARIFY + 0.05)

    return round(min(total, 1.0), 4)


def decide_mode(confidence: float, result_count: int, sim_score: float) -> str:
    if result_count == 0:
        return "NO_RESULTS"
    # Для прямой выдачи нужно хорошее семантическое сходство или высокий общий балл
    if (confidence >= THRESHOLD_SHOW and sim_score >= 0.35) or confidence > 0.75:
        return "SHOW_PRODUCTS"
    return "ASK_CLARIFICATION"


def evaluate(
    search_result: Dict[str, Any],
    intent: Dict[str, Any],
    user_query: str,
    profile: Optional[Dict] = None,
    category_map: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Точка входа. Возвращает строго фиксированную структуру для Kernel v7.8.x.
    """
    products = search_result.get("products", [])
    count = len(products)
    
    # Извлекаем сущности
    ent = intent or {}
    entities = ent.get("entities", {}) if "entities" in ent else ent

    # 1. Топ категорий из профиля
    profile = profile or {}
    cat_dist = profile.get("category_distribution", {})
    top_categories = sorted(
        [(k, v) for k, v in cat_dist.items() if isinstance(v, (int, float))],
        key=lambda x: x[1], reverse=True
    )[:5]

    # 2. Резолвинг категории
    intent_cat_raw = str(entities.get("category") or ent.get("category") or "").strip()
    intent_cat_en_list = resolve_category_en(intent_cat_raw, category_map)
    intent_cat_en_str = intent_cat_en_list[0] if intent_cat_en_list else ""

    # 3. Расчет компонент
    sim_score    = _extract_sim_score(products)
    direct_match = _direct_match_score(user_query, products)
    attr_score   = _attr_match_score(entities, products)
    result_qual  = 1.0 if count >= 3 else (0.7 if count > 0 else 0.0)
    clarity      = 1.0 if 2 < len(user_query.split()) < 10 else 0.6
    cat_score, category_prior = _category_score(intent_cat_en_list, top_categories)

    # 4. Бонус специализации
    main_cat = str(profile.get("main_category", "")).lower()
    spec_bonus = SPEC_BONUS if main_cat and intent_cat_en_str and main_cat in intent_cat_en_str else 0.0

    # 5. Финальный скор
    confidence = compute_confidence(
        sim_score            = sim_score,
        cat_score            = cat_score,
        attr_score           = attr_score,
        result_quality       = result_qual,
        clarity_score        = clarity,
        direct_match         = direct_match,
        category_prior       = category_prior,
        specialization_bonus = spec_bonus,
        brand_present        = bool(entities.get("brand"))
    )

    mode = decide_mode(confidence, count, sim_score)

    logger.info(f"[CONFIDENCE v1.8.2] mode={mode} score={confidence} sim={sim_score} q={user_query!r}")

    # ВОЗВРАЩАЕМ СТРОГО ТВОЮ СТРУКТУРУ
    return {
        "mode":           mode,
        "confidence":      confidence,
        "sim_score":       sim_score,
        "top_categories": top_categories,
        "all_products":    products, # Важно: Kernel ждет именно этот ключ
        "breakdown": {
            "sim":            sim_score,
            "direct_match":   direct_match,
            "cat":            cat_score,
            "attr":           attr_score,
            "result_quality": result_qual,
            "clarity":        clarity,
            "category_prior": category_prior,
            "spec_bonus":     spec_bonus,
            "intent_cat_en":  intent_cat_en_str
        }
    }