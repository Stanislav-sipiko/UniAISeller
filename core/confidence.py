# /root/ukrsell_v4/core/confidence.py v1.7.3
"""
Confidence Engine v1.7.0

Changelog:
    v1.4.0  resolve_category_en — UA/RU→EN резолвинг category_prior
    v1.5.0  animal attr_match
    v1.6.0  decide_mode: NO_RESULTS только при result_count==0
    v1.7.0  [КРИТИЧЕСКИЙ ФИКС]
            - sim_score теперь берётся из max(final_score) реранжированных результатов,
              а не из косинусного FAISS score (который всегда 0.8+ и не отражает релевантность).
            - Новый порог SHOW_PRODUCTS: confidence >= 0.52 при sim >= 0.35
            - Добавлен direct_match_bonus: если title содержит ключевые слова запроса → +0.15
            - Добавлен result_quality_score: учитывает кол-во результатов (1..5+)
            - category_map: множественные EN-значения — берём максимальный prior
    v1.7.3  animal translations driven by profile["intent_mapping"] (store-agnostic)
"""

from typing import Dict, Any, List, Optional, Tuple
from core.logger import logger


# ─── Веса компонент ──────────────────────────────────────────────────────────
W_SIM      = 0.40   # семантическое сходство (final_score из реранжира)
W_CAT      = 0.20   # категорийный score
W_ATTR     = 0.12   # атрибутный match (brand/animal/price)
W_RESULTS  = 0.12   # кол-во результатов
W_CLARITY  = 0.08   # ясность запроса (кол-во токенов)
W_DIRECT   = 0.08   # прямое совпадение слов запроса с title

# Пороги режимов
# v1.7.1: THRESHOLD_SHOW снижен с 0.52 до 0.45 — устраняет ложные NO_RESULTS
# для размытых запросов (годування, для кішок и т.п.) при наличии результатов.
THRESHOLD_SHOW     = 0.45   # >= → SHOW_PRODUCTS  (было 0.52)
THRESHOLD_CLARIFY  = 0.30   # >= → ASK_CLARIFICATION, иначе NO_RESULTS (только при result_count==0)

SPEC_BONUS         = 0.07   # бонус если intent_cat == main_category магазина


# ─── Категорийный резолвинг ───────────────────────────────────────────────────

def resolve_category_en(raw: str, category_map: Optional[Dict]) -> List[str]:
    """
    Переводит UA/RU/EN ключ категории в список EN-значений из category_map.
    Возвращает пустой список если не найдено.
    """
    if not raw or not category_map:
        return []
    key = str(raw).lower().strip()
    result = category_map.get(key)
    if result:
        return [v.lower() for v in result] if isinstance(result, list) else [str(result).lower()]
    # Частичное совпадение — ищем ключ как подстроку
    for map_key, map_val in category_map.items():
        if key in map_key or map_key in key:
            return [v.lower() for v in map_val] if isinstance(map_val, list) else [str(map_val).lower()]
    return []


# ─── Вспомогательные функции ─────────────────────────────────────────────────

def _extract_sim_score(products: List[Dict]) -> float:
    """
    Извлекает семантический score из реранжированных результатов.
    Приоритет: final_score (из _semantic_rerank) → score (FAISS cosine).

    ВАЖНО: final_score из RetrievalEngine._semantic_rerank нормирован иначе чем
    FAISS cosine. Нормируем к [0..1] делением на максимально возможный (1.0 для final_score).
    """
    if not products:
        return 0.0
    scores = []
    for res in products:
        # final_score — результат реранжирования (target_bonus + cat_bonus + vec*0.3)
        fs = res.get("final_score")
        if fs is not None:
            scores.append(float(fs))
            continue
        # Fallback: FAISS cosine (обычно 0.8–0.92, слабо дифференцирует)
        # Штрафуем: (score - 0.7) / 0.3 → нормируем в [0..1]
        raw = float(res.get("score", 0.0))
        normalized = max(0.0, (raw - 0.70) / 0.30)
        scores.append(normalized)
    return round(max(scores), 4) if scores else 0.0


def _direct_match_score(user_query: str, products: List[Dict]) -> float:
    """
    Проверяет прямое вхождение значимых слов запроса в title топ-3 товаров.
    Возвращает долю совпавших слов (0..1).
    """
    if not products or not user_query:
        return 0.0
    # Стоп-слова UA/RU
    stopwords = {
        "я", "мені", "мне", "хочу", "треба", "потрібен", "потрібна", "потрібно",
        "купити", "купить", "знайти", "знайди", "шукаю", "ищу", "що", "що",
        "якийсь", "якусь", "щось", "что", "какой", "какую", "для", "для",
        "і", "та", "й", "і", "чи", "або", "или", "the", "a", "an", "is",
        "be", "to", "of", "in", "и", "в", "на", "з", "з", "по", "від",
    }
    query_words = [w.lower() for w in user_query.split() if len(w) > 2 and w.lower() not in stopwords]
    if not query_words:
        return 0.0

    hits = 0
    checked = 0
    for res in products[:3]:
        prod = res.get("product") or res.get("data") or res
        title = str(prod.get("title") or prod.get("name") or "").lower()
        if not title:
            continue
        checked += 1
        for word in query_words:
            if word in title:
                hits += 1
                break  # достаточно одного совпадения на товар

    if checked == 0:
        return 0.0
    return round(hits / checked, 4)


def _attr_match_score(entities: Dict, products: List[Dict],
                      animal_translations: Optional[Dict] = None) -> float:
    """
    Проверяет совпадение извлечённых атрибутов (brand, animal, price_limit) с товарами.
    Возвращает 1.0 если хотя бы один атрибут совпал, 0.0 если нет атрибутов.
    animal_translations — dict переводов из profile["intent_mapping"]["animal"]["translations"].
    """
    ent = entities or {}
    props = ent.get("properties", {}) or {}

    brand      = str(ent.get("brand") or props.get("brand") or "").lower().strip()
    animal_raw = ent.get("animal") or props.get("animal") or ent.get("entities", {}).get("animal") or ""
    animal     = str(animal_raw).lower().strip() if animal_raw else ""
    price_limit = ent.get("price_limit") or props.get("price_limit")

    has_attr = bool(brand or animal or price_limit)
    if not has_attr:
        return 0.0

    if not products:
        return 0.0

    translations = animal_translations or {}
    matched = 0
    for res in products[:3]:
        prod = res.get("product") or res.get("data") or res
        if not prod:
            continue

        # Brand match
        if brand:
            prod_brand = str(prod.get("brand") or "").lower()
            if brand in prod_brand or prod_brand in brand:
                matched += 1
                continue

        # Animal match — translations come from profile["intent_mapping"] (store-agnostic)
        if animal:
            prod_animals = prod.get("animal") or []
            if isinstance(prod_animals, str):
                prod_animals = [prod_animals]
            prod_animals_lower = [str(a).lower() for a in prod_animals]
            animal_translated = translations.get(animal, animal)
            if any(
                animal in pa or pa in animal or animal_translated in pa or pa in animal_translated
                for pa in prod_animals_lower
            ):
                matched += 1
                continue

        # Price match
        if price_limit:
            try:
                if float(prod.get("price", 0)) <= float(price_limit):
                    matched += 1
                    continue
            except (ValueError, TypeError):
                pass

    return 1.0 if matched > 0 else 0.0


def _category_score(intent_cat_en: List[str], top_categories: List[Tuple]) -> Tuple[float, float]:
    """
    Возвращает (cat_score, category_prior).
    cat_score — базовый вес категории в формуле.
    category_prior — доля категории в каталоге (из top_categories).
    """
    if not intent_cat_en or not top_categories:
        return 0.3, 0.0

    # top_categories: [(cat_name_en, probability), ...]
    best_prior = 0.0
    for cat_name, prob in top_categories:
        cat_lower = cat_name.lower()
        for en in intent_cat_en:
            if en in cat_lower or cat_lower in en:
                best_prior = max(best_prior, float(prob))

    # cat_score — фиксированный компонент если категория в category_map
    cat_score = 0.3 if intent_cat_en else 0.0
    return cat_score, best_prior


def _clarity_score(user_query: str) -> float:
    """Ясность запроса по кол-ву токенов. 4-8 слов = максимум."""
    tokens = len(user_query.split())
    if tokens <= 1:
        return 0.1
    if tokens <= 3:
        return 0.5
    if tokens <= 8:
        return 1.0
    if tokens <= 15:
        return 0.8
    return 0.6


def _result_quality_score(result_count: int) -> float:
    """Нормированное кол-во результатов."""
    if result_count == 0:
        return 0.0
    if result_count == 1:
        return 0.4
    if result_count <= 3:
        return 0.7
    if result_count <= 5:
        return 0.9
    return 1.0


# ─── Основные функции ─────────────────────────────────────────────────────────

def compute_confidence(
    sim_score: float,
    cat_score: float,
    attr_score: float,
    result_quality: float,
    clarity_score: float,
    direct_match: float,
    category_prior: float,
    specialization_bonus: float,
) -> float:
    """
    Итоговая формула confidence.
    Веса: sim=0.40, cat=0.20, attr=0.12, results=0.12, clarity=0.08, direct=0.08
    + category_prior (аддитивный, до 0.3 влияния)
    + specialization_bonus (0.07 если main_category совпадает)
    """
    base = (
        W_SIM     * sim_score +
        W_CAT     * cat_score +
        W_ATTR    * attr_score +
        W_RESULTS * result_quality +
        W_CLARITY * clarity_score +
        W_DIRECT  * direct_match
    )
    # category_prior — аддитивный бонус (масштабируем до 0..0.3)
    prior_bonus = min(category_prior * 0.3, 0.30)
    total = base + prior_bonus + specialization_bonus
    return round(min(total, 1.0), 4)


def decide_mode(
    confidence: float,
    result_count: int,
    sim_score: float,
) -> str:
    """
    Логика маршрутизации:
      NO_RESULTS        — только если result_count == 0
      SHOW_PRODUCTS     — confidence >= THRESHOLD_SHOW И sim_score >= 0.35
      ASK_CLARIFICATION — всё остальное при result_count > 0
    """
    if result_count == 0:
        return "NO_RESULTS"
    # v1.7.1: sim_score порог снижен с 0.35 до 0.25 — для случаев когда
    # final_score низкий из-за отсутствия target/cat-bonus (размытые запросы).
    if confidence >= THRESHOLD_SHOW and sim_score >= 0.25:
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
    Главная точка входа Confidence Engine.

    Args:
        search_result: результат retrieval.search() — {"products": [...], "status": ...}
        intent:        результат intent extraction — {"entities": {...}, "action": ...}
        user_query:    исходный запрос пользователя
        profile:       store_profile.json (для top_categories, main_category)
        category_map:  category_map.json (для UA/RU→EN резолвинга)

    Returns:
        {
            "mode":           "SHOW_PRODUCTS" | "ASK_CLARIFICATION" | "NO_RESULTS",
            "confidence":     float,
            "sim_score":      float,
            "top_categories": [...],
            "all_products":   [...],
            "breakdown":      {...},  # для дебага
        }
    """
    products    = search_result.get("products", [])
    result_count = len(products)
    all_products = products  # сохраняем для передачи в ASK_CLARIFICATION

    ent = intent or {}
    entities = ent.get("entities", {}) if "entities" in ent else ent

    # ── top_categories из profile ─────────────────────────────────────────
    profile = profile or {}
    category_distribution = profile.get("category_distribution", {})
    top_categories = sorted(
        [(k, v) for k, v in category_distribution.items() if isinstance(v, (int, float))],
        key=lambda x: x[1], reverse=True
    )[:5]

    # ── intent category → EN резолвинг ────────────────────────────────────
    intent_cat_raw = str(
        entities.get("category") or
        ent.get("category") or
        search_result.get("detected_category") or
        ""
    ).strip()

    intent_cat_en_list = resolve_category_en(intent_cat_raw, category_map)
    intent_cat_en_str  = intent_cat_en_list[0] if intent_cat_en_list else ""
    detected_cat       = search_result.get("detected_category", "")

    logger.debug(
        f"[CONFIDENCE] category_map resolved: {intent_cat_raw!r} → "
        f"{intent_cat_en_list or 'not found'}"
    )

    # ── Animal translations from store profile (store-agnostic) ──────────
    intent_mapping = profile.get("intent_mapping", {})
    animal_cfg = intent_mapping.get("animal", {})
    animal_translations = animal_cfg.get("translations", {}) if isinstance(animal_cfg, dict) else {}

    # ── Scores ────────────────────────────────────────────────────────────
    sim_score    = _extract_sim_score(products)
    direct_match = _direct_match_score(user_query, products)
    attr_score   = _attr_match_score(entities, products, animal_translations=animal_translations)
    result_qual  = _result_quality_score(result_count)
    clarity      = _clarity_score(user_query)
    cat_score, category_prior = _category_score(intent_cat_en_list, top_categories)

    logger.debug(
        f"[CONFIDENCE] top_categories={top_categories}"
    )
    logger.debug(
        f"[CONFIDENCE] category_prior: category={intent_cat_en_str!r} "
        f"prior={category_prior} "
        f"(intent_raw={intent_cat_raw!r} intent_en={intent_cat_en_str!r} detected={detected_cat!r})"
    )

    # ── Specialization bonus ──────────────────────────────────────────────
    main_cat = str(profile.get("main_category", "")).lower()
    spec_bonus = 0.0
    if main_cat and intent_cat_en_str and (
        main_cat in intent_cat_en_str or intent_cat_en_str in main_cat
    ):
        spec_bonus = SPEC_BONUS
        logger.debug(f"[CONFIDENCE] specialization_bonus={spec_bonus} ({intent_cat_en_str!r} matches main_category={main_cat!r})")

    # ── Animal в entities для дебага ──────────────────────────────────────
    animal_debug = str(
        entities.get("animal") or
        (entities.get("properties") or {}).get("animal") or ""
    )

    # ── Итоговый confidence ───────────────────────────────────────────────
    confidence = compute_confidence(
        sim_score          = sim_score,
        cat_score          = cat_score,
        attr_score         = attr_score,
        result_quality     = result_qual,
        clarity_score      = clarity,
        direct_match       = direct_match,
        category_prior     = category_prior,
        specialization_bonus = spec_bonus,
    )

    mode = decide_mode(confidence, result_count, sim_score)

    logger.info(
        f"[CONFIDENCE] mode={mode} score={confidence} "
        f"sim={sim_score} direct={direct_match} cat={cat_score} attr={attr_score} "
        f"prior={category_prior} spec_bonus={spec_bonus} "
        f"results={result_count} tokens={len(user_query.split())} "
        f"animal={animal_debug!r} "
        f"intent_cat={intent_cat_raw!r} intent_cat_en={intent_cat_en_str!r} "
        f"detected={detected_cat!r} query={user_query!r}"
    )

    return {
        "mode":           mode,
        "confidence":     confidence,
        "sim_score":      sim_score,
        "top_categories": top_categories,
        "all_products":   all_products,
        "breakdown": {
            "sim":              sim_score,
            "direct_match":     direct_match,
            "cat":              cat_score,
            "attr":             attr_score,
            "result_quality":   result_qual,
            "clarity":          clarity,
            "category_prior":   category_prior,
            "spec_bonus":       spec_bonus,
            "intent_cat_raw":   intent_cat_raw,
            "intent_cat_en":    intent_cat_en_str,
            "animal":           animal_debug,
            "detected_cat":     detected_cat,
        },
    }