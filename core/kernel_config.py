# /root/ukrsell_v4/core/kernel_config.py v1.0.0

# ── LLM токены ────────────────────────────────────────────────────────────────
BASE_MAX_TOKENS    = 800
TOKENS_PER_PRODUCT = 120
MAX_TOKENS_CAP     = 2000

# ── Ограничения промпта ───────────────────────────────────────────────────────
MAX_PRODUCTS_IN_PROMPT = 5
MAX_ATTRS_PER_PRODUCT  = 3

# ── Heavy/Light path пороги ───────────────────────────────────────────────────
COMPLEX_ENTITIES_THRESHOLD = 2
COMPLEX_QUERY_LENGTH       = 40

# ── Валидация LLM ответа ──────────────────────────────────────────────────────
LLM_MIN_RESPONSE_LEN       = 60
LANG_RU_RATIO_THRESHOLD    = 0.35
MIN_VALID_PRODUCTS_FOR_LLM = 2

# ── Фильтр атрибутов для промпта ─────────────────────────────────────────────
ATTRS_BLACKLIST = frozenset({
    "id", "sku", "uid", "uuid", "product_id", "item_id",
    "created_at", "updated_at", "last_updated",
    "html", "description_html", "meta", "seo",
    "position", "sort_order", "weight_g", "volume_ml",
})