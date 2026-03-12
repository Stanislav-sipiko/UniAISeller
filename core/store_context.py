# /root/ukrsell_v4/core/store_context.py v7.1.2
import json
import re
import time
import os
from typing import List, Dict, Any, Optional
from core.logger import logger, log_event
from core.llm_selector import LLMSelector
from core.store_profiler import StoreProfiler

class StoreContext:
    """
    Universal Store Context v7.1.2 (Schema-Aware).
    - Dynamic Schema: Loads and exposes shop-specific attribute keys for Analyzer.
    - Integrated Filtering: Apply filters based on dynamic metadata.
    - Asset Sync: Manages ready-state for asynchronous initialization.
    - Zero Omission: Full script with safe fallback for LLM responses.
    """

    def __init__(self, base_path: str, db_engine: Any, llm_selector: LLMSelector, kernel: Any = None):
        self.base_path = base_path  
        self.db = db_engine
        self.selector = llm_selector
        self.kernel = kernel
        
        self.slug = os.path.basename(os.path.normpath(base_path))

        # Core store states
        self.config = {"currency": "грн", "bot_token": "NO_TOKEN", "use_llm_persona": True}
        self.profile = {}
        self.prompts = {}
        self.language = "Ukrainian"
        self.schema_keys = []  
        
        self.category_index = {}
        self._categories_cache = []
        self._last_cache_update = time.time()
        self.threshold_main = 0.35
        self.threshold_low = 0.22
        self.currency = "грн"
        self.category_map: dict = {}     # UA/RU → EN DB-ключи, из category_map.json
        self.search_synonyms: dict = {}  # EN категория → синонимы, из search_synonyms.json
        self.is_ready = False

        # Передаём shared selector — StoreProfiler не создаёт свой LLMSelector
        self.profiler = StoreProfiler(self.base_path, selector=llm_selector)

    async def initialize(self) -> bool:
        """Asynchronously loads store profile, configurations and metadata schema."""
        try:
            self.config = self._load_local_config()
            self.profile = await self.profiler.load_or_build()
            self.schema_keys = self._load_schema_keys()
            logger.info(f"[{self.slug}] Loaded {len(self.schema_keys)} schema keys for intent extraction.")
            self.language = self.config.get("language") or self._detect_language()
            self.prompts = self._load_local_prompts()
            self.threshold_main = float(self.config.get("threshold_main", 0.35))
            self.threshold_low = float(self.config.get("threshold_low", 0.22))
            self.currency = self.config.get("currency", "грн")
            self.category_map = self._load_json_config("category_map.json", required=False)
            self.search_synonyms = self._load_json_config("search_synonyms.json", required=False)
            if self.category_map:
                logger.info(f"[{self.slug}] category_map loaded: {len(self.category_map)} entries.")
            else:
                logger.info(f"[{self.slug}] category_map.json not found — category matching by stem only.")
            self.is_ready = True
            return True
        except Exception as e:
            logger.error(f"[{self.slug}] Critical error during StoreContext initialization: {e}")
            return False

    def _load_schema_keys(self) -> List[str]:
        """Loads available attribute keys from schema.json or store_profile.json."""
        schema_path = os.path.join(self.base_path, "schema.json")
        profile_path = os.path.join(self.base_path, "store_profile.json")
        if os.path.exists(schema_path):
            try:
                with open(schema_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("keys", [])
            except Exception: pass
        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("profile", {}).get("schema_keys", [])
            except Exception: pass
        return []

    def get_store_bio(self) -> str:
        """Generates store description for LLM system prompt."""
        p = self.profile if self.profile else {}
        main_cat = p.get("main_category", "товари")
        total = p.get("total_sku", 0)
        brands_list = p.get("brand_matrix", {}).get("top_brands", [])
        brands = ", ".join(brands_list[:5]) if brands_list else "різні бренди"
        prices = p.get("price_analytics", {})
        bio = (
            f"Магазин: {self.slug}. Спеціалізація: {main_cat} (всього {total} SKU). "
            f"Основні бренди: {brands}. "
            f"Ціновий діапазон: від {prices.get('min', 0)} до {prices.get('max', 0)} {self.currency}."
        )
        return bio

    def _detect_language(self) -> str:
        """Определяет язык магазина по выборке названий товаров."""
        data_path = os.path.join(self.base_path, "normalized_products_final.json")
        if not os.path.exists(data_path): return "Ukrainian" 
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                products = json.load(f)
            sample_text = " ".join([str(p.get('title', '')) for p in products[:15]]).lower()
            if any(char in sample_text for char in "іїєґ"): return "Ukrainian"
            if any(char in sample_text for char in "ыэёъ"): return "Russian"
            return "Ukrainian"
        except Exception: return "Ukrainian"

    def _load_local_config(self) -> Dict[str, Any]:
        config_path = os.path.join(self.base_path, "config.json")
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception: pass
        return {"currency": "грн", "use_llm_persona": True}

    def _load_local_prompts(self) -> Dict[str, str]:
        is_ukr = self.language == "Ukrainian"
        defaults = {
            "search_header": "Чудовий вибір! Ось результати:" if is_ukr else "Отличный выбор! Вот результаты:",
            "view_button": "Дивитись" if is_ukr else "Смотреть",
            "not_found": "Нічого не знайдено." if is_ukr else "Ничего не найдено."
        }
        prompts_path = os.path.join(self.base_path, "prompts.json")
        if os.path.exists(prompts_path):
            try:
                with open(prompts_path, 'r', encoding='utf-8') as f:
                    custom = json.load(f)
                    return {**defaults, **custom}
            except Exception: pass
        return defaults

    async def get_human_intro(self, query: str, product_count: int) -> str:
        """
        Generates a persona-driven intro sentence via LLM using selector tiers.
        Always returns a non-empty string.
        No hardcoded models.
        """
        if not self.config.get("use_llm_persona", True) or not self.selector:
            return self.prompts.get("search_header", "Ось результати:")

        try:
            # Выбираем Tier LLM через селектор в зависимости от длины запроса
            if len(query) > 40:
                client, model = await self.selector.get_heavy()
            else:
                client, model = await self.selector.get_fast()

            lang_note = "Пиши на русском языке." if self.language == "Russian" else "Пиши українською мовою."
            system_prompt = (
                f"{self.get_store_bio()}\nYou are a sales consultant. {lang_note} "
                f"Customer asked: '{query}'. Found {product_count} items in our catalog. "
                "Write ONE short, energetic intro sentence with 1 emoji. "
                "STRICT RULE: Do NOT mention any specific product names, brands, or prices — "
                "those will be listed below. Only set the mood."
            )

            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}],
                max_tokens=80
            )

            content = getattr(response.choices[0].message, 'content', None)
            if content and isinstance(content, str) and content.strip():
                return content.strip()
            else:
                return self.prompts.get("search_header", "Ось результати:")

        except Exception as e:
            print(f"💥 [SC_ERROR] get_human_intro fallback activated: {type(e).__name__}: {e}")
            return self.prompts.get("search_header", "Ось результати:")

    def _load_json_config(self, filename: str, required: bool = False) -> dict:
        """Загружает произвольный JSON-конфиг из директории магазина."""
        path = os.path.join(self.base_path, filename)
        if not os.path.exists(path):
            if required:
                logger.warning(f"[{self.slug}] Required config '{filename}' not found.")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Фильтруем служебные ключи _comment*
                return {k: v for k, v in data.items() if not k.startswith("_")}
        except Exception as e:
            logger.error(f"[{self.slug}] Failed to load '{filename}': {e}")
            return {}

    def validate_query_features(self, raw_features: Dict) -> Dict:
        validated = {
            "category": raw_features.get("category"),
            "price_limit": None,
            "brand": raw_features.get("brand"),
            "dynamic_filters": {}
        }
        price = raw_features.get("price_limit") or raw_features.get("max_price")
        if price:
            try:
                digits = re.findall(r'[\d.]+', str(price).replace(',', '.'))
                validated["price_limit"] = float(digits[0]) if digits else None
            except: pass
        props = raw_features.get("properties", {})
        if isinstance(props, dict):
            for k, v in props.items():
                if k in self.schema_keys:
                    validated["dynamic_filters"][k] = v
        return validated

    def __repr__(self):
        return f"<StoreContext {self.slug} ready={self.is_ready}>"