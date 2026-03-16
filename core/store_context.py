# -*- coding: utf-8 -*-
# /root/ukrsell_v4/core/store_context.py v7.2.1
import json
import re
import time
import os
import asyncio
from typing import List, Dict, Any, Optional
from core.logger import logger, log_event
from core.llm_selector import LLMSelector
from core.store_profiler import StoreProfiler

class StoreContext:
    """
    Universal Store Context v7.2.1 (Agnostic Taxonomy & Schema-Aware).
    
    Changelog:
        v7.1.2  Schema-Aware filtering and dynamic metadata.
        v7.2.0  [TAXONOMY UPDATE]
                - Интеграция Catalog Taxonomy Layer (cluster_products.json).
                - Построение subtype_index для O(1) роутинга.
        v7.2.1  [AGNOSTIC UPDATE]
                - Удаление жестко зашитых ссылок на 'animal'.
                - Динамическая валидация сущностей на основе schema_keys.
                - Полная совместимость с Analyzer v8.2.x.
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
        self.category_map: dict = {}     # UA/RU → EN DB-ключи
        self.search_synonyms: dict = {}  # EN категория → синонимы
        
        # Taxonomy & Clusters
        self.taxonomy_data = {}
        self.subtype_index = {} # Плоский индекс для O(1) поиска (Catalog Router)
        
        self.is_ready = False
        self.data_ready = asyncio.Event()

        # Передаём shared selector
        self.profiler = StoreProfiler(self.base_path, selector=llm_selector)

    async def initialize(self) -> bool:
        """Asynchronously loads store profile, configurations, taxonomy and metadata schema."""
        try:
            # 1. Загрузка базовых конфигов
            self.config = self._load_local_config()
            self.profile = await self.profiler.load_or_build()
            self.schema_keys = self._load_schema_keys()
            
            # 2. Определение языка и локализация
            self.language = self.config.get("language") or self._detect_language()
            self.prompts = self._load_local_prompts()
            
            # 3. Параметры поиска
            self.threshold_main = float(self.config.get("threshold_main", 0.35))
            self.threshold_low = float(self.config.get("threshold_low", 0.22))
            self.currency = self.config.get("currency", "грн")
            
            # 4. Загрузка маппингов и синонимов
            self.category_map = self._load_json_config("category_map.json", required=False)
            self.search_synonyms = self._load_json_config("search_synonyms.json", required=False)
            
            # 5. Catalog Taxonomy Layer
            self.taxonomy_data = self._load_json_config("cluster_products.json", required=False)
            self._build_taxonomy_index()

            if self.category_map:
                logger.info(f"[{self.slug}] category_map loaded: {len(self.category_map)} entries.")
            
            logger.info(f"[{self.slug}] StoreContext v7.2.1 initialized. Language: {self.language}")
            
            self.is_ready = True
            self.data_ready.set()
            return True
            
        except Exception as e:
            logger.error(f"[{self.slug}] Critical error during StoreContext initialization: {e}")
            return False

    def _build_taxonomy_index(self):
        """Создает быстрый агностический маппинг subtype -> context для Catalog Router."""
        if not self.taxonomy_data:
            return
        
        count = 0
        for category, subtypes in self.taxonomy_data.items():
            if not isinstance(subtypes, list): continue
            for item in subtypes:
                st_name = item.get("subtype", "").lower()
                if st_name:
                    # Сохраняем все метаданные из кластера для роутинга
                    self.subtype_index[st_name] = {
                        "category": category,
                        "count": item.get("count", 0)
                    }
                    # Динамически копируем любые другие атрибуты (animal, brand, etc)
                    for key, val in item.items():
                        if key not in ["subtype", "count", "category"]:
                            self.subtype_index[st_name][key] = val
                    
                    count += 1
        logger.info(f"[{self.slug}] Agnostic taxonomy index built: {count} subtypes indexed.")

    def get_taxonomy_hint(self, query: str) -> Optional[Dict]:
        """
        Catalog Router logic.
        Проверяет, упоминается ли какой-либо подтип из таксономии в запросе пользователя.
        """
        if not self.subtype_index:
            return None
            
        q = query.lower()
        matches = []
        for subtype, info in self.subtype_index.items():
            # Используем границы слов для точности
            if re.search(rf'\b{re.escape(subtype)}\b', q):
                matches.append({"subtype": subtype, **info})
        
        if not matches:
            return None
            
        # Возвращаем совпадение с самым длинным названием подтипа (самое точное)
        return max(matches, key=lambda x: len(x["subtype"]))

    def _load_schema_keys(self) -> List[str]:
        """Loads available attribute keys from schema.json or store_profile.json."""
        schema_path = os.path.join(self.base_path, "schema.json")
        profile_path = os.path.join(self.base_path, "store_profile.json")
        
        if os.path.exists(schema_path):
            try:
                with open(schema_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    keys = data.get("product_attributes", []) or data.get("keys", [])
                    if keys: return keys
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
                if not products: return "Ukrainian"
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
            "not_found": "Нічого не знайдено." if is_ukr else "Ничего не найдено.",
            "price_label": "Ціна" if is_ukr else "Цена"
        }
        patch_path = os.path.join(self.base_path, "fsm_soft_patch.json")
        if os.path.exists(patch_path):
            try:
                with open(patch_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    custom = data.get("prompts", {})
                    if custom: return {**defaults, **custom}
            except Exception: pass
            
        prompts_path = os.path.join(self.base_path, "prompts.json")
        if os.path.exists(prompts_path):
            try:
                with open(prompts_path, 'r', encoding='utf-8') as f:
                    custom = json.load(f)
                    return {**defaults, **custom}
            except Exception: pass
        return defaults

    async def get_human_intro(self, query: str, product_count: int) -> str:
        """Generates a persona-driven intro sentence via LLM."""
        if not self.config.get("use_llm_persona", True) or not self.selector:
            return self.prompts.get("search_header", "Ось результати:")

        try:
            if len(query) > 45:
                client, model = await self.selector.get_heavy()
            else:
                client, model = await self.selector.get_fast()

            lang_note = "Пиши на русском языке." if self.language == "Russian" else "Пиши українською мовою."
            system_prompt = (
                f"{self.get_store_bio()}\nYou are a professional sales assistant. {lang_note} "
                f"Customer asked: '{query}'. Found {product_count} items. "
                "Write ONE short, friendly and energetic sentence to introduce results. Use 1 emoji."
            )

            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}],
                max_tokens=100,
                temperature=0.7
            )

            content = getattr(response.choices[0].message, 'content', None)
            if content and isinstance(content, str) and content.strip():
                return content.strip().replace('"', '')
            return self.prompts.get("search_header", "Ось результати:")

        except Exception as e:
            logger.error(f"[{self.slug}] get_human_intro error: {e}")
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
                if isinstance(data, dict):
                    return {k: v for k, v in data.items() if not k.startswith("_")}
                return data
        except Exception as e:
            logger.error(f"[{self.slug}] Failed to load '{filename}': {e}")
            return {}

    def validate_query_features(self, raw_features: Dict) -> Dict:
        """Валидация и нормализация сущностей (v7.2.1 Agnostic)."""
        validated = {
            "category": raw_features.get("category"),
            "price_limit": None,
            "brand": raw_features.get("brand"),
            "dynamic_filters": {}
        }
        
        # Парсинг цены
        price = raw_features.get("price_limit") or raw_features.get("max_price")
        if price:
            try:
                if isinstance(price, (int, float)):
                    validated["price_limit"] = float(price)
                else:
                    digits = re.findall(r'[\d.]+', str(price).replace(',', '.'))
                    validated["price_limit"] = float(digits[0]) if digits else None
            except: pass
            
        # Динамический маппинг свойств на основе разрешенной схемы магазина
        props = raw_features.get("properties", {}) or raw_features.get("attributes", {})
        if isinstance(props, dict):
            for k, v in props.items():
                # Если ключ есть в схеме или является базовым (brand, color, etc)
                if k in self.schema_keys or k in ["color", "material", "size", "subtype"]:
                    validated["dynamic_filters"][k] = v
                # Перенос специфических ключей (например, animal в зоомагазине) в динамические фильтры
                elif k == "animal":
                    validated["dynamic_filters"]["animal"] = v
                    
        return validated

    def __repr__(self):
        return f"<StoreContext {self.slug} ready={self.is_ready} taxonomy={len(self.subtype_index)}>"