# /root/ukrsell_v4/core/store_context.py v8.2.0
"""
StoreContext v8.2.0 — SQL-native контекст магазина.

Changelog v8.2.0:
  - FIX (оптимизация #9): retry добавлен в _detect_language при database is locked
    (консистентно с _load_schema_keys).
  - FIX (важно #5): safe parse в _load_schema_keys — явный guard на None/пустые значения.
  - Minor: улучшены комментарии и типизация.

Changelog v8.1.0:
  - FIX (критично): finally: self.data_ready.set() в initialize().
  - FIX: retry в _load_schema_keys при database is locked.
  - FIX: _detect_language использует detect_language_from_titles из core.utils (DRY).

Changelog v8.0.0:
  - self.db_path — путь к products.db.
  - _load_schema_keys() async: приоритет store_meta, fallback JSON.
  - _detect_language() async: SELECT title FROM products LIMIT 15.
  - Справочные JSON (config.json, fsm_soft_patch.json, category_map.json и др.) — без изменений.
  - _build_taxonomy_index(), get_taxonomy_hint(), validate_query_features(),
    get_store_bio(), get_human_intro() — без изменений.
"""

import json
import re
import time
import os
import asyncio
import aiosqlite
from typing import List, Dict, Any, Optional

from core.logger import logger, log_event
from core.llm_selector import LLMSelector
from core.store_profiler import StoreProfiler
from core.utils import detect_language_from_titles

DB_TIMEOUT         = 10.0
DB_LOCK_RETRIES    = 2
DB_LOCK_RETRY_WAIT = 0.5


class StoreContext:
    """
    Universal Store Context v8.2.0 (SQL-Native, Agnostic Taxonomy & Schema-Aware).
    """

    def __init__(
        self,
        base_path: str,
        db_engine: Any,
        llm_selector: LLMSelector,
        kernel: Any = None,
    ):
        self.base_path = base_path
        self.db        = db_engine
        self.selector  = llm_selector
        self.kernel    = kernel

        self.slug    = os.path.basename(os.path.normpath(base_path))
        self.db_path = os.path.join(base_path, "products.db")

        # Core store states
        self.config:      Dict[str, Any] = {
            "currency": "грн", "bot_token": "NO_TOKEN", "use_llm_persona": True
        }
        self.profile:     Dict[str, Any] = {}
        self.prompts:     Dict[str, str] = {}
        self.language:    str            = "Ukrainian"
        self.schema_keys: List[str]      = []

        self.category_index:     Dict[str, Any] = {}
        self._categories_cache:  List[str]       = []
        self._last_cache_update: float           = time.time()
        self.threshold_main: float = 0.35
        self.threshold_low:  float = 0.22
        self.currency:       str   = "грн"
        self.category_map:    Dict[str, Any] = {}
        self.search_synonyms: Dict[str, Any] = {}

        # Taxonomy & Clusters
        self.taxonomy_data: Dict[str, Any] = {}
        self.subtype_index: Dict[str, Any] = {}

        self.is_ready   = False
        self.data_ready = asyncio.Event()

        self.profiler = StoreProfiler(self.base_path, selector=llm_selector)

    # ── Инициализация ─────────────────────────────────────────

    async def initialize(self) -> bool:
        """
        Асинхронная загрузка контекста магазина.

        FIX (критично): finally: self.data_ready.set() гарантирует что Event
        устанавливается всегда — даже при исключении. Без этого StoreEngine.__aenter__
        зависает на await data_ready.wait() навсегда.

        Порядок:
          1. config.json — токен, валюта, пороги.
          2. StoreProfiler.load_or_build() — агрегаты из store_meta (SQL).
          3. schema_keys — store_meta (SQL) → schema.json → store_profile.json.
          4. language — из profile → config → SELECT titles (SQL) → 'Ukrainian'.
          5. Промпты, маппинги, таксономия.
        """
        try:
            # 1. Базовые конфиги из файлов
            self.config  = self._load_local_config()
            self.profile = await self.profiler.load_or_build()

            # 2. schema_keys — async, приоритет SQL
            self.schema_keys = await self._load_schema_keys()

            # 3. Язык — приоритет из profiler (уже определён через SQL)
            lang_from_profile = self.profile.get("language")
            if lang_from_profile:
                self.language = lang_from_profile
            else:
                self.language = (
                    self.config.get("language")
                    or await self._detect_language()
                )

            # 4. Промпты
            self.prompts = self._load_local_prompts()

            # 5. Параметры поиска
            self.threshold_main = float(self.config.get("threshold_main", 0.35))
            self.threshold_low  = float(self.config.get("threshold_low",  0.22))
            self.currency       = self.config.get("currency", "грн")

            # 6. Справочные JSON-маппинги (остаются файлами)
            self.category_map    = self._load_json_config("category_map.json",    required=False)
            self.search_synonyms = self._load_json_config("search_synonyms.json", required=False)

            # 7. Catalog Taxonomy Layer
            self.taxonomy_data = self._load_json_config("cluster_products.json", required=False)
            self._build_taxonomy_index()

            if self.category_map:
                logger.info(
                    f"[{self.slug}] category_map loaded: {len(self.category_map)} entries."
                )

            logger.info(
                f"[{self.slug}] StoreContext v8.2.0 initialized. "
                f"Language: {self.language}, schema_keys: {len(self.schema_keys)}"
            )
            self.is_ready = True
            return True

        except Exception as e:
            logger.error(
                f"[{self.slug}] Critical error during StoreContext initialization: {e}",
                exc_info=True,
            )
            return False

        finally:
            # FIX (критично): гарантированно разблокируем всех ожидающих data_ready,
            # даже если initialize() упал с исключением. Без этого вечный deadlock.
            self.data_ready.set()

    # ── SQL-методы ────────────────────────────────────────────

    async def _load_schema_keys(self) -> List[str]:
        """
        Загружает schema_keys из store_meta (SQL) с retry при 'database is locked'.

        FIX #5: safe guard — проверяем что значение не None и не пустой список
        после парсинга JSON.

        Приоритет:
          1. SELECT value FROM store_meta WHERE key='schema_keys' — products.db.
          2. schema.json — поля 'product_attributes' или 'keys'.
          3. store_profile.json — profile.schema_keys.
          4. Пустой список.
        """
        # 1. SQL — главный источник
        if os.path.exists(self.db_path):
            for attempt in range(DB_LOCK_RETRIES + 1):
                try:
                    async with aiosqlite.connect(self.db_path, timeout=DB_TIMEOUT) as db:
                        async with db.execute(
                            "SELECT value FROM store_meta WHERE key = 'schema_keys'"
                        ) as cur:
                            row = await cur.fetchone()

                    if row and row[0]:
                        try:
                            keys = json.loads(row[0])
                            # FIX #5: guard на None и пустой список
                            if keys is not None and isinstance(keys, list) and len(keys) > 0:
                                logger.debug(
                                    f"[{self.slug}] schema_keys loaded from store_meta "
                                    f"({len(keys)} keys)."
                                )
                                return keys
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.warning(
                                f"[{self.slug}] schema_keys JSON parse error: {e}"
                            )
                    break  # успешно прочитали (пусто) — идём к fallback

                except aiosqlite.OperationalError as e:
                    err_str = str(e).lower()
                    if "no such table" in err_str:
                        break  # таблица ещё не создана
                    if "database is locked" in err_str and attempt < DB_LOCK_RETRIES:
                        logger.warning(
                            f"[{self.slug}] DB locked in _load_schema_keys, "
                            f"retry {attempt + 1}/{DB_LOCK_RETRIES}..."
                        )
                        await asyncio.sleep(DB_LOCK_RETRY_WAIT * (attempt + 1))
                        continue
                    logger.warning(f"[{self.slug}] _load_schema_keys SQL error: {e}")
                    break
                except Exception as e:
                    logger.warning(
                        f"[{self.slug}] _load_schema_keys unexpected error: {e}"
                    )
                    break

        # 2. schema.json fallback
        schema_path = os.path.join(self.base_path, "schema.json")
        if os.path.exists(schema_path):
            try:
                with open(schema_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                keys = data.get("product_attributes", []) or data.get("keys", [])
                if keys and isinstance(keys, list):
                    logger.debug(f"[{self.slug}] schema_keys loaded from schema.json.")
                    return keys
            except Exception as e:
                logger.warning(f"[{self.slug}] schema.json read error: {e}")

        # 3. store_profile.json fallback
        profile_path = os.path.join(self.base_path, "store_profile.json")
        if os.path.exists(profile_path):
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                keys = data.get("profile", {}).get("schema_keys", [])
                if keys and isinstance(keys, list):
                    logger.debug(
                        f"[{self.slug}] schema_keys loaded from store_profile.json fallback."
                    )
                    return keys
            except Exception as e:
                logger.warning(f"[{self.slug}] store_profile.json read error: {e}")

        return []

    async def _detect_language(self) -> str:
        """
        Определяет язык магазина по заголовкам из products.db.

        FIX #9: retry при database is locked (консистентно с _load_schema_keys).
        Использует detect_language_from_titles() из core.utils (DRY).

        Приоритет:
          1. SELECT title FROM products LIMIT 15 → detect_language_from_titles().
          2. 'Ukrainian' как безопасный default.
        """
        if not os.path.exists(self.db_path):
            return "Ukrainian"

        for attempt in range(DB_LOCK_RETRIES + 1):
            try:
                async with aiosqlite.connect(self.db_path, timeout=DB_TIMEOUT) as db:
                    async with db.execute(
                        "SELECT title FROM products WHERE title IS NOT NULL LIMIT 15"
                    ) as cur:
                        rows = await cur.fetchall()

                if rows:
                    return detect_language_from_titles(
                        [row[0] for row in rows if row[0]]
                    )
                return "Ukrainian"

            except aiosqlite.OperationalError as e:
                err_str = str(e).lower()
                if "no such table" in err_str:
                    return "Ukrainian"
                if "database is locked" in err_str and attempt < DB_LOCK_RETRIES:
                    logger.warning(
                        f"[{self.slug}] DB locked in _detect_language, "
                        f"retry {attempt + 1}/{DB_LOCK_RETRIES}..."
                    )
                    await asyncio.sleep(DB_LOCK_RETRY_WAIT * (attempt + 1))
                    continue
                logger.warning(f"[{self.slug}] _detect_language SQL error: {e}")
                return "Ukrainian"
            except Exception as e:
                logger.warning(f"[{self.slug}] _detect_language error: {e}")
                return "Ukrainian"

        return "Ukrainian"

    # ── Taxonomy ──────────────────────────────────────────────

    def _build_taxonomy_index(self):
        """Создает быстрый агностический маппинг subtype → context для Catalog Router."""
        if not self.taxonomy_data:
            return

        count = 0
        for category, subtypes in self.taxonomy_data.items():
            if not isinstance(subtypes, list):
                continue
            for item in subtypes:
                st_name = item.get("subtype", "").lower()
                if not st_name:
                    continue
                self.subtype_index[st_name] = {
                    "category": category,
                    "count":    item.get("count", 0),
                }
                for key, val in item.items():
                    if key not in ["subtype", "count", "category"]:
                        self.subtype_index[st_name][key] = val
                count += 1

        logger.info(
            f"[{self.slug}] Agnostic taxonomy index built: {count} subtypes indexed."
        )

    def get_taxonomy_hint(self, query: str) -> Optional[Dict]:
        """Catalog Router: проверяет упоминание подтипа из таксономии в запросе."""
        if not self.subtype_index:
            return None

        q       = query.lower()
        matches = []
        for subtype, info in self.subtype_index.items():
            if re.search(rf'\b{re.escape(subtype)}\b', q):
                matches.append({"subtype": subtype, **info})

        if not matches:
            return None

        return max(matches, key=lambda x: len(x["subtype"]))

    # ── Публичные методы ──────────────────────────────────────

    def get_store_bio(self) -> str:
        """Генерирует описание магазина для системного промпта LLM."""
        p           = self.profile if self.profile else {}
        main_cat    = p.get("main_category", "товари")
        total       = p.get("total_sku", 0)
        brands_list = p.get("brand_matrix", {}).get("top_brands", [])
        brands      = ", ".join(brands_list[:5]) if brands_list else "різні бренди"
        prices      = p.get("price_analytics", {})

        return (
            f"Магазин: {self.slug}. Спеціалізація: {main_cat} (всього {total} SKU). "
            f"Основні бренди: {brands}. "
            f"Ціновий діапазон: від {prices.get('min', 0)} "
            f"до {prices.get('max', 0)} {self.currency}."
        )

    async def get_human_intro(self, query: str, product_count: int) -> str:
        """Генерирует persona-driven вступление через LLM."""
        if not self.config.get("use_llm_persona", True) or not self.selector:
            return self.prompts.get("search_header", "Ось результати:")

        try:
            if len(query) > 45:
                client, model = await self.selector.get_heavy()
            else:
                client, model = await self.selector.get_fast()

            lang_note = (
                "Пиши на русском языке."
                if self.language == "Russian"
                else "Пиши українською мовою."
            )
            system_prompt = (
                f"{self.get_store_bio()}\nYou are a professional sales assistant. "
                f"{lang_note} Customer asked: '{query}'. Found {product_count} items. "
                "Write ONE short, friendly and energetic sentence to introduce results. "
                "Use 1 emoji."
            )

            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}],
                max_tokens=100,
                temperature=0.7,
            )

            content = getattr(response.choices[0].message, "content", None)
            if content and isinstance(content, str) and content.strip():
                return content.strip().replace('"', "")
            return self.prompts.get("search_header", "Ось результати:")

        except Exception as e:
            logger.error(f"[{self.slug}] get_human_intro error: {e}")
            return self.prompts.get("search_header", "Ось результати:")

    def validate_query_features(self, raw_features: Dict) -> Dict:
        """Валидация и нормализация сущностей (Agnostic v8.2.0)."""
        validated: Dict[str, Any] = {
            "category":        raw_features.get("category"),
            "price_limit":     None,
            "brand":           raw_features.get("brand"),
            "dynamic_filters": {},
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
            except (ValueError, IndexError):
                pass

        # Динамический маппинг свойств на основе разрешённой схемы
        props = raw_features.get("properties", {}) or raw_features.get("attributes", {})
        if isinstance(props, dict):
            for k, v in props.items():
                if k in self.schema_keys or k in ["color", "material", "size", "subtype"]:
                    validated["dynamic_filters"][k] = v
                elif k == "animal":
                    validated["dynamic_filters"]["animal"] = v

        return validated

    # ── Загрузка файловых конфигов ────────────────────────────

    def _load_local_config(self) -> Dict[str, Any]:
        """Загружает config.json. Остаётся файлом."""
        config_path = os.path.join(self.base_path, "config.json")
        try:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"[{self.slug}] config.json read error: {e}")
        return {"currency": "грн", "use_llm_persona": True}

    def _load_local_prompts(self) -> Dict[str, str]:
        """Загружает промпты из fsm_soft_patch.json или prompts.json. Остаются файлами."""
        is_ukr = self.language == "Ukrainian"
        defaults: Dict[str, str] = {
            "search_header": (
                "Чудовий вибір! Ось результати:" if is_ukr else "Отличный выбор! Вот результаты:"
            ),
            "view_button": "Дивитись" if is_ukr else "Смотреть",
            "not_found":   "Нічого не знайдено." if is_ukr else "Ничего не найдено.",
            "price_label": "Ціна" if is_ukr else "Цена",
        }

        patch_path = os.path.join(self.base_path, "fsm_soft_patch.json")
        if os.path.exists(patch_path):
            try:
                with open(patch_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                custom = data.get("prompts", {})
                if custom:
                    return {**defaults, **custom}
            except Exception:
                pass

        prompts_path = os.path.join(self.base_path, "prompts.json")
        if os.path.exists(prompts_path):
            try:
                with open(prompts_path, "r", encoding="utf-8") as f:
                    custom = json.load(f)
                return {**defaults, **custom}
            except Exception:
                pass

        return defaults

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

    def __repr__(self):
        return (
            f"<StoreContext v8.2.0 {self.slug} "
            f"ready={self.is_ready} taxonomy={len(self.subtype_index)}>"
        )