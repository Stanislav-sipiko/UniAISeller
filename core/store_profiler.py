# /root/ukrsell_v4/core/store_profiler.py v8.2.2
"""
StoreProfiler v8.2.2 — SQL-native профилировщик магазина.

"""

import json
import os
import logging
import re
import asyncio
import datetime
import aiosqlite
from typing import Dict, Any, Optional, List

from core.utils import detect_language_from_titles

logger = logging.getLogger("UkrSell_Profiler")

# Минимальный порог: ключ атрибута должен встречаться у >5% товаров
SCHEMA_KEY_MIN_SHARE = 0.05

# Таймаут подключения к БД
DB_TIMEOUT = 10.0

# Retry при 'database is locked'
DB_LOCK_RETRIES    = 2
DB_LOCK_RETRY_WAIT = 0.5  # секунды между попытками


class StoreProfiler:
    """
    StoreProfiler v8.2.2 — SQL-native.

    Читает агрегированную статистику из products.db (store_meta).
    При пересборке — вычисляет агрегаты через SQL, сохраняет в store_meta.
    LLM-персона генерируется только при первой сборке или если приветствие отсутствует.
    Всё через одно соединение aiosqlite на rebuild.
    """

    def __init__(self, store_path: str, selector: Any = None):
        self.store_path   = store_path
        self.db_path      = os.path.join(store_path, "products.db")
        self.config_path  = os.path.join(store_path, "config.json")
        self.profile_path = os.path.join(store_path, "store_profile.json")
        self.profile: Optional[Dict[str, Any]] = None
        self.selector = selector
        self._slug = os.path.basename(os.path.normpath(store_path)) or "unknown_store"

    # ── Точка входа ───────────────────────────────────────────

    async def load_or_build(self) -> Dict[str, Any]:
        """
        Загружает профиль из store_meta или пересобирает если данные изменились.

        Порядок:
          1. products.db недоступна → JSON fallback.
          2. Проверка нужна ли пересборка (_needs_rebuild_sql).
          3. Если нет — грузим из store_meta (одно соединение).
          4. Если да — одно соединение на весь rebuild:
             a. SQL-агрегаты (_compute_db_stats).
             b. Сохраняем базовый профиль в store_meta (без персоны) — bootstrap.
             c. Генерируем LLM-персону вне основного соединения.
             d. Дописываем ai_welcome_message в store_meta.
        """
        if not os.path.exists(self.db_path):
            logger.warning(f"[{self._slug}] products.db not found, falling back to JSON profile.")
            return self._load_json_fallback()

        try:
            needs_rebuild = await self._needs_rebuild_sql()

            if not needs_rebuild:
                async with aiosqlite.connect(self.db_path, timeout=DB_TIMEOUT) as db:
                    await self._apply_pragma(db)
                    profile = await self._load_from_store_meta(db)

                if profile:
                    logger.info(f"[{self._slug}] Profile loaded from store_meta (SQL).")
                    self.profile = profile
                    return profile

                logger.info(f"[{self._slug}] store_meta empty, triggering rebuild.")

            # Пересборка через одно соединение
            if self.selector is None:
                logger.error(
                    f"[{self._slug}] Cannot rebuild profile: selector not provided. "
                    f"Using JSON fallback."
                )
                return self._load_json_fallback()

            await self.selector.ensure_ready()
            logger.info(f"[{self._slug}] Rebuilding store profile via SQL aggregates...")

            async with aiosqlite.connect(self.db_path, timeout=DB_TIMEOUT) as db:
                await self._apply_pragma(db)

                # a. SQL-агрегаты
                stats = await self._compute_db_stats(db)

                # b. Читаем старое приветствие чтобы не пересоздавать без нужды
                old_profile = await self._load_from_store_meta(db)
                existing_welcome = (old_profile or {}).get("ai_welcome_message", "")

                # c. Сохраняем базовый профиль — bootstrap всегда успевает
                stats["ai_welcome_message"] = existing_welcome
                await self._save_to_store_meta(db, stats)
                logger.info(f"[{self._slug}] Base profile saved to store_meta.")

            # d. Генерация персоны — вне основного соединения, может быть долгой
            if existing_welcome and len(existing_welcome) > 10:
                stats["ai_welcome_message"] = existing_welcome
            else:
                logger.info(f"[{self._slug}] Generating AI Persona welcome message...")
                stats["ai_welcome_message"] = await self._generate_ai_persona(stats)

                # Дописываем только ai_welcome_message
                async with aiosqlite.connect(self.db_path, timeout=DB_TIMEOUT) as db:
                    await self._apply_pragma(db)
                    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    await db.execute(
                        "INSERT OR REPLACE INTO store_meta (key, value, updated_at) "
                        "VALUES ('ai_welcome_message', ?, ?)",
                        (stats["ai_welcome_message"], now),
                    )
                    await db.commit()

            self.profile = stats
            return stats

        except Exception as e:
            logger.error(
                f"[{self._slug}] StoreProfiler error: {e}. Falling back to JSON.",
                exc_info=True,
            )
            return self._load_json_fallback()

    # ── Триггер пересборки ────────────────────────────────────

    async def _needs_rebuild_sql(self) -> bool:
        """
        Проверяет нужна ли пересборка профиля.

        FIX #3: guard 'last_data_update' not in meta → сразу True.
        FIX #8: COUNT(*) и MAX(last_updated) в одном запросе.
        FIX: обработан max_updated is None.
        FIX: retry при database is locked.

        Критерии пересборки:
          1. store_meta не содержит 'total_sku' → первый запуск.
          2. store_meta не содержит 'last_data_update' → неполный профиль.
          3. COUNT(*) FROM products != total_sku → загружены новые/удалены товары.
          4. MAX(last_updated) FROM products > last_data_update → обновлены товары.
        """
        for attempt in range(DB_LOCK_RETRIES + 1):
            try:
                async with aiosqlite.connect(self.db_path, timeout=DB_TIMEOUT) as db:
                    await self._apply_pragma(db)

                    # Читаем мета-ключи за один запрос
                    async with db.execute(
                        "SELECT key, value FROM store_meta "
                        "WHERE key IN ('total_sku', 'last_data_update')"
                    ) as cur:
                        rows = await cur.fetchall()

                    meta = {row[0]: row[1] for row in rows}

                    # FIX #3: guard на отсутствие ключей
                    if "total_sku" not in meta:
                        return True
                    if "last_data_update" not in meta:
                        return True

                    # FIX #8: COUNT(*) + MAX(last_updated) в одном запросе
                    async with db.execute("""
                        SELECT
                            COUNT(*)         AS total_sku,
                            MAX(last_updated) AS max_updated
                        FROM products
                    """) as cur:
                        row = await cur.fetchone()

                    current_count   = row[0] or 0
                    max_updated_str = row[1]  # может быть None если нет товаров

                    # Проверка количества
                    try:
                        stored_sku = int(meta["total_sku"])
                    except (ValueError, TypeError):
                        return True

                    if current_count != stored_sku:
                        return True

                    # FIX: обработка max_updated is None
                    if max_updated_str is None:
                        return False  # нет данных о времени — не перестраиваем

                    # FIX: сравниваем даты через fromisoformat, не строками
                    try:
                        max_updated = datetime.datetime.fromisoformat(max_updated_str)
                        last_update = datetime.datetime.fromisoformat(
                            meta["last_data_update"]
                        )
                        # Нормализуем timezone для корректного сравнения
                        if max_updated.tzinfo is None:
                            max_updated = max_updated.replace(tzinfo=datetime.timezone.utc)
                        if last_update.tzinfo is None:
                            last_update = last_update.replace(tzinfo=datetime.timezone.utc)
                        if max_updated > last_update:
                            return True
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"[{self._slug}] Date parse error in _needs_rebuild_sql: {e}. "
                            f"Triggering rebuild."
                        )
                        return True

                    return False

            except aiosqlite.OperationalError as e:
                err_str = str(e).lower()
                if "no such table" in err_str:
                    return True
                if "database is locked" in err_str and attempt < DB_LOCK_RETRIES:
                    logger.warning(
                        f"[{self._slug}] DB locked in _needs_rebuild_sql, "
                        f"retry {attempt + 1}/{DB_LOCK_RETRIES}..."
                    )
                    await asyncio.sleep(DB_LOCK_RETRY_WAIT * (attempt + 1))
                    continue
                logger.error(f"[{self._slug}] _needs_rebuild_sql OperationalError: {e}")
                return True
            except Exception as e:
                logger.error(f"[{self._slug}] _needs_rebuild_sql error: {e}")
                return True

        return True  # после всех retry — лучше пересобрать

    # ── Построение профиля из SQL ─────────────────────────────

    async def _compute_db_stats(self, db: aiosqlite.Connection) -> Dict[str, Any]:
        """
        SQL-агрегаты вместо Python-итерации по JSON.
        Принимает уже открытое соединение — не создаёт новое.

        FIX #1: schema_keys SQL — подзапрос с LIMIT внутри, GROUP BY снаружи.
        FIX #2: key_counts[row[0]] = row[1] — GROUP BY уже агрегирует.
        FIX #6: attributes LIKE '{%' — строгая проверка начала JSON-объекта.
        FIX #7: total_brands через SELECT COUNT(DISTINCT brand).
        """
        try:
            # 1. Базовые агрегаты
            async with db.execute("""
                SELECT
                    COUNT(*)          AS total_sku,
                    MIN(price_min)    AS price_min,
                    MAX(price_max)    AS price_max,
                    AVG(price_min)    AS price_avg
                FROM products
            """) as cur:
                base = await cur.fetchone()

            total_sku = base[0] or 0
            price_min = float(base[1] or 0.0)
            price_max = float(base[2] or 0.0)
            price_avg = round(float(base[3] or 0.0), 2)

            # 2. Распределение категорий
            async with db.execute("""
                SELECT category, COUNT(*) AS cnt
                FROM products
                WHERE category IS NOT NULL AND category != ''
                GROUP BY category
                ORDER BY cnt DESC
            """) as cur:
                cat_rows = await cur.fetchall()

            category_distribution: Dict[str, float] = {}
            main_category = "none"
            store_type    = "multi_category"

            if cat_rows and total_sku > 0:
                main_category = cat_rows[0][0].lower()
                main_share    = cat_rows[0][1] / total_sku
                store_type    = "mono_vertical" if main_share > 0.8 else "multi_category"
                category_distribution = {
                    row[0].lower(): round(row[1] / total_sku, 3)
                    for row in cat_rows
                }

            # 3. Топ-10 брендов
            async with db.execute("""
                SELECT brand, COUNT(*) AS cnt
                FROM products
                WHERE brand IS NOT NULL AND brand != ''
                GROUP BY brand
                ORDER BY cnt DESC
                LIMIT 10
            """) as cur:
                brand_rows = await cur.fetchall()

            top_brands = [row[0].capitalize() for row in brand_rows]

            # FIX #7: реальное количество уникальных брендов, не len(top10)
            async with db.execute("""
                SELECT COUNT(DISTINCT brand)
                FROM products
                WHERE brand IS NOT NULL AND brand != ''
            """) as cur:
                brand_count_row = await cur.fetchone()
            total_brands = brand_count_row[0] if brand_count_row else 0

            # 4. schema_keys через json_each(attributes).
            # FIX #1: подзапрос с LIMIT внутри ограничивает сканирование,
            #         GROUP BY снаружи агрегирует без DISTINCT (дешевле).
            # FIX #6: LIKE '{%' — строгая проверка что attributes начинается с '{'.
            # ORDER BY last_updated DESC — schema строится по актуальным товарам,
            # а не по первым попавшимся (важно при LIMIT 100000 на больших каталогах).
            min_count = max(1, int(total_sku * SCHEMA_KEY_MIN_SHARE))
            async with db.execute("""
                SELECT je.key, COUNT(*) AS cnt
                FROM (
                    SELECT attributes
                    FROM products
                    WHERE attributes LIKE '{%'
                    ORDER BY last_updated DESC
                    LIMIT 100000
                ) p,
                json_each(p.attributes) AS je
                GROUP BY je.key
                HAVING cnt >= ?
                ORDER BY cnt DESC
            """, (min_count,)) as cur:
                key_rows = await cur.fetchall()

            # FIX #2: row[1] — уже агрегированный count из GROUP BY
            schema_keys = [row[0] for row in key_rows if row[0]]

            # 5. Язык магазина
            async with db.execute(
                "SELECT title FROM products WHERE title IS NOT NULL LIMIT 15"
            ) as cur:
                title_rows = await cur.fetchall()

            language = detect_language_from_titles(
                [row[0] for row in title_rows if row[0]]
            )

            # 6. Timestamp последнего обновления
            async with db.execute(
                "SELECT MAX(last_updated) FROM products WHERE last_updated IS NOT NULL"
            ) as cur:
                ts_row = await cur.fetchone()
                last_data_update = (
                    ts_row[0]
                    if ts_row and ts_row[0]
                    else datetime.datetime.now(datetime.timezone.utc).isoformat()
                )

            return {
                "total_sku":             total_sku,
                "store_type":            store_type,
                "main_category":         main_category,
                "category_distribution": category_distribution,
                "expertise_fields":      list(category_distribution.keys()),
                "brand_matrix": {
                    "total_brands": total_brands,
                    "top_brands":   top_brands,
                },
                "price_analytics": {
                    "min": price_min,
                    "max": price_max,
                    "avg": price_avg,
                },
                "schema_keys":      schema_keys,
                "language":         language,
                "last_data_update": last_data_update,
                "ai_welcome_message": "",  # заполняется в load_or_build
            }

        except Exception as e:
            logger.error(f"[{self._slug}] _compute_db_stats error: {e}", exc_info=True)
            return self._get_empty_profile()

    # ── Чтение / запись store_meta ────────────────────────────

    async def _load_from_store_meta(
        self, db: aiosqlite.Connection
    ) -> Optional[Dict[str, Any]]:
        """
        Читает все ключи из store_meta и собирает профиль.
        Принимает уже открытое соединение.
        Возвращает None если таблица пуста или недоступна.

        FIX #5: safe parse — guard на None после json.loads,
        float(None)/int(None) больше не падают.
        """
        try:
            async with db.execute("SELECT key, value FROM store_meta") as cur:
                rows = await cur.fetchall()

            if not rows:
                return None

            raw: Dict[str, str] = {row[0]: row[1] for row in rows}

            def _parse(key: str, default: Any) -> Any:
                val = raw.get(key)
                if val is None:
                    return default
                try:
                    parsed = json.loads(val)
                    # FIX #5: json.loads("null") → None, защищаем от этого
                    if parsed is None:
                        return default
                    return parsed
                except (json.JSONDecodeError, ValueError):
                    # Строка не является JSON — возвращаем как есть
                    return val

            def _safe_float(key: str, default: float) -> float:
                val = _parse(key, default)
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return default

            def _safe_int(key: str, default: int) -> int:
                val = _parse(key, default)
                try:
                    return int(val)
                except (TypeError, ValueError):
                    return default

            def _safe_list(key: str) -> List:
                val = _parse(key, [])
                return val if isinstance(val, list) else []

            def _safe_dict(key: str) -> Dict:
                val = _parse(key, {})
                return val if isinstance(val, dict) else {}

            return {
                "total_sku":             _safe_int("total_sku", 0),
                "store_type":            _parse("store_type", "multi_category"),
                "main_category":         _parse("main_category", "none"),
                "category_distribution": _safe_dict("category_distribution"),
                "expertise_fields":      _safe_list("expertise_fields"),
                "brand_matrix": {
                    "total_brands": _safe_int("total_brands", 0),
                    "top_brands":   _safe_list("top_brands"),
                },
                "price_analytics": {
                    "min": _safe_float("price_min", 0.0),
                    "max": _safe_float("price_max", 0.0),
                    "avg": _safe_float("price_avg", 0.0),
                },
                "schema_keys":        _safe_list("schema_keys"),
                "language":           _parse("language", "Ukrainian"),
                "last_data_update":   raw.get("last_data_update", ""),
                "ai_welcome_message": raw.get("ai_welcome_message", ""),
                "intent_mapping":     _safe_dict("intent_mapping"),
            }

        except aiosqlite.OperationalError as e:
            if "no such table" in str(e).lower():
                return None
            logger.error(f"[{self._slug}] _load_from_store_meta OperationalError: {e}")
            return None
        except Exception as e:
            logger.error(f"[{self._slug}] _load_from_store_meta error: {e}")
            return None

    async def _save_to_store_meta(
        self, db: aiosqlite.Connection, profile: Dict[str, Any]
    ) -> None:
        """
        Сохраняет профиль в store_meta через INSERT OR REPLACE.
        Принимает уже открытое соединение.
        Каждый ключ — отдельная строка. JSON-поля сериализуются.
        """
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        def _serialize(val: Any) -> str:
            if isinstance(val, str):
                return val
            return json.dumps(val, ensure_ascii=False)

        brand_matrix    = profile.get("brand_matrix", {})
        price_analytics = profile.get("price_analytics", {})

        rows = [
            ("total_sku",             _serialize(profile.get("total_sku", 0)),               now),
            ("store_type",            _serialize(profile.get("store_type", "")),              now),
            ("main_category",         _serialize(profile.get("main_category", "")),           now),
            ("category_distribution", _serialize(profile.get("category_distribution", {})),   now),
            ("expertise_fields",      _serialize(profile.get("expertise_fields", [])),        now),
            ("top_brands",            _serialize(brand_matrix.get("top_brands", [])),         now),
            ("total_brands",          _serialize(brand_matrix.get("total_brands", 0)),        now),
            ("price_min",             _serialize(price_analytics.get("min", 0)),              now),
            ("price_max",             _serialize(price_analytics.get("max", 0)),              now),
            ("price_avg",             _serialize(price_analytics.get("avg", 0)),              now),
            ("schema_keys",           _serialize(profile.get("schema_keys", [])),             now),
            ("language",              _serialize(profile.get("language", "Ukrainian")),       now),
            ("last_data_update",      profile.get("last_data_update", now),                   now),
            ("ai_welcome_message",    profile.get("ai_welcome_message", ""),                  now),
            ("intent_mapping",        _serialize(profile.get("intent_mapping", {})),           now),
        ]

        try:
            await db.executemany(
                "INSERT OR REPLACE INTO store_meta (key, value, updated_at) "
                "VALUES (?, ?, ?)",
                rows,
            )
            await db.commit()
            logger.info(
                f"[{self._slug}] Profile saved to store_meta ({len(rows)} keys)."
            )
        except Exception as e:
            logger.error(f"[{self._slug}] _save_to_store_meta error: {e}", exc_info=True)

    # ── LLM-персона ───────────────────────────────────────────

    async def _generate_ai_persona(self, stats: Dict[str, Any]) -> str:
        """Генерирует приветствие бота через LLM. Без изменений относительно v7.2.0."""
        try:
            real_name = "Наш магазин"
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    real_name = config.get("real_name", real_name)

            categories  = stats.get("expertise_fields", [])
            cat_context = ", ".join(categories[:3]) if categories else "товари"
            schema_keys = ", ".join(stats.get("schema_keys", [])[:5])

            prompt = (
                f"Напиши привітання для Telegram-бота магазину '{real_name}'. "
                f"Ми спеціалізуємось на: {cat_context}. "
                f"Ти — професійний консультант. Ти можеш шукати товари за такими "
                f"параметрами як {schema_keys}. "
                f"Скажи, що можеш підібрати ідеальний варіант за характеристиками. "
                f"Стиль: експертний, привітний. Мова: Ukrainian. 3 речення."
            )

            client, model = await self.selector.get_heavy()
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"[{self._slug}] LLM Persona generation failed: {e}")
            return "Вітаємо! Я ваш персональний консультант. Чим можу допомогти?"

    # ── Вспомогательные методы ────────────────────────────────

    @staticmethod
    async def _apply_pragma(db: aiosqlite.Connection) -> None:
        """Базовые PRAGMA для каждого соединения."""
        await db.execute("PRAGMA journal_mode = WAL")
        await db.execute("PRAGMA synchronous = NORMAL")
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute("PRAGMA busy_timeout = 10000")

    def _load_json_fallback(self) -> Dict[str, Any]:
        """
        Читает store_profile.json как bootstrap fallback.
        Используется если products.db недоступна или ещё не создана.
        """
        if not os.path.exists(self.profile_path):
            return self._get_empty_profile()
        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            profile = data.get("profile", data)
            if isinstance(profile, dict) and profile:
                logger.info(f"[{self._slug}] Loaded profile from JSON fallback.")
                return profile
        except Exception as e:
            logger.error(f"[{self._slug}] JSON fallback load error: {e}")
        return self._get_empty_profile()

    @staticmethod
    def _get_empty_profile() -> Dict[str, Any]:
        return {
            "total_sku":             0,
            "store_type":            "multi_category",
            "main_category":         "none",
            "category_distribution": {},
            "expertise_fields":      [],
            "ai_welcome_message":    "",
            "schema_keys":           [],
            "language":              "Ukrainian",
            "brand_matrix":          {"total_brands": 0, "top_brands": []},
            "price_analytics":       {"min": 0, "max": 0, "avg": 0},
            "last_data_update":      "",
        }

    def _parse_price(self, price: Any) -> Optional[float]:
        """Оставлен для обратной совместимости с внешним кодом."""
        if isinstance(price, (int, float)):
            return float(price)
        if isinstance(price, str):
            clean = re.sub(r"[^\d.,]", "", price.replace("\xa0", ""))
            if "," in clean and "." in clean:
                clean = clean.replace(",", "")
            elif "," in clean:
                clean = clean.replace(",", ".")
            try:
                return float(clean)
            except ValueError:
                return None
        return None