# -*- coding: utf-8 -*-
# /root/ukrsell_v4/core/dialog_manager.py v7.7.5

import json
import os
import asyncio
import re
import time
import logging
import aiosqlite
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union

from core.logger import logger, log_event
from core.intelligence import (
    safe_extract_json,
    semantic_guard,
    entity_filter,
    merge_followup,
    deduplicate_products,
)

_ALLOWED_ENTITY_KEYS = frozenset({"category", "brand", "price_limit", "properties"})

# Дефолтный маппинг категорий — используется если нет intent_hints.json
_DEFAULT_CATEGORY_MAP = {
    "шлея": "walking",
    "нашийник": "walking",
    "повідець": "walking",
    "рулетка": "walking",
    "светр": "apparel",
    "куртка": "apparel",
    "комбінезон": "apparel",
    "одяг": "apparel",
    "іграшка": "toys",
    "м'яч": "toys",
    "корм": "feeding",
    "миска": "feeding",
    "поїлка": "feeding",
    "лежанка": "beds & furniture",
    "лежак": "beds & furniture",
    "будиночок": "beds & furniture",
    "кігтеріз": "grooming",
    "щітка": "grooming",
    "шампунь": "grooming",
    "ліки": "meds",
    "вітаміни": "meds",
    "таблетки": "meds",
    "краплі": "meds",
    "антипаразитарний": "meds",
    "противоблошиний": "meds",
    "глисти": "meds",
    "антигельмінтний": "meds",
}


class DialogManager:
    r"""Intelligent Sales & Consultation Manager v7.7.4."""

    def __init__(self, ctx, llm_selector):
        self.ctx         = ctx
        self.selector    = llm_selector
        self.language    = getattr(ctx, 'language', 'Ukrainian')
        self.slug        = getattr(ctx, 'slug', 'Store')
        self.base_path   = getattr(ctx, 'base_path', '/root/ukrsell_v4')
        self.patch_path  = os.path.join(self.base_path, "fsm_soft_patch.json")
        self.session_db_path = os.path.join(self.base_path, "sessions.db")

        self._intent_cache: Dict[str, Dict] = {}
        self._intent_hints = self._load_intent_hints()

        self._init_db()
        logger.info(f"✅ [DM_INIT] DialogManager v7.7.5 Active. System: {self.slug}")

    # ── DB Init ───────────────────────────────────────────────────────────────

    def _init_db(self):
        try:
            conn   = sqlite3.connect(self.session_db_path, timeout=10.0)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id         INTEGER  PRIMARY KEY AUTOINCREMENT,
                    chat_id    TEXT     NOT NULL,
                    role       TEXT     NOT NULL,
                    content    TEXT     NOT NULL,
                    session_id TEXT,
                    timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            try:
                cursor.execute("ALTER TABLE chat_history ADD COLUMN session_id TEXT")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS intent_cache (
                    chat_id     TEXT    PRIMARY KEY,
                    intent_json TEXT    NOT NULL,
                    ts          REAL    NOT NULL,
                    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"[{self.slug}] Database Init Error: {e}")

    # ── Config ────────────────────────────────────────────────────────────────

    def _load_intent_hints(self) -> dict:
        hints_path = os.path.join(self.base_path, "intent_hints.json")
        try:
            if os.path.exists(hints_path):
                with open(hints_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ [DM_INIT] intent_hints.json not found or invalid: {e}")
        return {}

    def get_negative_examples(self) -> list:
        try:
            if os.path.exists(self.patch_path):
                with open(self.patch_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("troll_patterns", [])
        except Exception as e:
            logger.error(f"❌ [DM_ERROR] Failed to load negative patterns: {e}")
        return []

    # ── Intent Cache ──────────────────────────────────────────────────────────

    def _cache_get(self, chat_id: str) -> Optional[Dict]:
        key = str(chat_id)
        now = time.time()
        ttl = 2700

        cached = self._intent_cache.get(key)
        if cached and now - cached.get("ts", 0) < ttl:
            return cached

        try:
            conn = sqlite3.connect(self.session_db_path, timeout=5.0)
            row  = conn.execute(
                "SELECT intent_json, ts FROM intent_cache WHERE chat_id = ?", (key,)
            ).fetchone()
            conn.close()
            if row and now - row[1] < ttl:
                intent = json.loads(row[0])
                entry  = {"ts": row[1], "intent": intent}
                self._intent_cache[key] = entry
                return entry
        except Exception as e:
            logger.debug(f"[{self.slug}] intent_cache L2 read error: {e}")

        return None

    def _cache_set(self, chat_id: str, intent: dict) -> None:
        key   = str(chat_id)
        now   = time.time()
        entry = {"ts": now, "intent": intent}
        self._intent_cache[key] = entry

        try:
            conn = sqlite3.connect(self.session_db_path, timeout=5.0)
            conn.execute(
                "INSERT OR REPLACE INTO intent_cache (chat_id, intent_json, ts) "
                "VALUES (?, ?, ?)",
                (key, json.dumps(intent, ensure_ascii=False), now),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"[{self.slug}] intent_cache L2 write error: {e}")

    # ── Entity normalization ──────────────────────────────────────────────────

    @staticmethod
    def _normalize_entities(entities: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(entities, dict):
            return {}

        normalized: Dict[str, Any] = {}
        extra: Dict[str, Any]      = {}

        for key, val in entities.items():
            if key in _ALLOWED_ENTITY_KEYS:
                normalized[key] = val
            else:
                extra[key] = val

        if extra:
            existing_props = normalized.get("properties", {})
            if not isinstance(existing_props, dict):
                existing_props = {}
            existing_props.update(extra)
            normalized["properties"] = existing_props

        if "properties" not in normalized:
            normalized["properties"] = {}

        return normalized

    # ── Chat history ──────────────────────────────────────────────────────────

    async def get_chat_context(self, chat_id: str, minutes: int = 45) -> str:
        try:
            if not os.path.exists(self.session_db_path):
                logger.warning(f"⚠️ [DB_TRACE] Session DB missing at {self.session_db_path}")
                return ""

            time_threshold = (datetime.now() - timedelta(minutes=minutes)).strftime(
                '%Y-%m-%d %H:%M:%S'
            )
            logger.debug(f"📅 [DB_TRACE] Fetching history for {chat_id} since {time_threshold}")

            async with aiosqlite.connect(self.session_db_path, timeout=10.0) as db:
                async with db.execute(
                    """
                    SELECT role, content FROM chat_history
                    WHERE chat_id = ? AND timestamp > ?
                    ORDER BY timestamp DESC LIMIT 7
                    """,
                    (str(chat_id), time_threshold),
                ) as cursor:
                    rows = await cursor.fetchall()

            if not rows:
                logger.debug(f"ℹ️ [DB_TRACE] No history found for chat_id: {chat_id}")
                return ""

            history = []
            for row in reversed(rows):
                label = "Покупець" if row[0] == "user" else "Консультант"
                if self.language != "Ukrainian":
                    label = "Buyer" if row[0] == "user" else "Consultant"
                history.append(f"{label}: {row[1]}")

            full_context = "\n".join(history)
            logger.debug(f"📜 [DB_TRACE] Retrieved {len(rows)} messages. Context snippet: {full_context[:100]}...")
            return full_context

        except Exception as e:
            logger.error(f"❌ [DM_ERROR] SQL Context retrieval error: {e}")
            return ""

    async def save_history(
        self,
        chat_id: str,
        role: str,
        content: str,
        session_id: Optional[str] = None,
    ):
        try:
            async with aiosqlite.connect(self.session_db_path, timeout=10.0) as db:
                await db.execute(
                    """
                    INSERT INTO chat_history (chat_id, role, content, session_id, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(chat_id),
                        role,
                        content,
                        session_id,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    ),
                )
                await db.commit()
            logger.debug(f"💾 [DB_TRACE] Saved {role} message for {chat_id} (session={session_id})")
        except Exception as e:
            logger.error(f"[{self.slug}] History Save Error: {e}")

    def record_troll_pattern(self, user_text: str):
        try:
            data: Dict[str, Any] = {"troll_patterns": [], "fsm_errors": []}
            if os.path.exists(self.patch_path):
                with open(self.patch_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except Exception:
                        pass
            clean_text = str(user_text).lower().strip()
            if clean_text not in data.get("troll_patterns", []):
                data.setdefault("troll_patterns", []).append(clean_text)
                data.setdefault("fsm_errors", []).append({
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "request":   user_text,
                    "type":      "negative_example",
                })
                with open(self.patch_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.warning(f"⚠️ [TROLL_DETECTED] Pattern saved: {clean_text}")
        except Exception as e:
            logger.error(f"❌ [DM_ERROR] Failed to record troll pattern: {e}")

    # ── Prompt building ───────────────────────────────────────────────────────

    def _build_troll_hint(self) -> str:
        examples = self._intent_hints.get("troll_examples", [])
        if examples:
            joined = '", "'.join(examples[:4])
            return 'Приклади: "' + joined + '". Якщо є агресія або негатив — TROLL, навіть без матів.'
        return "Якщо є агресія або негатив — TROLL, даже без матів."

    def _build_brand_ignore_hint(self) -> str:
        ignore = self._intent_hints.get("brand_ignore", [])
        if ignore:
            examples = ", ".join(ignore[:5])
            return "Уточнення (" + examples + " тощо) — НЕ бренд, brand=null."
        return "Уточнення типу \"великий\", \"дитячий\" — НЕ бренд, brand=null."

    def _build_category_hint(self) -> str:
        SEP   = "\n"
        parts = [
            "- \"category\": тип товару. Правила:",
            "  1. Явна назва товару — використовуй дослівно як category, навіть якщо є уточнення.\n"
            "      \"X для Y\" -> category=\"X\". Наприклад: \"нашийник для лабрадора\" -> category=\"нашийник\".\n",
        ]
        fuzzy = self._intent_hints.get("fuzzy_mappings", [])
        if fuzzy:
            parts.append("  2. Нечіткий образ — визнач з контексту:")
            for m in fuzzy:
                parts.append("      \"" + m["pattern"] + "\" -> " + m["category"])
        else:
            parts.append("  2. Нечіткий образ — виведи категорію з контексту.")
        parts.append("  3. Неможливо визначити -> category=null.")
        return SEP.join(parts) + SEP

    def _build_category_mapping_hint(self) -> str:
        # Читаем маппинг из intent_hints.json, fallback на дефолтный
        mapping = self._intent_hints.get("category_mapping", _DEFAULT_CATEGORY_MAP)
        if not mapping:
            return ""
        lines = ["СТРОГИЙ СЛОВНИК КАТЕГОРІЙ (використовуй ТІЛЬКИ ці значення для category):"]
        for keyword, category in list(mapping.items())[:20]:
            lines.append(f'  "{keyword}" → category="{category}"')
        lines.append(
            "Якщо запит не підходить до жодного ключового слова — category=null.\n"
            "STRICT MODE: будь-яке значення category НЕ зі словника вище = null."
        )
        return "\n".join(lines)

    def _build_intent_prompt(
        self, negative_examples: list, chat_context: str, model_name: str
    ) -> str:
        schema_keys = getattr(self.ctx, "schema_keys", [])
        schema_hint = (
            "Доступні поля фільтрації (entities.properties): " + ", ".join(schema_keys)
            if schema_keys
            else ""
        )

        expertise    = getattr(self.ctx, 'profile', {}).get('expertise_fields', [])
        cat_examples = (
            ", ".join(f'"{c}"' for c in expertise[:6])
            if expertise
            else '"іграшка", "лежак", "одяг", "корм", "нашийник"'
        )

        example_cat = expertise[0] if expertise else "лежанка"

        # Few-shot пример с правильной структурой JSON
        few_shot_example = (
            "\nОБОВ'ЯЗКОВИЙ ЗРАЗОК JSON (копіюй структуру точно):\n"
            "{\n"
            "  \"action\": \"SEARCH\",\n"
            f"  \"reason\": \"Запит про {example_cat}\",\n"
            "  \"entities\": {\n"
            f"    \"category\": \"{example_cat}\",\n"
            "    \"brand\": null,\n"
            "    \"price_limit\": null,\n"
            "    \"properties\": {\n"
            "      \"Тварина\": \"кіт\"\n"
            "    }\n"
            "  },\n"
            "  \"temperature\": \"HARD_SEARCH\",\n"
            "  \"language\": \"Ukrainian\"\n"
            "}\n"
            "КРИТИЧНО: \"category\" = назва товару (" + cat_examples + ").\n"
            "ЗАБОРОНЕНО в \"category\": назви полів схеми ("
            + ", ".join(f'"{k}"' for k in schema_keys[:4])
            + ") — це фільтри, а не категорії.\n"
            "Всі характеристики (тварина, розмір, колір) — ТІЛЬКИ всередині \"properties\".\n"
        )

        # Строгий маппинг категорий — ключевое изменение v7.7.4
        category_mapping_hint = self._build_category_mapping_hint()

        slug_line = "\nТи — суворий бібліотекар-класифікатор у магазині \"" + self.slug + "\".\n"
        role_hint = (
            "НЕ аналізуй, НЕ інтерпретуй вільно. "
            "Тільки заповнюй поля за словником вище.\n"
        )
        obov = "ОБОВ'ЯЗКОВІ ПОЛЯ В JSON:\n"

        header = (
            slug_line
            + role_hint
            + "Завдання: перетворити запит користувача на структурований JSON.\n"
            + "\nІСТОРІЯ ДІАЛОГУ:\n" + chat_context + "\n"
            + "\nПРИКЛАДИ НЕГАТИВУ: " + str(negative_examples[:5]) + "\n"
            + ("\n" + schema_hint + "\n" if schema_hint else "")
            + ("\n" + category_mapping_hint + "\n" if category_mapping_hint else "")
            + few_shot_example
            + "\n" + obov
            + "1. \"action\": SEARCH (пошук), CONSULT (консультація), CHAT (флуд), TROLL (атака).\n"
            + "2. \"temperature\": HARD_SEARCH (конкретна покупка), SOFT_ADVISORY (порада), VIBE_CHECK (флуд).\n"
            + "3. \"entities\": {\"category\": str, \"brand\": str, \"price_limit\": int, \"properties\": dict}.\n"
            + "4. \"language\": мова запиту.\n"
            + "5. \"reason\": коротке пояснення логіки (string).\n"
        )
        rules = (
            "\nПРАВИЛА ДЛЯ \"action\":\n"
            + "- SEARCH: будь-який запит про товар, в т.ч. питальна форма (\"є куртки?\").\n"
            + "- TROLL: образи, мат, агресія.\n"
            + "  " + self._build_troll_hint() + "\n"
            + "- CHAT: беззмістовний текст, вітання.\n"
            + "- CONSULT: загальні питання БЕЗ конкретного товару (\"що порадите для кота?\").\n"
            + "\nПРАВИЛА ДЛЯ \"entities\":\n"
            + self._build_category_hint()
            + "- \"brand\": ТІЛЬКИ власні назви брендів виробника.\n"
            + "  " + self._build_brand_ignore_hint() + "\n"
            + "- \"price_limit\": числова межа ціни в грн або null.\n"
            + "- \"properties\": dict з додатковими характеристиками (Тварина, розмір тощо).\n"
            + "\nVERY IMPORTANT: ВІДПОВІДАЙ ВИКЛЮЧНО JSON. NO MARKDOWN. NO BACKTICKS. NO PREAMBLE.\n"
        )
        return header + rules

    # ── Fallback intent ───────────────────────────────────────────────────────

    def _get_fallback_intent(self, text: str) -> dict:
        r"""Fallback regex extraction при падении LLM."""
        logger.info(f"🛠️ [FALLBACK] Universal regex extraction for: {str(text)[:50]}...")
        text_low = str(text).lower()

        price_limit: Optional[int] = None

        price_with_marker = re.search(
            r'(\d{3,6})\s*(грн|₴|uah|грн\.|гривен|гривень)',
            text_low,
        )
        price_with_context = re.search(
            r'(?:до|за|не\s*більше|менше\s*ніж|не\s*дорожче)\s+(\d{3,6})',
            text_low,
        )

        if price_with_marker:
            candidate = int(price_with_marker.group(1))
            if candidate >= 100:
                price_limit = candidate
        elif price_with_context:
            candidate = int(price_with_context.group(1))
            if candidate >= 100:
                price_limit = candidate
        else:
            standalone = re.search(
                r'(?<![a-zа-яёіїєґ])(\d{3,6})(?![a-zа-яёіїєґ])',
                text_low,
            )
            if standalone:
                candidate = int(standalone.group(1))
                if candidate >= 100:
                    price_limit = candidate

        category = None
        for key in getattr(self.ctx, 'schema_keys', []):
            if key.lower() in text_low:
                category = key
                break

        brand      = None
        profile    = getattr(self.ctx, 'profile', {})
        top_brands = profile.get('brand_matrix', {}).get('top_brands', [])
        for b in top_brands:
            if b and b.lower() in text_low:
                brand = b
                break

        return {
            "action":      "SEARCH",
            "temperature": "HARD_SEARCH",
            "entities": {
                "category":    category,
                "brand":       brand,
                "price_limit": price_limit,
                "properties":  {},
            },
            "language": getattr(self.ctx, 'language', 'Ukrainian'),
        }

    # ── Intent analysis ───────────────────────────────────────────────────────

    async def analyze_intent(
        self, user_text: str, chat_id: str, session_id: Optional[str] = None
    ) -> dict:
        try:
            user_text = str(user_text)
            await asyncio.wait_for(self.selector.ensure_ready(), timeout=5.0)
            chat_context = await self.get_chat_context(chat_id)
            patterns     = self.get_negative_examples()

            if user_text.lower().strip() in patterns:
                logger.warning(f"🛡️ [INTENT_TRACE] Fast-Reject: Known Troll Pattern for {chat_id}")
                return {
                    "action":      "TROLL",
                    "temperature": "VIBE_CHECK",
                    "entities":    {"properties": {}},
                    "language":    self.language,
                }

            _cached      = self._cache_get(str(chat_id))
            _is_followup = bool(_cached and time.time() - _cached.get("ts", 0) < 2700)

            tier_hint  = "heavy" if (len(user_text) > 80 or _is_followup) else "light"
            result_obj = await (
                self.selector.get_heavy() if tier_hint == "heavy"
                else self.selector.get_light()
            )

            if isinstance(result_obj, tuple):
                client, model = result_obj
            else:
                client = result_obj
                model  = getattr(result_obj, 'model_name', None)

            prompt = self._build_intent_prompt(patterns, chat_context, model)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"\n--- [PROMPT_OUT] Chat: {chat_id} | Session: {session_id} ---\n"
                    f"{prompt}\n"
                    f"User query: {user_text}\n"
                    f"--------------------------------------------------"
                )

            t_start_llm = time.perf_counter()
            response    = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user",   "content": user_text},
                ],
                temperature=0.0,  # v7.7.4: 0.1 → 0.0 для детерминизма категорий
                max_tokens=150,
            )

            content = getattr(response.choices[0].message, 'content', '{}')

            # v7.7.4: принудительная очистка markdown-оберток до парсинга
            content = re.sub(r'^```(?:json)?\s*', '', content.strip(), flags=re.IGNORECASE)
            content = re.sub(r'\s*```$', '', content.strip())
            content = content.strip()

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"\n--- [RESPONSE_IN] Chat: {chat_id} | Session: {session_id} ---\n"
                    f"{content}\n"
                    f"-------------------------------------------------"
                )

            raw_intent = safe_extract_json(content)

            # Нормализация entities — зайві ключі → properties
            raw_entities = raw_intent.get("entities", {})
            if isinstance(raw_entities, dict):
                raw_intent["entities"] = self._normalize_entities(raw_entities)

            schema_keys = set(getattr(self.ctx, "schema_keys", []))
            intent_ents = raw_intent.get("entities", {})
            top_level   = {k for k in intent_ents if k not in _ALLOWED_ENTITY_KEYS}
            if top_level:
                logger.warning(f"🚨 [KEY_MISMATCH] Unexpected top-level entity keys after normalize: {top_level}")

            prev_intent  = _cached["intent"] if _is_followup else {"entities": {}}
            category_map = getattr(self.ctx, 'category_map', {})
            result       = merge_followup(prev_intent, raw_intent, category_map=category_map)

            if "temperature" not in result:
                result["temperature"] = (
                    "HARD_SEARCH" if result.get("action") == "SEARCH" else "SOFT_ADVISORY"
                )

            if "entities" not in result:
                result["entities"] = {}
            if "properties" not in result.get("entities", {}):
                result["entities"]["properties"] = {}

            self._cache_set(str(chat_id), result)

            ms_spent = round((time.perf_counter() - t_start_llm) * 1000, 1)
            log_event("INTENT_RESULT", {
                "slug":        self.slug,
                "chat_id":     str(chat_id),
                "action":      result.get("action"),
                "temperature": result.get("temperature"),
                "entities":    result.get("entities"),
                "ms":          ms_spent,
                "model":       model,
                "reason":      result.get("reason", "N/A"),
            }, session_id=session_id)

            return result

        except Exception as e:
            logger.error(f"💥 [DM_CRITICAL] Intent analysis failure: {e}", exc_info=True)
            return self._get_fallback_intent(user_text)

    # ── Main pipeline ─────────────────────────────────────────────────────────

    async def process_query(
        self,
        user_query: str,
        chat_id: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        t_total_start = time.perf_counter()
        user_query    = str(user_query)

        logger.info(f"🚀 [QUERY_START] Chat: {chat_id} | Session: {session_id} | Query: {user_query}")

        self.ctx.current_session_id = session_id

        intent         = await self.analyze_intent(user_query, chat_id, session_id=session_id)
        search_results = {"products": [], "status": "SKIPPED"}

        if intent.get("action") in ["SEARCH", "CONSULT"]:
            retrieval = getattr(self.ctx, 'retrieval', None)
            if retrieval:
                logger.info(f"🔍 [RETRIEVAL_STEP] Triggering search with entities: {intent.get('entities')}")
                search_results = await retrieval.search(user_query, entities=intent)
            else:
                logger.error(f"[{self.slug}] Retrieval index missing.")

        validated_products = await self.process_search_pipeline(
            chat_id=chat_id,
            search_response=search_results,
            intent=intent,
        )

        total_ms = round((time.perf_counter() - t_total_start) * 1000, 1)

        log_event("QUERY_PROCESSED", {
            "slug":           self.slug,
            "chat_id":        str(chat_id),
            "intent":         intent.get("action"),
            "temp":           intent.get("temperature"),
            "products_count": len(validated_products),
            "total_ms":       total_ms,
            "search_status":  search_results.get("status", "UNKNOWN"),
        }, session_id=session_id)

        asyncio.create_task(
            self.save_history(chat_id, "user", user_query, session_id=session_id)
        )

        self.ctx.last_intent = intent

        return {
            "intent":        intent,
            "products":      validated_products,
            "search_status": search_results.get("status", "UNKNOWN"),
            "status":        "SUCCESS",
            "total_ms":      total_ms,
        }

    async def process_search_pipeline(
        self,
        chat_id: str,
        search_response: Optional[Union[Dict, List]] = None,
        intent: Dict = None,
        top_k: int = 5,
        raw_products: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        status              = "SUCCESS"
        products_to_process: List[Dict] = []
        is_empty            = False

        if raw_products is not None:
            products_to_process = raw_products
            is_empty            = not raw_products
        elif isinstance(search_response, dict):
            status              = search_response.get("status", "SUCCESS")
            products_to_process = search_response.get("products", [])
            is_empty            = search_response.get("is_empty", False)
        elif isinstance(search_response, list):
            products_to_process = search_response
            is_empty            = not search_response
        else:
            is_empty = True

        if status == "LOW_CONFIDENCE" or is_empty:
            logger.info(f"🚫 [PIPELINE_TRACE] Skipping processing: status={status}, is_empty={is_empty}")
            return []

        try:
            sem_threshold   = getattr(self.ctx, 'threshold_semantic', 0.55)
            guarded         = semantic_guard(products_to_process, threshold=sem_threshold)
            retrieval_layer = getattr(self.ctx, 'retrieval', None)
            intent_mapping  = (
                getattr(retrieval_layer, 'intent_mapping', {})
                if retrieval_layer
                else {}
            )

            filtered = entity_filter(
                guarded if guarded else products_to_process,
                intent,
                intent_mapping=intent_mapping,
                category_map=getattr(self.ctx, 'category_map', {}),
            )

            final_list = deduplicate_products(filtered, top_k=top_k)
            logger.info(f"🧪 [PIPELINE_TRACE] Pipeline complete: {len(products_to_process)} -> {len(final_list)} products")
            return final_list

        except Exception as e:
            logger.error(f"❌ [DM_ERROR] Search pipeline crash: {e}")
            return products_to_process[:top_k]

    def __repr__(self):
        return f"<DialogManager v7.7.5 slug={self.slug}>"


def get_version():
    return "7.7.5"