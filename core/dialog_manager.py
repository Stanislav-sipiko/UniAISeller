# /root/ukrsell_v4/core/dialog_manager.py v7.6.4
import json
import os
import asyncio
import re
import sqlite3
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union

# Импорты вспомогательных утилит для обработки данных
from core.logger import logger, log_event
from core.intelligence import (
    safe_extract_json, 
    semantic_guard, 
    entity_filter, 
    merge_followup, 
    deduplicate_products
)

class DialogManager:
    """
    Intelligent Sales & Consultation Manager v7.6.4.
    UNIVERSAL: Works with multiple store types (pets, phones, auto parts).
    STRICT: Zero Omission mode. Integrated with Confidence Gate v7.6.0.
    
    Changelog v7.6.4:
        - Fixed: Updated process_search_pipeline signature to match Kernel v7.8.4.
        - Added: Support for 'raw_products' argument to prevent TypeError.
        - Refactored: Internal logic to handle both raw lists and dict responses.
    """
    def __init__(self, ctx, llm_selector):
        self.ctx = ctx
        self.selector = llm_selector
        self.language = getattr(ctx, 'language', 'Ukrainian')
        self.slug = getattr(ctx, 'slug', 'Store')
        self.base_path = getattr(ctx, 'base_path', '/root/ukrsell_v4')
        self.patch_path = os.path.join(self.base_path, "fsm_soft_patch.json")
        self.session_db_path = os.path.join(self.base_path, "sessions.db")
        
        self._intent_cache = {}  # {chat_id: {"ts": float, "intent": dict}}
        self._intent_hints = self._load_intent_hints()
        
        # Инициализация БД сессий
        self._init_db()
        logger.info(f"✅ [DM_INIT] DialogManager v7.6.4 Active. System: {self.slug}")

    def _init_db(self):
        """Создание таблицы сессий, если она не существует."""
        try:
            conn = sqlite3.connect(self.session_db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    chat_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"[{self.slug}] Database Init Error: {e}")

    def _load_intent_hints(self) -> dict:
        """Загружает intent_hints.json из директории магазина."""
        hints_path = os.path.join(self.base_path, "intent_hints.json")
        try:
            if os.path.exists(hints_path):
                with open(hints_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data
        except Exception as e:
            logger.warning(f"⚠️ [DM_INIT] intent_hints.json not found or invalid: {e}")
        return {}

    def get_negative_examples(self) -> list:
        """Загрузка паттернов троллинга из патч-файла."""
        try:
            if os.path.exists(self.patch_path):
                with open(self.patch_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("troll_patterns", [])
        except Exception as e:
            logger.error(f"❌ [DM_ERROR] Failed to load negative patterns: {e}")
        return []

    def get_chat_context(self, chat_id: str, minutes: int = 45) -> str:
        """Извлечение истории сообщений из SQLite для поддержания контекста."""
        conn = None
        try:
            if not os.path.exists(self.session_db_path): 
                return ""
            
            conn = sqlite3.connect(self.session_db_path, timeout=5.0)
            cursor = conn.cursor()
            time_threshold = (datetime.now() - timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
            
            query = """
                SELECT role, content FROM chat_history 
                WHERE chat_id = ? AND timestamp > ?
                ORDER BY timestamp DESC LIMIT 7
            """
            cursor.execute(query, (str(chat_id), time_threshold))
            rows = cursor.fetchall()
            
            if not rows: 
                return ""
            
            history = []
            for row in reversed(rows):
                label = "Покупець" if row[0] == "user" else "Консультант"
                if self.language != "Ukrainian": 
                    label = "Buyer" if row[0] == "user" else "Consultant"
                history.append(f"{label}: {row[1]}")
            
            return "\n".join(history)
        except Exception as e:
            logger.error(f"❌ [DM_ERROR] SQL Context retrieval error: {e}")
            return ""
        finally:
            if conn: 
                conn.close()

    async def save_history(self, chat_id: str, role: str, content: str):
        """Сохранение сообщения в историю (совместимо с v7.6.0)."""
        try:
            conn = sqlite3.connect(self.session_db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO chat_history (chat_id, role, content, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (str(chat_id), role, content, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"[{self.slug}] History Save Error: {e}")

    def record_troll_pattern(self, user_text: str):
        """Логирование новых подозрительных запросов в базу паттернов."""
        try:
            data = {"troll_patterns": [], "fsm_errors": []}
            if os.path.exists(self.patch_path):
                with open(self.patch_path, 'r', encoding='utf-8') as f:
                    try: 
                        data = json.load(f)
                    except: 
                        pass
            
            clean_text = str(user_text).lower().strip()
            if clean_text not in data.get("troll_patterns", []):
                data.setdefault("troll_patterns", []).append(clean_text)
                data.setdefault("fsm_errors", []).append({
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "request": user_text, 
                    "type": "negative_example"
                })
                with open(self.patch_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.warning(f"⚠️ [TROLL_DETECTED] Pattern saved: {clean_text}")
        except Exception as e:
            logger.error(f"❌ [DM_ERROR] Failed to record troll pattern: {e}")

    def _build_troll_hint(self) -> str:
        examples = self._intent_hints.get("troll_examples", [])
        if examples:
            joined = '", "'.join(examples[:4])
            return 'Приклади: "' + joined + '". Якщо є агресія або негатив — TROLL, навіть без матів.'
        return "Якщо є агресія або негатив — TROLL, навіть без матів."

    def _build_brand_ignore_hint(self) -> str:
        ignore = self._intent_hints.get("brand_ignore", [])
        if ignore:
            examples = ", ".join(ignore[:5])
            return "Уточнення (" + examples + " тощо) — НЕ бренд, brand=null."
        return "Уточнення типу \"великий\", \"дитячий\" — НЕ бренд, brand=null."

    def _build_category_hint(self) -> str:
        SEP = "\n"
        parts = [
            "- \"category\": тип товару. Правила:",
            "  1. Явна назва товару — використовуй дослівно как category, навіть якщо есть уточнення.\n"
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

    def _build_intent_prompt(self, negative_examples: list, chat_context: str, model_name: str) -> str:
        schema_keys = getattr(self.ctx, "schema_keys", [])
        schema_hint = ("Доступні поля фільтрації: " + ", ".join(schema_keys)) if schema_keys else ""
        slug_line = "\nТвоя роль — AI-аналітик у магазині \"" + self.slug + "\".\n"
        obov = "ОБОВ\'ЯЗКОВІ ПОЛЯ В JSON:\n"
        header = (
            slug_line
            + "Твоє завдання: перетворити запит користувача на структурований JSON.\n"
            + "\nІСТОРІЯ ДІАЛОГУ:\n" + chat_context + "\n"
            + "\nПРИКЛАДИ НЕГАТИВУ: " + str(negative_examples[:5]) + "\n"
            + ("\n" + schema_hint + "\n" if schema_hint else "")
            + "\n" + obov
            + "1. \"action\": SEARCH (пошук), CONSULT (консультація), CHAT (флуд), TROLL (атака).\n"
            + "2. \"entities\": {\"category\": str, \"brand\": str, \"price_limit\": int}.\n"
            + "3. \"language\": мова запиту.\n"
        )
        rules = (
            "\nПРАВИЛА ДЛЯ \"action\":\n"
            + "- SEARCH: будь-який запит про товар, в т.ч. питальна форма (\"є куртки?\", \"чи є у вас?\").\n"
            + "  Якщо в запиті есть назва товару або категории — завжди SEARCH, навіть якщо це питання.\n"
            + "- TROLL: образи, мат, агресія, погрози на адресу магазину або співробітников.\n"
            + "  " + self._build_troll_hint() + "\n"
            + "- CHAT: беззмістовний текст, мусор, символи без сенсу.\n"
            + "- CONSULT: загальні питання БЕЗ конкретного товару.\n"
            + "\nПРАВИЛА ДЛЯ \"entities\":\n"
            + self._build_category_hint()
            + "- \"brand\": ТІЛЬКИ власные назви брендів виробника.\n"
            + "  " + self._build_brand_ignore_hint() + "\n"
            + "- \"price_limit\": числова межа ціни в грн або null.\n"
            + "\nВІДПОВІДАЙ ВИКЛЮЧНО JSON БЕЗ ЗАЙВОГО ТЕКСТУ.\n"
        )
        return header + rules

    def _get_fallback_intent(self, text: str) -> dict:
        logger.info(f"🛠️ [FALLBACK] Universal regex extraction for: {str(text)[:50]}...")
        text_low = str(text).lower()
        price_match = re.search(r'(\d{2,6})', text_low)
        
        category = None
        for key in getattr(self.ctx, 'schema_keys', []):
            if key.lower() in text_low:
                category = key
                break
        
        price_limit = int(price_match.group(1)) if price_match else None

        return {
            "action": "SEARCH",
            "entities": {
                "category": category,
                "brand": None,
                "price_limit": price_limit
            },
            "language": getattr(self.ctx, 'language', 'Ukrainian')
        }

    async def analyze_intent(self, user_text: str, chat_id: str, session_id: Optional[str] = None) -> dict:
        """Определение намерения пользователя с логированием сессии."""
        try:
            user_text = str(user_text) # Force string cast
            await asyncio.wait_for(self.selector.ensure_ready(), timeout=5.0)
            chat_context = self.get_chat_context(chat_id)
            patterns = self.get_negative_examples()
            
            if user_text.lower().strip() in patterns:
                return {"action": "TROLL", "entities": {}, "language": self.language}

            _cached = self._intent_cache.get(str(chat_id))
            _is_followup = bool(_cached and time.time() - _cached.get("ts", 0) < 2700)
            tier_hint = "heavy" if (len(user_text) > 80 or _is_followup) else "light"
            
            result_obj = await (self.selector.get_heavy() if tier_hint == "heavy" else self.selector.get_light())
            client, model = result_obj if isinstance(result_obj, tuple) else (result_obj, getattr(result_obj, 'model_name', None))

            prompt = self._build_intent_prompt(patterns, chat_context, model)
            
            t_start_llm = time.perf_counter()
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_text}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            content = getattr(response.choices[0].message, 'content', '{}')
            raw_intent = safe_extract_json(content)
            
            prev_intent = _cached["intent"] if (_is_followup) else {"entities": {}}
            category_map = getattr(self.ctx, 'category_map', {})
            result = merge_followup(prev_intent, raw_intent, category_map=category_map)

            self._intent_cache[str(chat_id)] = {"ts": time.time(), "intent": result}

            log_event("INTENT_RESULT", {
                "slug": self.slug,
                "action": result.get("action"),
                "entities": result.get("entities"),
                "ms": round((time.perf_counter() - t_start_llm) * 1000, 1)
            }, session_id=session_id)

            return result
        except Exception as e:
            logger.error(f"💥 [DM_CRITICAL] Intent analysis failure: {e}")
            return self._get_fallback_intent(user_text)

    async def process_query(self, user_query: str, chat_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Основной цикл обработки диалога v7.6.4."""
        user_query = str(user_query)
        # 1. Анализ интента
        intent = await self.analyze_intent(user_query, chat_id, session_id=session_id)
        
        # 2. Выполнение поиска
        search_results = {"products": [], "status": "SKIPPED"}
        if intent.get("action") == "SEARCH":
            retrieval = getattr(self.ctx, 'retrieval', None)
            if retrieval:
                search_results = await retrieval.search(user_query, intent=intent)
            else:
                asset_path = os.path.join(self.base_path, "stores", self.slug)
                logger.error(f"[{self.slug}] FAILED: Index not found at {asset_path}")

        # 3. Валидация через пайплайн
        validated_products = await self.process_search_pipeline(
            chat_id=chat_id, 
            search_response=search_results, 
            intent=intent
        )
        
        # 4. Синтез ответа (через Analyzer в ядре)
        analyzer = getattr(self.ctx, 'analyzer', None)
        response = await analyzer.synthesize_response(
            search_results={"products": validated_products, "status": search_results.get("status")},
            user_query=user_query,
            intent=intent,
            chat_context=self.get_chat_context(chat_id)
        )

        # 5. КРИТИЧЕСКОЕ ЛОГИРОВАНИЕ ФИНАЛЬНОГО ОТВЕТА
        log_event("FINAL_RESPONSE", {
            "slug": self.slug,
            "chat_id": str(chat_id),
            "text": response.get("text", "")[:250],
            "status": response.get("status", "SUCCESS")
        }, session_id=session_id)

        # 6. Сохранение истории
        await self.save_history(chat_id, "user", user_query)
        await self.save_history(chat_id, "assistant", response.get("text", ""))

        return response

    async def process_search_pipeline(
        self, 
        chat_id: str, 
        search_response: Optional[Union[Dict, List]] = None, 
        intent: Dict = None, 
        top_k: int = 5,
        raw_products: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Улучшенная обработка результатов с учетом Confidence Gate v7.6.0.
        v7.6.4: Совместимость с Kernel v7.8.4 (поддержка raw_products).
        """
        status = "SUCCESS"
        confidence = 1.0
        is_empty = False
        
        # Приоритет raw_products (если вызвано из Kernel v7.8.4)
        if raw_products is not None:
            products_to_process = raw_products
            is_empty = not raw_products
        # Поддержка классического search_response (Dict)
        elif isinstance(search_response, dict):
            status = search_response.get("status", "SUCCESS")
            products_to_process = search_response.get("products", [])
            is_empty = search_response.get("is_empty", False)
            confidence = search_response.get("confidence", 0.0)
        # Поддержка прямого списка (Fallback)
        elif isinstance(search_response, list):
            products_to_process = search_response
            is_empty = not search_response
        else:
            products_to_process = []
            is_empty = True

        if status == "LOW_CONFIDENCE" or is_empty:
            return []

        try:
            # 1. Семантическая защита (отсечение галлюцинаций FAISS)
            guarded = semantic_guard(products_to_process, threshold=0.55)
            
            # 2. Фильтрация по атрибутам (бренд, категория, цена)
            filtered = entity_filter(
                guarded if guarded else products_to_process, 
                intent,
                intent_mapping=getattr(getattr(self.ctx, 'retrieval', None), 'intent_mapping', {}),
                category_map=getattr(self.ctx, 'category_map', {}),
            )
            
            # 3. Дедупликация и обрезка до top_k
            return deduplicate_products(filtered, top_k=top_k)
            
        except Exception as e:
            logger.error(f"❌ [DM_ERROR] Search pipeline crash: {e}")
            return products_to_process[:top_k]