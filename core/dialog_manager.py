# /root/ukrsell_v4/core/dialog_manager.py v7.5.6
import json
import os
import asyncio
import re
import sqlite3
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

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
    Intelligent Sales & Consultation Manager v7.5.4.
    UNIVERSAL: Works with multiple store types (pets, phones, auto parts).
    OBJECTIVE: Async-safe LLM Selector integration + fully dynamic fallback.
    STRICT: Zero Omission mode.
    """
    def __init__(self, ctx, llm_selector):
        self.ctx = ctx
        self.selector = llm_selector
        self.language = getattr(ctx, 'language', 'Ukrainian')
        self.slug = getattr(ctx, 'slug', 'Store')
        self.base_path = getattr(ctx, 'base_path', '/root/ukrsell_v4')
        self.patch_path = os.path.join(self.base_path, "fsm_soft_patch.json")
        self.session_db_path = os.path.join(self.base_path, "sessions.db")
        
        self._intent_cache = {}
        self._intent_hints = self._load_intent_hints()
        print(f"✅ [DM_INIT] DialogManager v7.5.6 Active. System: {self.slug}")

    def _load_intent_hints(self) -> dict:
        """Загружает intent_hints.json из директории магазина."""
        hints_path = os.path.join(self.base_path, "intent_hints.json")
        try:
            if os.path.exists(hints_path):
                with open(hints_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"✅ [DM_INIT] intent_hints loaded: {len(data.get('fuzzy_mappings', []))} fuzzy, {len(data.get('troll_examples', []))} troll examples")
                    return data
        except Exception as e:
            print(f"⚠️ [DM_INIT] intent_hints.json not found or invalid: {e}")
        return {}

    def get_negative_examples(self) -> list:
        """Загрузка паттернов троллинга из патч-файла."""
        try:
            if os.path.exists(self.patch_path):
                with open(self.patch_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("troll_patterns", [])
        except Exception as e:
            print(f"❌ [DM_ERROR] Failed to load negative patterns: {e}")
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
            print(f"❌ [DM_ERROR] SQL Context retrieval error: {e}")
            return ""
        finally:
            if conn: 
                conn.close()

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
            
            clean_text = user_text.lower().strip()
            if clean_text not in data.get("troll_patterns", []):
                data.setdefault("troll_patterns", []).append(clean_text)
                data.setdefault("fsm_errors", []).append({
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "request": user_text, 
                    "type": "negative_example"
                })
                with open(self.patch_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"⚠️ [TROLL_DETECTED] Pattern saved: {clean_text}")
        except Exception as e:
            print(f"❌ [DM_ERROR] Failed to record troll pattern: {e}")

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
            "  1. Явна назва товару — використовуй дослівно як category, навіть якщо є уточнення.\n"
            "     \"X для Y\" -> category=\"X\". Наприклад: \"нашийник для лабрадора\" -> category=\"нашийник\".\n",
        ]
        fuzzy = self._intent_hints.get("fuzzy_mappings", [])
        if fuzzy:
            parts.append("  2. Нечіткий образ — визнач з контексту:")
            for m in fuzzy:
                parts.append("     \"" + m["pattern"] + "\" -> " + m["category"])
        else:
            parts.append("  2. Нечіткий образ — виведи категорію з контексту.")
        parts.append("  3. Неможливо визначити -> category=null.")
        return SEP.join(parts) + SEP

    def _build_intent_prompt(self, negative_examples: list, chat_context: str, model_name: str) -> str:
        """Формирование системного промпта для классификации интента с динамическими полями."""
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
            + "  Якщо в запиті є назва товару або категорії — завжди SEARCH, навіть якщо це питання.\n"
            + "- TROLL: образи, мат, агресія, погрози на адресу магазину або співробітників.\n"
            + "  " + self._build_troll_hint() + "\n"
            + "- CHAT: беззмістовний текст, мусор, символи без сенсу.\n"
            + "- CONSULT: загальні питання БЕЗ конкретного товару.\n"
            + "\nПРАВИЛА ДЛЯ \"entities\":\n"
            + self._build_category_hint()
            + "- \"brand\": ТІЛЬКИ власні назви брендів виробника.\n"
            + "  " + self._build_brand_ignore_hint() + "\n"
            + "- \"price_limit\": числова межа ціни в грн або null.\n"
            + "\nВІДПОВІДАЙ ВИКЛЮЧНО JSON БЕЗ ЗАЙВОГО ТЕКСТУ.\n"
        )
        return header + rules

    def _get_fallback_intent(self, text: str) -> dict:
        """Резервный метод извлечения интента через регулярные выражения, полностью универсальный."""
        print(f"🛠️ [FALLBACK] Universal regex extraction for: {text[:50]}...")
        text_low = text.lower()
        price_match = re.search(r'(\d{2,6})', text_low)
        
        # Динамическая категоризация через schema_keys
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

    async def analyze_intent(self, user_text: str, chat_id: str) -> dict:
        """Асинхронный цикл анализа намерений пользователя с динамическим fallback."""
        print(f"\n[DM_TRACE] >>> ENTRY: analyze_intent for Chat:{chat_id}")
        
        try:
            # ШАГ 1: Проверка готовности селектора
            print("[DM_TRACE] Step 1: Waiting for Selector readiness...")
            await asyncio.wait_for(self.selector.ensure_ready(), timeout=5.0)
            
            # ШАГ 2: Сбор контекста и негативных паттернов
            print("[DM_TRACE] Step 2: Extracting context and negative patterns...")
            chat_context = self.get_chat_context(chat_id)
            patterns = self.get_negative_examples()
            
            # Быстрая проверка на троллинг
            if user_text.lower().strip() in patterns:
                print("🛑 [DM_TRACE] Early Exit: Match found in TROLL_PATTERNS")
                return {"action": "TROLL", "entities": {}, "language": getattr(self.ctx, 'language', 'Ukrainian')}

            # ШАГ 3: Выбор модели
            print(f"[DM_TRACE] Step 3: Triggering Selector Tier Selection (Length: {len(user_text)})...")
            t_start_select = time.perf_counter()
            
            tier_hint = "heavy" if len(user_text) > 80 else "light"
            if tier_hint == "heavy":
                selection_coro = self.selector.get_heavy()
            else:
                selection_coro = self.selector.get_light()
            
            result_obj = await selection_coro
            if isinstance(result_obj, tuple):
                client, model = result_obj
            else:
                client = result_obj
                model = getattr(client, 'model_name', None)
                if not model:
                    raise RuntimeError(f"[DM_ERROR] LLM client returned without model_name for chat {chat_id}")

            # Провайдер: берём из активной записи selector или определяем по базовому URL
            _active = (self.selector.active or {}).get(tier_hint) or {}
            _provider = _active.get("type", "unknown")

            t_end_select = time.perf_counter()
            print(f"[DM_TRACE] Step 3.2: Selection SUCCESS in {t_end_select - t_start_select:.4f}s. Model: {model}")

            # ШАГ 4: Формирование промпта и запрос к LLM
            print(f"[DM_TRACE] Step 4: Dispatching request to LLM ({model})...")
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
            t_end_llm = time.perf_counter()
            
            content = getattr(response.choices[0].message, 'content', '{}')
            print(f"[DM_TRACE] Step 5: LLM Response received in {t_end_llm - t_start_llm:.2f}s")
            
            # ШАГ 6: Парсинг и мерж результатов
            raw_intent = safe_extract_json(content)
            prev_intent = self._intent_cache.get(str(chat_id), {"entities": {}})
            result = merge_followup(prev_intent, raw_intent)
            
            # Кэшируем результат
            self._intent_cache[str(chat_id)] = result
            print(f"[DM_TRACE] <<< EXIT: Intent analysis complete. Action: {result.get('action')}")

            # Записываем результат в рейтинг
            # correct=True: модель вернула понятный action (не упала, не UNKNOWN)
            # correct=False: JSON не распарсился, action отсутствует или мусор
            _action = result.get("action", "")
            _intent_correct = _action in ("SEARCH", "TROLL", "CHAT", "CONSULT")
            _latency_ms = (t_end_llm - t_start_llm) * 1000
            try:
                _rating = getattr(getattr(self, "selector", None), "rating", None)
                if _rating:
                    await _rating.record(
                        model_id=model,
                        provider=_provider,
                        tier_hint=tier_hint,
                        latency_ms=_latency_ms,
                        correct=_intent_correct,
                        call_type="intent",
                    )
            except Exception as _re:
                logger.debug(f"[DM] Rating record skipped: {_re}")

            return result

        except asyncio.TimeoutError:
            print(f"🚨 [DM_CRITICAL] Step 1 or 3 TIMEOUT. The Selector is unresponsive.")
            return self._get_fallback_intent(user_text)
        except Exception as e:
            # При 429 (rate limit) — заносим модель в блэклист чтобы следующий
            # запрос ушёл к другой модели, а не снова упёрся в TPM-лимит.
            _err_str = str(e)
            if "429" in _err_str or "rate_limit_exceeded" in _err_str.lower():
                _model_for_bl = locals().get("model")
                if _model_for_bl and hasattr(self, "selector"):
                    self.selector.report_failure(_model_for_bl, 120)
                    print(f"⚠️ [DM_TRACE] 429 on {_model_for_bl} → blacklisted 120s")
            print(f"💥 [DM_CRITICAL] Unexpected error in analyze_intent: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_intent(user_text)

    async def process_search_pipeline(self, chat_id: str, raw_products: List[Dict], intent: Dict, top_k: int = 5) -> List[Dict]:
        """Обработка результатов поиска: фильтрация, проверка семантики и дедупликация."""
        print(f"\n[DM_TRACE] Pipeline Start: Processing {len(raw_products)} raw items")
        
        # Берём category_map и intent_mapping из контекста
        category_map    = getattr(self.ctx, 'category_map', {})
        retrieval       = getattr(self.ctx, 'retrieval', None)
        intent_mapping  = getattr(retrieval, 'intent_mapping', {}) if retrieval else {}

        try:
            # 1. Семантический фильтр
            guarded = semantic_guard(raw_products, threshold=1.15)
            
            # 2. Жесткий фильтр по сущностям — с category_map для резолвинга UA→EN
            filtered = entity_filter(
                guarded, intent,
                intent_mapping=intent_mapping,
                category_map=category_map,
            )
            
            # 3. Удаление дублей и ограничение выдачи
            final = deduplicate_products(filtered, top_k=top_k)
            
            print(f"[DM_TRACE] Pipeline End: {len(final)} items remaining after all filters")
            return final
            
        except Exception as e:
            print(f"❌ [DM_ERROR] Search pipeline failed: {e}")
            return raw_products[:top_k]