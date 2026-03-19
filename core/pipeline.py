# -*- coding: utf-8 -*-
# /root/ukrsell_v4/core/pipeline.py v1.2.1
"""
Changelog v1.2.1:
  - FIX (#1): Устранён двойной синтез. process_query теперь возвращает
    {"intent": ..., "products": [...]} без синтеза. handle_request вызывает
    analyzer.synthesize_response ОДИН РАЗ — после Shadow Recovery или напрямую.
  - FIX (#4): save_history вызывается с session_id для связи записей БД с логами LLM.
  - FIX: get_chat_context вызывается через await (async в DM v7.7.1).
  - Shadow Retrieval Condition B: action==CHAT + >= 2 слов из schema_keys → SEARCH.
    Порог >= 2 исключает случайные триггеры на одиночное слово.

Changelog v1.2.0:
  - process_query возвращает только Intent + Products.
  - session_id гарантировано в финальном ответе и логах.
  - Shadow Retrieval расширен на CHAT + schema_keywords.
"""

import os
import time
import asyncio
import json
from typing import List, Dict, Any, Optional, Union

from core.logger import logger, log_event
from core.dialog_manager import DialogManager
from core.analyzer import Analyzer


class StoreEngine:
    """
    StoreEngine v1.2.1 — Головний оркестратор системи ukrsell_v4.
    Об'єднує DialogManager (аналіз + retrieval) і Analyzer (синтез) в єдиний пайплайн.

    Архитектурный контракт v1.2.1:
      DialogManager.process_query  — Intent + Products (без синтеза).
      Analyzer.synthesize_response — единственная точка синтеза текста.
      Shadow Retrieval             — принудительный повторный поиск при ошибке классификации.
    """

    def __init__(self, ctx, llm_selector):
        self.ctx  = ctx
        self.slug = getattr(ctx, 'slug', 'Store')

        # Stage-2: Голос системы. Инъекция ДО инициализации DialogManager —
        # DialogManager.process_query достаёт analyzer из ctx.
        self.analyzer     = Analyzer(ctx)
        self.ctx.analyzer = self.analyzer

        # Stage-1: Мозг системы
        self.dm = DialogManager(ctx, llm_selector)

        logger.info(f"🚀 [ENGINE_INIT] StoreEngine v1.2.1 Active. System: {self.slug}")

    # ── Вспомогательные методы ────────────────────────────────────────────────

    def _log_llm_interaction(
        self, stage: str, user_query: str, response: Any, session_id: str
    ):
        """Трассировка диалога с LLM в лог."""
        try:
            resp_str = (
                json.dumps(response, ensure_ascii=False, indent=2)
                if isinstance(response, dict)
                else str(response)
            )
            logger.info(
                f"\n--- [LLM_DIALOG_TRACE] [{stage}] ---\n"
                f"Session: {session_id}\n"
                f"User: {user_query}\n"
                f"AI Response: {resp_str}\n"
                f"---------------------------------------"
            )
        except Exception as e:
            logger.warning(f"[ENGINE] Failed to trace LLM dialog: {e}")

    def _count_schema_keywords(self, text: str) -> int:
        """
        Считает количество слов из schema_keys в тексте запроса.
        Используется для Shadow Retrieval Condition B.
        """
        schema_keys = getattr(self.ctx, 'schema_keys', [])
        if not schema_keys:
            return 0
        text_low = text.lower()
        return sum(1 for key in schema_keys if key.lower() in text_low)

    # ── Основной обработчик ───────────────────────────────────────────────────

    async def handle_request(
        self,
        user_query: str,
        chat_id: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Единая точка входа для обработки запроса.

        FIX (#1): Синтез вызывается ОДИН РАЗ — после всех проверок включая Shadow Recovery.
          Алгоритм:
            1. DM.process_query → Intent + Products (без синтеза).
            2. Shadow Recovery если нужно → обновляем products.
            3. Analyzer.synthesize_response → финальный текст.

        FIX (#4): save_history с session_id для связи записей БД с логами LLM.
        """
        t_total_start      = time.perf_counter()
        current_session_id = session_id or f"sid_{int(time.time())}_{chat_id}"

        log_event("PIPELINE_START", {
            "slug":      self.slug,
            "chat_id":   str(chat_id),
            "query_len": len(user_query),
        }, session_id=current_session_id)

        try:
            # 1. Intent + Retrieval — DialogManager возвращает структурированные данные
            dm_result = await self.dm.process_query(
                user_query=user_query,
                chat_id=chat_id,
                session_id=current_session_id,
            )

            intent   = dm_result.get("intent", {})
            products = dm_result.get("products", [])
            action   = intent.get("action", "")
            entities = intent.get("entities", {})

            self._log_llm_interaction(
                "INTENT_RETRIEVAL",
                user_query,
                {"action": action, "entities": entities, "products_count": len(products)},
                current_session_id,
            )

            # 2. Shadow Retrieval
            # Condition A: action==OFF_TOPIC + нет продуктов + есть entities
            # Condition B: action==CHAT + >= 2 слов из schema_keys
            _shadow_a = (
                action == "OFF_TOPIC"
                and not products
                and (entities.get("category") or entities.get("properties"))
            )
            _shadow_b = (
                action == "CHAT"
                and self._count_schema_keywords(user_query) >= 2
            )

            if _shadow_a or _shadow_b:
                retrieval = getattr(self.ctx, "retrieval", None)
                if retrieval:
                    reason = "OFF_TOPIC+entities" if _shadow_a else "CHAT+schema_keywords"
                    logger.info(
                        f"🔍 [SHADOW_RETRIEVAL] Forcing search ({reason}): {user_query}"
                    )
                    shadow_results = await retrieval.search(
                        user_query, entities=entities or {}
                    )
                    shadow_products = shadow_results.get("products", [])

                    if shadow_products:
                        # Переопределяем продукты и интент для синтеза
                        products = shadow_products
                        intent   = {"action": "SEARCH", "entities": entities}
                        logger.info(
                            f"✅ [SHADOW_RECOVERY] Found {len(products)} products."
                        )
                        self._log_llm_interaction(
                            "SHADOW_RECOVERY",
                            user_query,
                            {"products_count": len(products)},
                            current_session_id,
                        )

            # 3. Синтез — ОДИН РАЗ, после всех проверок
            chat_context = await self.dm.get_chat_context(chat_id)

            response = await self.analyzer.synthesize_response(
                search_results={"products": products, "status": dm_result.get("search_status")},
                intent=intent,
                user_query=user_query,
                chat_context=chat_context,
                session_id=current_session_id,
            )

            # FIX (#4): сохраняем ответ ассистента с session_id
            asyncio.create_task(
                self.dm.save_history(
                    chat_id, "assistant", response.get("text", ""),
                    session_id=current_session_id,
                )
            )

            # 4. Финализация
            total_ms = round((time.perf_counter() - t_total_start) * 1000, 1)

            log_event("PIPELINE_SUCCESS", {
                "slug":           self.slug,
                "total_ms":       total_ms,
                "status":         response.get("status", "SUCCESS"),
                "products_count": len(products),
                "action":         action,
            }, session_id=current_session_id)

            response["session_id"] = current_session_id
            return response

        except Exception as e:
            logger.error(
                f"💥 [PIPELINE_CRITICAL] Global failure for chat {chat_id}: {e}",
                exc_info=True,
            )
            total_ms_error = round((time.perf_counter() - t_total_start) * 1000, 1)
            log_event("PIPELINE_ERROR", {
                "slug":     self.slug,
                "error":    str(e),
                "total_ms": total_ms_error,
            }, session_id=current_session_id)

            return {
                "text":       "Вибачте, виникла технічна помилка при обробці запиту. "
                              "Наші фахівці вже працюють над її усуненням.",
                "products":   [],
                "status":     "CRITICAL_ERROR",
                "session_id": current_session_id,
            }

    # ── Прогрев ───────────────────────────────────────────────────────────────

    async def warmup(self) -> bool:
        """
        Прогрев компонентов системы.
        Используется в CI/CD и при старте контейнера для проверки готовности.
        """
        logger.info(f"⏳ [WARMUP] Starting system checks for '{self.slug}'...")
        try:
            if hasattr(self.dm, 'selector'):
                await self.dm.selector.ensure_ready()

            retrieval = getattr(self.ctx, 'retrieval', None)
            if retrieval is None:
                logger.warning(
                    f"⚠️ [WARMUP] Retrieval layer is missing in Context for {self.slug}"
                )
            else:
                if hasattr(retrieval, 'ensure_ready'):
                    await retrieval.ensure_ready()

            await asyncio.sleep(0.1)
            logger.info("✨ [WARMUP] All systems green. Pipeline is ready for traffic.")
            return True
        except Exception as e:
            logger.error(f"❌ [WARMUP_FAILED] Pipeline not ready: {e}")
            return False


def get_version():
    return "1.2.1-ZERO-OMISSION"


# Пример использования:
# from core.pipeline import StoreEngine
# engine = StoreEngine(ctx, llm_selector)
# await engine.warmup()
# result = await engine.handle_request("Шукаю іграшку для великого собаки", chat_id="user_123")