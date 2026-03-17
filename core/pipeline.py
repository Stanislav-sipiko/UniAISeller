# -*- coding: utf-8 -*-
# /root/ukrsell_v4/core/pipeline.py v1.1.0

import os
import time
import asyncio
import json
from typing import List, Dict, Any, Optional, Union

# Импорты базовых компонентов системы core
from core.logger import logger, log_event
from core.dialog_manager import DialogManager
from core.analyzer import Analyzer

class StoreEngine:
    """
    StoreEngine v1.1.0 — Главный оркестратор системы ukrsell_v4.
    Объединяет DialogManager (анализ) и Analyzer (синтез) в единый пайплайн.
    Обеспечивает сквозное логирование session_id и Zero Omission обработку.
    Invisible Personalization: Тихая инъекция аналитики в контекст.
    """
    def __init__(self, ctx, llm_selector):
        """
        Инициализация движка магазина.
        :param ctx: Объект контекста магазина (StoreContext)
        :param llm_selector: Объект выбора моделей (LLMSelector)
        """
        self.ctx = ctx
        self.slug = getattr(ctx, 'slug', 'Store')
        
        # Инициализация Stage-2 (Голос системы)
        self.analyzer = Analyzer(ctx)
        
        # Инъекция аналайзера в контекст ДО инициализации DialogManager.
        # Это критически важно для соблюдения архитектурного контракта.
        self.ctx.analyzer = self.analyzer
        
        # Инициализация Stage-1 (Мозг системы)
        self.dm = DialogManager(ctx, llm_selector)
        
        logger.info(f"🚀 [ENGINE_INIT] StoreEngine v1.1.0 Active. System: {self.slug}")

    def _log_llm_interaction(self, stage: str, user_query: str, response: Any, session_id: str):
        """
        Вспомогательный метод для вывода диалога с ЛЛМ в лог.
        Позволяет видеть 'сырые' мысли и ответы модели.
        """
        try:
            # Превращаем объект ответа в строку для лога, если это словарь
            resp_str = json.dumps(response, ensure_ascii=False, indent=2) if isinstance(response, dict) else str(response)
            
            logger.info(
                f"\n--- [LLM_DIALOG_TRACE] [{stage}] ---\n"
                f"Session: {session_id}\n"
                f"User: {user_query}\n"
                f"AI Response: {resp_str}\n"
                f"---------------------------------------"
            )
        except Exception as e:
            logger.warning(f"[ENGINE] Failed to trace LLM dialog: {e}")

    async def handle_request(
        self, 
        user_query: str, 
        chat_id: str, 
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Единая точка входа для обработки запроса (The Single Entry Point).
        Проводит запрос через полный цикл: Intent -> Search -> Pipeline -> Synthesize.
        
        Внедрена логика Shadow Retrieval: если классификатор ошибся с OFF_TOPIC, 
        но есть товарные сущности, система все равно попробует найти товары.
        
        :param user_query: Текст запроса пользователя
        :param chat_id: Идентификатор чата (Telegram/Web)
        :param session_id: Уникальный ID сессии для трассировки (опционально)
        :return: Структурированный JSON ответ (text, products, status)
        """
        t_total_start = time.perf_counter()
        
        # Генерация session_id для сквозной трассировки, если не передан
        current_session_id = session_id or f"sid_{int(time.time())}_{chat_id}"
        
        log_event("PIPELINE_START", {
            "slug": self.slug,
            "chat_id": str(chat_id),
            "query_len": len(user_query)
        }, session_id=current_session_id)

        try:
            # 1. Первичная обработка через DialogManager (Intent + Initial Retrieval)
            response = await self.dm.process_query(
                user_query=user_query,
                chat_id=chat_id,
                session_id=current_session_id
            )
            
            # Логирование взаимодействия с ЛЛМ (Первичный Интент / Поиск)
            self._log_llm_interaction("INTENT_RETRIEVAL", user_query, response, current_session_id)
            
            # 2. Логика Shadow Retrieval (Архитектурный предохранитель)
            # Если интент OFF_TOPIC, но в контексте есть продукты или поиск вернул 0 из-за ошибки классификации
            if response.get("intent_applied") == "OFF_TOPIC" and not response.get("products"):
                # Пытаемся вытащить сущности из анализатора напрямую для перепроверки
                intent_data = getattr(self.ctx, "last_intent", {})
                entities = intent_data.get("entities", {})
                
                # Если обнаружена категория (например, 'одяг'), принудительно вызываем поиск
                if entities.get("category") or entities.get("properties"):
                    retrieval = getattr(self.ctx, "retrieval", None)
                    if retrieval:
                        logger.info(f"🔍 [SHADOW_RETRIEVAL] Forcing search for apparent product query: {user_query}")
                        search_results = await retrieval.search(user_query, entities=entities)
                        
                        if search_results.get("products"):
                            # Пересобираем ответ через синтезатор с новыми данными
                            response = await self.analyzer.synthesize_response(
                                search_results=search_results,
                                intent={"action": "SEARCH", "entities": entities},
                                user_query=user_query,
                                chat_context=response.get("chat_context", "")
                            )
                            # Логируем результат после Shadow Recovery
                            self._log_llm_interaction("SHADOW_RECOVERY", user_query, response, current_session_id)
            
            # Расчет общего времени выполнения пайплайна
            total_ms = round((time.perf_counter() - t_total_start) * 1000, 1)
            
            log_event("PIPELINE_SUCCESS", {
                "slug": self.slug,
                "total_ms": total_ms,
                "status": response.get("status", "SUCCESS"),
                "products_count": len(response.get("products", []))
            }, session_id=current_session_id)
            
            # Гарантируем наличие session_id в ответе для фронтенда
            response["session_id"] = current_session_id
            return response

        except Exception as e:
            # Глобальный перехват критических ошибок (Zero Omission Fallback)
            logger.error(f"💥 [PIPELINE_CRITICAL] Global failure for chat {chat_id}: {e}", exc_info=True)
            
            total_ms_error = round((time.perf_counter() - t_total_start) * 1000, 1)
            log_event("PIPELINE_ERROR", {
                "slug": self.slug,
                "error": str(e),
                "total_ms": total_ms_error
            }, session_id=current_session_id)
            
            # Аварийный ответ, который не "вешает" фронтенд и бота
            return {
                "text": "Вибачте, виникла технічна помилка при обробці запиту. Наші фахівці вже працюють над її усуненням.",
                "products": [],
                "status": "CRITICAL_ERROR",
                "session_id": current_session_id
            }

    async def warmup(self):
        """
        Прогрев компонентов системы (Warmup Mode).
        Используется в CI/CD и при старте контейнера для проверки готовности LLM и индексов.
        """
        logger.info(f"⏳ [WARMUP] Starting system checks for '{self.slug}'...")
        try:
            # 1. Проверка доступности селектора моделей
            if hasattr(self.dm, 'selector'):
                # Допустим, у селектора есть метод ensure_ready
                await self.dm.selector.ensure_ready()
            
            # 2. Проверка наличия индексов поиска
            retrieval = getattr(self.ctx, 'retrieval', None)
            if retrieval is None:
                logger.warning(f"⚠️ [WARMUP] Retrieval layer is missing in Context for {self.slug}")
            else:
                if hasattr(retrieval, 'ensure_ready'):
                    await retrieval.ensure_ready()
            
            await asyncio.sleep(0.1)  # Минимальная задержка для стабилизации событий
            logger.info(f"✨ [WARMUP] All systems green. Pipeline is ready for traffic.")
            return True
        except Exception as e:
            logger.error(f"❌ [WARMUP_FAILED] Pipeline not ready: {e}")
            return False

def get_version():
    return "1.1.0-ZERO-OMISSION"

# Пример использования (As-Is интеграция):
# -------------------------------------------
# from core.pipeline import StoreEngine
# from core.context import StoreContext 
# 
# engine = StoreEngine(ctx, llm_selector)
# await engine.warmup()
# result = await engine.handle_request("Шукаю іграшку для великого собаки", chat_id="user_123")