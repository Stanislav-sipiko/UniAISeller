# /root/ukrsell_v4/kernel.py v7.8.0
import os
import json
import asyncio
import time
import hashlib
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

# Project structure imports
from core.logger import logger, log_event, log_pipeline_step
from core.config import HF_TOKEN
from core.llm_selector import LLMSelector
from core.analyzer import Analyzer
from core.registry import StoreRegistry
from core.retrieval import RetrievalEngine
from core.store_context import StoreContext
from core.translator import TextTranslator
from core.confidence import evaluate as confidence_evaluate
from engine.base import StoreEngine


class UkrSellKernel:
    """
    AI Kernel v7.8.0.

    Архитектурный принцип:
        Ядро универсально — не содержит логики конкретного магазина.
        Все специфические данные (category_map, profile, prompts, currency)
        берутся исключительно из StoreContext.

    Pipeline: запрос → intent → retrieval → confidence → _decide_response_mode
    Маршрутизация: единая точка _decide_response_mode.
    Рендеринг карточек: _build_template_response (все режимы с товарами).

    Changelog:
        v7.7.1  category_map передаётся в confidence_evaluate (UA/RU→EN резолвинг)
        v7.7.2  product_url приоритет над url в карточке товара
        v7.7.3  ASK_CLARIFICATION передаёт all_products в synthesize_response
        v7.8.0  Рефакторинг: единая маршрутизация _decide_response_mode,
                убраны дубли веток, log_pipeline_step сохранён для observability,
                ядро не содержит магазино-специфичной логики
    """

    def __init__(self, model_name: str = 'intfloat/multilingual-e5-small'):
        if HF_TOKEN:
            os.environ["HF_TOKEN"] = HF_TOKEN
            logger.info("HF_TOKEN set in environment.")

        self.model_name = model_name
        self._request_cache: Dict[str, float] = {}
        self._cache_ttl = 5

        self.model = None
        self.translator = TextTranslator()
        self.selector = LLMSelector()
        self.registry = None
        self.llm_ready = asyncio.Event()

    ##############################
    # Heavy initialization
    ##############################
    async def initialize(self):
        """Full async initialization of the kernel."""
        start_time = time.time()
        logger.info(f"🚀 Initializing UkrSell Kernel v7.8.0. Model: {self.model_name}")

        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info("✅ SentenceTransformer loaded successfully.")
        except Exception as e:
            logger.critical(f"❌ Failed to load embedding model: {e}")
            raise

        await self._init_llm_selector()

        self.registry = StoreRegistry(stores_root="/root/ukrsell_v4/stores", kernel=self)

        def engine_factory(ctx: StoreContext):
            ctx.selector = self.selector
            ctx.kernel = self
            ctx.negative_examples = self._load_store_negative_examples(ctx.base_path)
            ctx.analyzer = Analyzer(ctx)
            # ИСПРАВЛЕНО: убрали self.translator
            ctx.retrieval = RetrievalEngine(ctx, self.model) 
            return StoreEngine(ctx)

        try:
            await self.registry.load_all(engine_factory, self.selector)
        except Exception as e:
            logger.error(f"❌ Error during registry.load_all: {e}")
            raise

        duration = round(time.time() - start_time, 2)
        active_slugs = self.registry.get_all_slugs()
        log_event("KERNEL_READY", {
            "stores_loaded": len(active_slugs),
            "slugs": active_slugs,
            "llm_status": self.selector.get_status(),
            "init_time_sec": duration,
        })
        logger.info(f"UkrSell Platform ready. Stores: {len(active_slugs)} in {duration}s")
        self.llm_ready.set()

    ##############################
    # LLMSelector init with retries
    ##############################
    async def _init_llm_selector(self):
        max_attempts = 3
        logger.info("📡 Warming up LLM Selector...")
        for attempt in range(1, max_attempts + 1):
            try:
                await self.selector.refresh(force=True)
                status = self.selector.get_status()
                if any(v != "OFFLINE" for v in status.values()):
                    logger.info(f"✅ LLM Stacks ONLINE (attempt {attempt}). Status: {status}")
                    return
                logger.warning(f"⏳ LLM Stacks OFFLINE. Retry {attempt}/{max_attempts} in 2s...")
            except Exception as e:
                logger.error(f"⚠️ LLM Warmup attempt {attempt} failed: {e}")
            await asyncio.sleep(2.0)
        logger.critical("🚨 KERNEL STARTUP CRITICAL: All LLM stacks remain OFFLINE after retries.")

    ##############################
    # Store data readiness
    ##############################
    async def wait_for_store(self, slug: str, timeout: float = 30.0) -> bool:
        ctx = self.registry.get_context(slug)
        if not ctx:
            return False
        if not hasattr(ctx, 'data_ready'):
            return True
        try:
            if not ctx.data_ready.is_set():
                logger.info(f"⏳ Waiting for store assets [{slug}]...")
                await asyncio.wait_for(ctx.data_ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.error(f"❌ Timeout waiting for store {slug} assets.")
            return False

    ##############################
    # Negative examples loader
    ##############################
    def _load_store_negative_examples(self, base_path: str) -> list:
        patch_path = os.path.join(base_path, "fsm_soft_patch.json")
        if os.path.exists(patch_path):
            try:
                with open(patch_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    return data.get("troll_patterns", []) or data.get("fsm_errors", [])
            except Exception as e:
                logger.error(f"Error loading negative examples: {e}")
        return []

    ##############################
    # Duplicate request protection
    ##############################
    def _is_duplicate(self, user_id: Any, text: str) -> bool:
        if not text:
            return False
        request_hash = hashlib.md5(f"{user_id}:{text}".encode()).hexdigest()
        now = time.time()
        self._request_cache = {k: v for k, v in self._request_cache.items() if now - v < self._cache_ttl}
        if request_hash in self._request_cache:
            return True
        self._request_cache[request_hash] = now
        return False

    ##############################
    # Language normalization (RU→UA)
    ##############################
    async def _normalize_update_language(self, engine: StoreEngine, update: Dict[str, Any]):
        message = update.get("message", {})
        text = message.get("text")
        if not text or len(text.strip()) < 2:
            return
        store_lang = getattr(engine.ctx, "language", "Ukrainian")
        if store_lang == "Ukrainian":
            try:
                if await self.translator.detect_language(text) == "ru":
                    translated = await self.translator.translate(text, target_lang="uk")
                    if translated and translated.lower() != text.lower():
                        logger.info(f"[{engine.ctx.slug}] RU→UA: '{text}' → '{translated}'")
                        update["message"]["original_text"] = text
                        update["message"]["text"] = translated
            except Exception as e:
                logger.error(f"Kernel language normalization error: {e}")

    ##############################
    # Webhook entry
    ##############################
    async def handle_webhook(self, slug: str, update: dict) -> bool:
        if not self.llm_ready.is_set():
            try:
                await asyncio.wait_for(self.llm_ready.wait(), timeout=20.0)
            except asyncio.TimeoutError:
                logger.error(f"🚨 LLM readiness timeout for {slug}. Aborting.")
                return False

        await self.wait_for_store(slug)
        message = update.get("message", {})
        user_id = message.get("from", {}).get("id")
        text = message.get("text", "")

        if self._is_duplicate(user_id, text):
            return True

        engine = self.registry.get_engine(slug)
        if not engine:
            return False

        log_event("WEBHOOK_IN", {"slug": slug, "user_id": user_id})
        try:
            await self._normalize_update_language(engine, update)
            await engine.handle_update(update)
            return True
        except Exception as e:
            logger.error(f"Critical webhook error for {slug}: {e}", exc_info=True)
            return False

    ##############################
    # Core pipeline
    ##############################
    async def process_request(self, user_text: str, chat_id: str, ctx: Any, limit: int = 5) -> str:
        if not self.llm_ready.is_set(): await asyncio.wait_for(self.llm_ready.wait(), timeout=10.0)
        try: u_id = int(chat_id) if str(chat_id).isdigit() else None
        except: u_id = None
        return await self.get_recommendations(ctx=ctx, query=user_text, top_k=limit, user_id=u_id)

    async def get_recommendations(
        self,
        ctx: StoreContext,
        query: str,
        filters: Dict = None,
        top_k: int = 5,
        user_id: int = None,
    ) -> str:
        if not self.llm_ready.is_set():
            await asyncio.wait_for(self.llm_ready.wait(), timeout=10.0)

        start_time = time.time()

        # ── Intent ──────────────────────────────────────────────────
        intent = await ctx.dialog_manager.analyze_intent(query, user_id)
        if not intent.get("entities"):
            intent["entities"] = {}
        if filters:
            intent["entities"].update(filters)
        log_pipeline_step("INTENT_ANALYSIS", time.time() - start_time, extra={"intent": intent})

        # ── Action shortcuts (no search needed) ─────────────────────
        action = intent.get("action", "SEARCH")
        if action in ("CHAT", "OFF_TOPIC"):
            logger.debug(f"[{ctx.slug}] Action={action} → short-circuit")
            return intent.get("response_text") or "Я тут, щоб допомогти з вибором товарів! 🐾"
        if action == "TROLL":
            logger.debug(f"[{ctx.slug}] Action=TROLL → short-circuit")
            return intent.get("response_text") or "🧐"
        if action not in ("SEARCH", "CONSULT", "INFO", "EMOTION"):
            logger.debug(f"[{ctx.slug}] Unknown action '{action}' → remapped to SEARCH")
            intent["action"] = "SEARCH"

        # ── Retrieval ────────────────────────────────────────────────
        retrieval_start = time.time()
        search_results = await ctx.retrieval.search(query=query, entities=intent, top_k=top_k * 5)
        raw_hits = search_results.get("products", [])
        log_pipeline_step("RETRIEVAL", time.time() - retrieval_start, extra={"hits": len(raw_hits)})

        # ── Intelligence pipeline ────────────────────────────────────
        pipeline_start = time.time()
        final_products = await ctx.dialog_manager.process_search_pipeline(
            chat_id=str(user_id),
            raw_products=raw_hits,
            intent=intent,
            top_k=top_k,
        )
        log_pipeline_step("INTEL_PIPELINE", time.time() - pipeline_start, extra={"final": len(final_products)})

        # ── Confidence Engine ────────────────────────────────────────
        confidence_result = confidence_evaluate(
            search_result={**search_results, "products": final_products},
            intent=intent,
            user_query=query,
            profile=getattr(ctx, "profile", None),
            category_map=getattr(ctx, "category_map", None),
        )
        mode = confidence_result["mode"]
        logger.info(
            f"[{ctx.slug}] mode={mode} score={confidence_result['confidence']} "
            f"products={len(final_products)}"
        )

        # ── Unified response routing ─────────────────────────────────
        return await self._decide_response_mode(
            ctx=ctx,
            mode=mode,
            products=final_products,
            intent=intent,
            query=query,
            user_id=user_id,
            search_status=search_results.get("status"),
            top_categories=confidence_result.get("top_categories", []),
        )

    ##############################
    # Unified response router
    ##############################
    async def _decide_response_mode(
        self,
        ctx: StoreContext,
        mode: str,
        products: list,
        intent: Dict,
        query: str,
        user_id: Any,
        search_status: Optional[str] = None,
        top_categories: Optional[list] = None,
    ) -> str:
        history_context = ctx.dialog_manager.get_chat_context(user_id, minutes=5)

        # ── NO_RESULTS — только при реально пустой выдаче ───────────
        if mode == "NO_RESULTS":
            logger.warning(f"[{ctx.slug}] NO_RESULTS. Intent: {intent}")
            synthesis = await ctx.analyzer.synthesize_response(
                search_results={"products": [], "status": "NO_RESULTS"},
                entities=intent,
                user_query=query,
                chat_context=history_context,
                mode="NO_RESULTS",
            )
            if synthesis and synthesis.get("text"):
                return synthesis["text"]
            return ctx.prompts.get("not_found", "На жаль, нічого не знайдено.")

        # ── ASK_CLARIFICATION — LLM получает товары для контекста ───
        if mode == "ASK_CLARIFICATION":
            logger.info(f"[{ctx.slug}] ASK_CLARIFICATION. products={len(products)}")
            synthesis = await ctx.analyzer.synthesize_response(
                search_results={"products": products, "status": "ASK_CLARIFICATION"},
                entities=intent,
                user_query=query,
                chat_context=history_context,
                mode="ASK_CLARIFICATION",
                top_categories=top_categories or [],
            )
            if synthesis and synthesis.get("text"):
                return synthesis["text"]
            # Fallback: показать товары шаблоном если LLM не вернул текст
            if products:
                return await self._build_template_response(ctx, products, query)
            return "Уточніть, будь ласка, що саме вас цікавить?"

        # ── SHOW_PRODUCTS ────────────────────────────────────────────
        # GRAY_ZONE / SEMANTIC_REJECT → LLM synthesis
        if search_status in ("GRAY_ZONE_SUCCESS", "SEMANTIC_REJECT"):
            logger.debug(f"[{ctx.slug}] LLM synthesis ({search_status})")
            synth_start = time.time()
            synthesis = await ctx.analyzer.synthesize_response(
                search_results={"products": products, "status": search_status},
                entities=intent,
                user_query=query,
                chat_context=history_context,
                mode="SHOW_PRODUCTS",
            )
            log_pipeline_step("SYNTHESIS_LLM", time.time() - synth_start)
            if synthesis and synthesis.get("text"):
                return synthesis["text"]
            return ctx.prompts.get("error_msg", "Помилка синтезу.")

        # Fast path — template
        template_start = time.time()
        response = await self._build_template_response(ctx, products, query)
        log_pipeline_step("SYNTHESIS_TEMPLATE", time.time() - template_start)
        return response

    ##############################
    # Template card renderer
    ##############################
    async def _build_template_response(self, ctx: StoreContext, products: list, query: str) -> str:
        """
        Универсальный рендерер карточек товаров.
        Все тексты, валюта и лейблы берутся из ctx (StoreContext).
        Используется в SHOW_PRODUCTS и как fallback в ASK_CLARIFICATION.

        URL: product_url → url → "#"  (fix v7.7.2)
        """
        intro = await ctx.get_human_intro(query, len(products))
        view_label  = ctx.prompts.get("view_button", "Переглянути")
        price_label = ctx.prompts.get("price_label", "Ціна")
        parts = [intro, ""]

        for i, res in enumerate(products, 1):
            p = res.get("data") or res.get("product") or res
            if not p:
                continue
            name  = p.get("name") or p.get("title", "Товар")
            price = p.get("price", "---")
            url   = p.get("product_url") or p.get("url") or "#"
            parts.append(
                f"{i}. 🛍 **{name}**\n"
                f"    {price_label}: {price} {ctx.currency}\n"
                f"    [{view_label}]({url})"
            )

        footer = (
            "Підсказати щось конкретне щодо цих варіантів?"
            if ctx.language == "Ukrainian"
            else "Подсказать что-то конкретное по этим вариантам?"
        )
        parts.append(f"\n{footer}")
        return "\n".join(parts)

    ##############################
    # Clean shutdown
    ##############################
    async def close(self):
        logger.info("⏳ Closing Kernel resources...")
        if self.selector and hasattr(self.selector, 'close'):
            if asyncio.iscoroutinefunction(self.selector.close):
                await self.selector.close()
            else:
                self.selector.close()
        logger.info("✅ Kernel resources released.")

    ##############################
    # Utility
    ##############################
    def get_all_active_slugs(self) -> List[str]:
        return self.registry.get_all_slugs() if self.registry else []

    def __repr__(self):
        stores_count = len(self.registry.get_all_slugs()) if self.registry else 0
        return f"<UkrSellKernel v7.8.0 stores_active={stores_count}>"