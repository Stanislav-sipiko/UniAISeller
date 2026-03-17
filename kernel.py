import os
import json
import asyncio
import time
import hashlib
import traceback
from typing import List, Dict, Any, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer

from core.logger import logger, log_event, log_pipeline_step
from core.config import HF_TOKEN
from core.llm_selector import LLMSelector
from core.analyzer import Analyzer
from core.registry import StoreRegistry
from core.retrieval import RetrievalEngine
from core.store_context import StoreContext
from core.translator import TextTranslator
from core.confidence import evaluate as confidence_evaluate
from core.cache_manager import SemanticCache
from core.dialog_manager import DialogManager
from engine.base import StoreEngine

class UkrSellKernel:
    """
    AI Kernel v8.4.1 Industrial Hybrid.

    Архитектурный принцип:
        Ядро универсально — не содержит логики конкретного магазина.
        Все специфические данные (category_map, profile, prompts, currency)
        бертся исключительно из StoreContext.

    Changelog:
        v8.4.1: CRITICAL FIX: Added missing llm_selector to DialogManager init.
        v8.4.0: FIX: Explicit DialogManager initialization in factory.
                FIX: Corrected intent/entities passing to RetrievalEngine v9.3.0.
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

    async def initialize(self):
        """Full async initialization of the kernel."""
        start_time = time.time()
        logger.info(f"🚀 Initializing UkrSell Kernel v8.4.1. Model: {self.model_name}")

        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info("✅ SentenceTransformer loaded successfully.")
        except Exception as e:
            logger.critical(f"❌ Failed to load embedding model: {e}")
            raise

        await self._init_llm_selector()

        self.registry = StoreRegistry(stores_root="/root/ukrsell_v4/stores", kernel=self)

        def engine_factory(ctx: StoreContext):
            """
            Dependency Injection Factory. 
            Fixed: Added self.selector as second argument for DialogManager.
            """
            ctx.selector = self.selector
            ctx.kernel = self
            ctx.negative_examples = self._load_store_negative_examples(ctx.base_path)
            
            ctx.analyzer = Analyzer(ctx)
            ctx.retrieval = RetrievalEngine(ctx, self.model)
            ctx.semantic_cache = SemanticCache(ctx, self.model)
            # FIX: Передаем селектор моделей
            ctx.dialog_manager = DialogManager(ctx, self.selector)
            
            if not hasattr(ctx, 'data_ready'):
                ctx.data_ready = asyncio.Event()
            
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

    async def _init_llm_selector(self):
        """Warms up LLM connections with retry logic."""
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
        logger.critical("🚨 KERNEL STARTUP CRITICAL: All LLM stacks remain OFFLINE.")

    async def handle_message(self, text: str, chat_id: str, slug: Optional[str] = None) -> Dict[str, Any]:
        """Industrial Entry Point for test_chat.py and CLI."""
        if not self.llm_ready.is_set():
            await asyncio.wait_for(self.llm_ready.wait(), timeout=10.0)

        slugs = self.registry.get_all_slugs()
        if not slugs:
            return {"text": "Ошибка: Магазины не загружены", "status": "ERROR"}
            
        target_slug = slug or slugs[0]
        ctx = self.registry.get_context(target_slug)
        
        if not ctx:
            return {"text": f"Ошибка: Магазин {target_slug} не найден", "status": "ERROR"}

        await self.wait_for_store(target_slug)
        start_time_perf = time.perf_counter()

        try:
            history_context = ""
            if hasattr(ctx, "dialog_manager"):
                history_context = ctx.dialog_manager.get_chat_context(chat_id, minutes=10)

            intent = await ctx.analyzer.extract_intent(text, chat_context=history_context)
            
            search_results = {"products": [], "status": "EMPTY"}
            if intent.get("action") not in ("OFF_TOPIC", "TROLL"):
                search_results = await ctx.retrieval.search(
                    query=text, 
                    entities=intent.get("entities", {}), 
                    top_k=5
                )

            response = await ctx.analyzer.synthesize_response(
                search_results=search_results,
                intent=intent,
                user_query=text,
                chat_context=history_context
            )

            llm_status = self.selector.get_status()
            active_model = "N/A"
            for k, v in llm_status.items():
                if v != "OFFLINE":
                    active_model = k
                    break

            response.update({
                "intent_applied": intent.get("action", "UNKNOWN"),
                "status": "SUCCESS",
                "model": active_model,
                "ms": int((time.perf_counter() - start_time_perf) * 1000),
                "trace": {
                    "entities": intent.get("entities", {}),
                    "count": len(search_results.get("products", [])),
                    "chat_id": chat_id,
                    "slug": target_slug
                }
            })
            
            return response

        except Exception as e:
            logger.error(f"Kernel handle_message error: {e}")
            traceback.print_exc()
            return {"text": "Виникла технічна помилка. Спробуйте пізніше.", "status": "ERROR"}

    async def handle_webhook(self, slug: str, update: dict) -> bool:
        """Standard webhook handler for production integration."""
        if not self.llm_ready.is_set():
            await asyncio.wait_for(self.llm_ready.wait(), timeout=20.0)

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

    async def process_request(self, user_text: str, chat_id: str, ctx: Any, limit: int = 5) -> str:
        """Legacy compatibility layer for internal calls."""
        try: 
            u_id = int(chat_id) if str(chat_id).isdigit() else 999
        except: 
            u_id = 999
        return await self.get_recommendations(ctx=ctx, query=user_text, top_k=limit, user_id=u_id)

    async def get_recommendations(
        self,
        ctx: StoreContext,
        query: str,
        filters: Dict = None,
        top_k: int = 5,
        user_id: int = None,
    ) -> str:
        """Core recommendation pipeline with semantic caching."""
        start_time = time.time()

        if hasattr(ctx, 'semantic_cache'):
            cached_answer = ctx.semantic_cache.get_answer(query)
            if cached_answer and not isinstance(cached_answer, (dict, list)):
                log_pipeline_step("SEMANTIC_CACHE_HIT", time.time() - start_time)
                return str(cached_answer)

        intent = await ctx.analyzer.extract_intent(
            query, 
            chat_context=ctx.dialog_manager.get_chat_context(user_id, minutes=5)
        )
        
        if filters:
            if not intent.get("entities"): intent["entities"] = {}
            intent["entities"].update(filters)
        
        log_pipeline_step("INTENT_ANALYSIS", time.time() - start_time, extra={"intent": intent})

        action = intent.get("action", "SEARCH")
        if action in ("CHAT", "OFF_TOPIC", "TROLL"):
            return str(intent.get("response_text") or "Я тут, щоб допомогти!")

        search_results = await ctx.retrieval.search(
            query=query, 
            entities=intent.get("entities", {}), 
            top_k=top_k * 3
        )
        
        final_products = await ctx.dialog_manager.process_search_pipeline(
            chat_id=str(user_id),
            search_response=search_results,
            intent=intent,
            top_k=top_k,
        )

        confidence_result = confidence_evaluate(
            search_result={**search_results, "products": final_products},
            intent=intent,
            user_query=query,
            profile=getattr(ctx, "profile", None),
            category_map=getattr(ctx, "category_map", None),
        )

        history = ctx.dialog_manager.get_chat_context(user_id, minutes=5)
        
        try:
            synthesis = await ctx.analyzer.synthesize_response(
                search_results={"products": final_products, "status": search_results.get("status")},
                intent=intent,
                user_query=query,
                chat_context=history,
                mode=confidence_result["mode"],
                top_categories=confidence_result.get("top_categories", [])
            )
            response_text = str(synthesis.get("text")) if synthesis.get("text") else ""
        except Exception as e:
            logger.error(f"Synthesis error in recommendations: {e}")
            response_text = ""

        if not response_text:
            response_text = await self._build_template_response(ctx, final_products, query)

        if hasattr(ctx, 'semantic_cache') and len(final_products) > 0 and len(response_text) > 20:
            ctx.semantic_cache.add(query, response_text)

        return response_text

    async def _build_template_response(self, ctx: StoreContext, products: list, query: str) -> str:
        """Constructs a markdown response based on store templates."""
        intro = await ctx.get_human_intro(query, len(products))
        view_label = ctx.prompts.get("view_button", "Переглянути")
        price_label = ctx.prompts.get("price_label", "Ціна")
        
        parts = [str(intro), ""]
        for i, res in enumerate(products[:5], 1):
            p = res.get("data") or res.get("product") or res
            name = p.get("name") or p.get("title", "Товар")
            price = p.get("price", "---")
            url = p.get("product_url") or p.get("url") or "#"
            parts.append(f"{i}. 🛍 **{name}**\n   {price_label}: {price} {ctx.currency}\n   [{view_label}]({url})")
        
        footer = "Підсказати щось конкретне щодо этих моделей?" if ctx.language == "Ukrainian" else "Подсказать детали по этим моделям?"
        parts.append(f"\n{footer}")
        return "\n".join(parts)

    def _load_store_negative_examples(self, base_path: str) -> list:
        """Loads FSM soft patches for specific store behavior."""
        patch_path = os.path.join(base_path, "fsm_soft_patch.json")
        if os.path.exists(patch_path):
            try:
                with open(patch_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else data.get("troll_patterns", [])
            except Exception: pass
        return []

    def _is_duplicate(self, user_id: Any, text: str) -> bool:
        """Prevents processing of rapid duplicate requests."""
        if not text: return False
        request_hash = hashlib.md5(f"{user_id}:{text}".encode()).hexdigest()
        now = time.time()
        self._request_cache = {k: v for k, v in self._request_cache.items() if now - v < self._cache_ttl}
        if request_hash in self._request_cache: return True
        self._request_cache[request_hash] = now
        return False

    async def _normalize_update_language(self, engine: StoreEngine, update: Dict[str, Any]):
        """Translates incoming text to Ukrainian if the store requires it."""
        message = update.get("message", {})
        text = message.get("text")
        if not text or len(text.strip()) < 2: return
        store_lang = getattr(engine.ctx, "language", "Ukrainian")
        if store_lang == "Ukrainian":
            try:
                if await self.translator.detect_language(text) == "ru":
                    translated = await self.translator.translate(text, target_lang="uk")
                    if translated:
                        update["message"]["original_text"] = text
                        update["message"]["text"] = translated
            except Exception: pass

    async def wait_for_store(self, slug: str, timeout: float = 30.0) -> bool:
        """Blocks until the specified store is fully initialized."""
        ctx = self.registry.get_context(slug)
        if not ctx: return False
        if hasattr(ctx, 'data_ready'):
            try:
                await asyncio.wait_for(ctx.data_ready.wait(), timeout=timeout)
                return True
            except asyncio.TimeoutError:
                logger.error(f"❌ Timeout waiting for store {slug}")
        return True

    async def close(self):
        """Cleanup kernel resources."""
        logger.info("⏳ Closing Kernel resources...")
        if self.selector:
            try:
                if asyncio.iscoroutinefunction(self.selector.close):
                    await self.selector.close()
                else:
                    self.selector.close()
            except Exception: pass
        logger.info("✅ Kernel resources released.")

    def get_all_active_slugs(self) -> List[str]:
        """Returns list of all registered store slugs."""
        return self.registry.get_all_slugs() if self.registry else []

    def __repr__(self):
        return f"<UkrSellKernel v8.4.1 stores={len(self.get_all_active_slugs())}>"