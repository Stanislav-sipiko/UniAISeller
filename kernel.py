# /root/ukrsell_v4/kernel.py v7.8.6
import os
import json
import asyncio
import time
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
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
from core.cache_manager import SemanticCache
from engine.base import StoreEngine


class UkrSellKernel:
    """
    AI Kernel v7.8.6.

    Архитектурный принцип:
        Ядро универсально — не содержит логики конкретного магазина.
        Все специфические данные (category_map, profile, prompts, currency)
        берутся исключительно из StoreContext.

    Changelog:
        v7.8.5: SYNC: Updated synthesize_response calls to match Analyzer v7.8.6 signature.
        v7.8.6: SAFETY: Added robust try-except blocks around LLM synthesis with fallback 
                to template rendering to prevent pipeline crashes.
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

    async def initialize(self, target_slug: Optional[str] = None):
        """Full async initialization of the kernel."""
        start_time = time.time()
        logger.info(f"🚀 Initializing UkrSell Kernel. Target: {target_slug or 'ALL'}")
        
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"✅ Shared SentenceTransformer [{self.model_name}] loaded successfully.")
        except Exception as e:
            logger.critical(f"❌ Failed to load embedding model: {e}")
            raise

        await self._init_llm_selector()

        self.registry = StoreRegistry(stores_root="/root/ukrsell_v4/stores", kernel=self)

        def engine_factory(ctx: StoreContext):
            ctx.selector = self.selector
            ctx.kernel = self
            
            self._validate_store_assets(ctx)
            
            from core.dialog_manager import DialogManager
            ctx.dialog_manager = DialogManager(ctx, self.selector) 
            
            ctx.cache = SemanticCache(ctx, self.model)
            ctx.negative_examples = self._load_store_negative_examples(ctx.base_path)
            ctx.analyzer = Analyzer(ctx)
            ctx.retrieval = RetrievalEngine(ctx, self.model)
            
            return StoreEngine(ctx)

        try:
            await self.registry.load_all(
                engine_factory, 
                self.selector, 
                only_slug=target_slug
            )
            self.llm_ready.set()
            logger.info(f"✨ Kernel initialization completed in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"❌ Error during registry.load_all: {e}")
            raise

    def _validate_store_assets(self, ctx: StoreContext):
        base = Path(ctx.base_path)
        assets_found = True
        
        if not (base / "faiss.index").exists():
            logger.error(f"[{ctx.slug}] Missing critical asset: faiss.index")
            assets_found = False
            
        id_map_variants = ["id_map.json", "idmap.json"]
        actual_id_map = next((v for v in id_map_variants if (base / v).exists()), None)
        
        if actual_id_map:
            ctx.id_map_path = str(base / actual_id_map)
            logger.debug(f"[{ctx.slug}] Found id_map at {actual_id_map}")
        else:
            logger.error(f"[{ctx.slug}] Missing critical asset: id_map.json (checked variants: {id_map_variants})")
            assets_found = False
            
        if not assets_found:
            logger.warning(f"[{ctx.slug}] Store initialized with MISSING assets. Search might fail.")

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

    async def process_request(
        self,
        user_text: str,
        chat_id: Any,
        ctx: StoreContext,
        top_k: int = 5
    ) -> str:
        start_time = time.time()
        
        # 0. Semantic Cache Check
        if hasattr(ctx, 'cache') and ctx.cache:
            cached_answer = ctx.cache.get_answer(user_text)
            if cached_answer:
                log_pipeline_step("SEMANTIC_CACHE", time.time() - start_time, extra={"hit": True})
                return cached_answer

        # 1. Intent Analysis
        intent = await ctx.dialog_manager.analyze_intent(user_text, chat_id)
        log_pipeline_step("INTENT_ANALYSIS", time.time() - start_time, extra={"intent": intent})

        action = intent.get("action", "SEARCH")
        if action in ("CHAT", "OFF_TOPIC", "TROLL"):
            return intent.get("response_text") or "Я можу допомогти вам обрати товари з нашого каталогу."

        # 2. Retrieval
        retrieval_start = time.time()
        search_results = await ctx.retrieval.search(query=user_text, entities=intent, top_k=top_k * 4)
        raw_hits = search_results.get("products", [])
        log_pipeline_step("RETRIEVAL", time.time() - retrieval_start, extra={"hits": len(raw_hits)})

        # 3. Dialog Manager Pipeline
        pipeline_start = time.time()
        filtered_products = await ctx.dialog_manager.process_search_pipeline(
            chat_id=str(chat_id),
            raw_products=raw_hits,
            intent=intent,
            top_k=top_k
        )
        log_pipeline_step("INTEL_PIPELINE", time.time() - pipeline_start, extra={"final": len(filtered_products)})

        # 4. Confidence Evaluation
        wrapped_search_result = {
            "products": filtered_products,
            "detected_category": intent.get("category") or intent.get("entities", {}).get("category")
        }
        
        conf_data = confidence_evaluate(
            search_result=wrapped_search_result,
            intent=intent,
            user_query=user_text,
            profile=ctx.profile,
            category_map=ctx.category_map
        )
        mode = conf_data["mode"]

        # 5. Response Generation
        return await self._decide_response_mode(
            ctx=ctx,
            mode=mode,
            products=filtered_products,
            intent=intent,
            query=user_text,
            user_id=chat_id,
            top_categories=conf_data.get("top_categories", [])
        )

    async def _decide_response_mode(
        self,
        ctx: StoreContext,
        mode: str,
        products: list,
        intent: Dict,
        query: str,
        user_id: Any,
        top_categories: Optional[list] = None,
    ) -> str:
        """
        Unified response router with Safety Net fallback.
        """
        history_context = ctx.dialog_manager.get_chat_context(user_id, minutes=30)

        # ── NO_RESULTS ──
        if mode == "NO_RESULTS":
            try:
                synthesis = await ctx.analyzer.synthesize_response(
                    intent=intent,
                    products=[],
                    mode="NO_RESULTS",
                    query=query,
                    history=history_context
                )
                if synthesis and isinstance(synthesis, dict):
                    return synthesis.get("text", ctx.prompts.get("not_found", "Нічого не знайдено."))
                return synthesis or ctx.prompts.get("not_found", "Нічого не знайдено.")
            except Exception as e:
                logger.error(f"[{ctx.slug}] Synthesis fallback (NO_RESULTS): {e}")
                return ctx.prompts.get("not_found", "Нічого не знайдено.")

        # ── ASK_CLARIFICATION ──
        if mode == "ASK_CLARIFICATION":
            try:
                synthesis = await ctx.analyzer.synthesize_response(
                    intent=intent,
                    products=products,
                    mode="ASK_CLARIFICATION",
                    query=query,
                    history=history_context,
                    top_categories=top_categories
                )
                if synthesis and isinstance(synthesis, dict):
                    return synthesis.get("text", "Уточніть, будь ласка, ваш запит.")
                return synthesis or "Уточніть, будь ласка, ваш запит."
            except Exception as e:
                logger.error(f"[{ctx.slug}] Synthesis fallback (CLARIFY): {e}")
                return "Уточніть, будь ласка, ваш запит."

        # ── SHOW_PRODUCTS ──
        if mode == "SHOW_PRODUCTS":
            if intent.get("action") in ("CONSULT", "INFO", "SEARCH"):
                try:
                    synthesis_result = await ctx.analyzer.synthesize_response(
                        intent=intent,
                        products=products,
                        mode="SHOW_PRODUCTS",
                        query=query,
                        history=history_context
                    )
                    if synthesis_result:
                        if isinstance(synthesis_result, dict):
                            return synthesis_result.get("text")
                        return synthesis_result
                except Exception as e:
                    logger.warning(f"[{ctx.slug}] LLM Synthesis failed, falling back to Template: {e}")

        # Default Fallback: Template rendering
        return await self._build_template_response(ctx, products, query)

    async def _build_template_response(self, ctx: StoreContext, products: list, query: str) -> str:
        """Универсальный рендерер карточек товаров."""
        if not products:
            return ctx.prompts.get("not_found", "Нічого не знайдено.")

        intro = await ctx.get_human_intro(query, len(products))
        view_label  = ctx.prompts.get("view_button", "Переглянути")
        price_label = ctx.prompts.get("price_label", "Ціна")
        parts = [intro, ""]

        for i, p in enumerate(products[:5], 1):
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

    async def close(self):
        logger.info("⏳ Closing Kernel resources...")
        if self.selector and hasattr(self.selector, 'close'):
            if asyncio.iscoroutinefunction(self.selector.close):
                await self.selector.close()
            else:
                self.selector.close()
        logger.info("✅ Kernel resources released.")

    def get_all_active_slugs(self) -> List[str]:
        return self.registry.get_all_slugs() if self.registry else []

    def __repr__(self):
        stores_count = len(self.registry.get_all_slugs()) if self.registry else 0
        return f"<UkrSellKernel v7.8.6 stores_active={stores_count} model={self.model_name}>"