# /root/ukrsell_v4/kernel.py v8.6.0

import os
import json
import re
import asyncio
import time
import hashlib
import traceback
from typing import List, Dict, Any, Optional, Tuple
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
from core.kernel_config import (
    BASE_MAX_TOKENS, TOKENS_PER_PRODUCT, MAX_TOKENS_CAP,
    MAX_PRODUCTS_IN_PROMPT, MAX_ATTRS_PER_PRODUCT,
    COMPLEX_ENTITIES_THRESHOLD, COMPLEX_QUERY_LENGTH,
    LLM_MIN_RESPONSE_LEN, LANG_RU_RATIO_THRESHOLD,
    MIN_VALID_PRODUCTS_FOR_LLM, ATTRS_BLACKLIST,
)
from engine.base import StoreEngine

_DEFAULT_PROMPTS_PATH = os.path.join(
    os.path.dirname(__file__), "core", "kernel_prompts.json"
)


class UkrSellKernel:
    """AI Kernel v8.6.0."""

    def __init__(self, model_name: str = 'intfloat/multilingual-e5-small'):
        if HF_TOKEN:
            os.environ["HF_TOKEN"] = HF_TOKEN
            logger.info("HF_TOKEN set in environment.")

        self.model_name    = model_name
        self._request_cache: Dict[str, float] = {}
        self._cache_ttl    = 5
        self._default_prompts: Dict[str, Any] = self._load_default_prompts()

        self.model      = None
        self.translator = TextTranslator()
        self.selector   = LLMSelector()
        self.registry   = None
        self.llm_ready  = asyncio.Event()

    # ── Промпты ───────────────────────────────────────────────────────────────

    def _load_default_prompts(self) -> Dict[str, Any]:
        try:
            with open(_DEFAULT_PROMPTS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[kernel] Failed to load kernel_prompts.json: {e}")
            return {}

    def _load_prompts(self, base_path: str) -> Dict[str, Any]:
        store_path = os.path.join(base_path, "prompts.json")
        store_prompts: Dict[str, Any] = {}
        if os.path.exists(store_path):
            try:
                with open(store_path, "r", encoding="utf-8") as f:
                    store_prompts = json.load(f)
                logger.info(f"[kernel] Store prompts loaded from {store_path}")
            except Exception as e:
                logger.warning(f"[kernel] Store prompts load error: {e}. Using defaults.")
        merged = {**self._default_prompts, **store_prompts}
        return merged

    def _p(self, prompts: Dict, key: str, lang: str = "uk", **kwargs) -> str:
        block = prompts.get(key, self._default_prompts.get(key, {}))
        lang_key = "uk" if lang == "Ukrainian" else "ru"
        text = block.get(lang_key, block.get("uk", "")) if isinstance(block, dict) else str(block)
        if kwargs:
            try:
                text = text.format(**kwargs)
            except (KeyError, ValueError):
                pass
        return text

    # ── Инициализация ─────────────────────────────────────────────────────────

    async def initialize(self):
        start_time = time.time()
        logger.info(f"🚀 Initializing UkrSell Kernel v8.6.0. Model: {self.model_name}")

        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info("✅ SentenceTransformer loaded successfully.")
        except Exception as e:
            logger.critical(f"❌ Failed to load embedding model: {e}")
            raise

        await self._init_llm_selector()

        self.registry = StoreRegistry(stores_root="/root/ukrsell_v4/stores", kernel=self)

        def engine_factory(ctx: StoreContext):
            ctx.selector          = self.selector
            ctx.kernel            = self
            ctx.negative_examples = self._load_store_negative_examples(ctx.base_path)
            ctx.kernel_prompts    = self._load_prompts(ctx.base_path)

            ctx.analyzer       = Analyzer(ctx)
            ctx.retrieval      = RetrievalEngine(ctx, self.model)
            ctx.semantic_cache = SemanticCache(ctx, self.model)
            ctx.dialog_manager = DialogManager(ctx, self.selector)

            if not hasattr(ctx, 'data_ready'):
                ctx.data_ready = asyncio.Event()

            return StoreEngine(ctx)

        try:
            await self.registry.load_all(engine_factory, self.selector)
        except Exception as e:
            logger.error(f"❌ Error during registry.load_all: {e}")
            raise

        duration     = round(time.time() - start_time, 2)
        active_slugs = self.registry.get_all_slugs()

        log_event("KERNEL_READY", {
            "stores_loaded": len(active_slugs),
            "slugs":         active_slugs,
            "llm_status":    self.selector.get_status(),
            "init_time_sec": duration,
        })

        logger.info(f"UkrSell Platform ready. Stores: {len(active_slugs)} in {duration}s")
        self.llm_ready.set()

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
                logger.warning(
                    f"⏳ LLM Stacks OFFLINE. Retry {attempt}/{max_attempts} in 2s..."
                )
            except Exception as e:
                logger.error(f"⚠️ LLM Warmup attempt {attempt} failed: {e}")
            await asyncio.sleep(2.0)
        logger.critical("🚨 KERNEL STARTUP CRITICAL: All LLM stacks remain OFFLINE.")

    # ── Утилиты нормализации ──────────────────────────────────────────────────

    @staticmethod
    def _parse_price_limit(raw: Any) -> Optional[float]:
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            return float(raw)
        s = str(raw)
        s = s.replace("\xa0", "").replace("₴", "").replace("грн", "").strip()
        s = re.sub(r"\s+", "", s)
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        elif "," in s:
            s = s.replace(",", ".")
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_variants(raw_variants: Any) -> List[Dict]:
        if isinstance(raw_variants, list):
            return raw_variants
        if isinstance(raw_variants, str) and raw_variants.strip():
            try:
                parsed = json.loads(raw_variants)
                return parsed if isinstance(parsed, list) else []
            except (json.JSONDecodeError, ValueError):
                pass
        return []

    @staticmethod
    def _parse_attributes(raw_attrs: Any) -> Dict[str, Any]:
        if isinstance(raw_attrs, dict):
            return raw_attrs
        if isinstance(raw_attrs, str) and raw_attrs.strip():
            try:
                parsed = json.loads(raw_attrs)
                return parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, ValueError):
                pass
        return {}

    def _normalize_attrs_for_prompt(
        self,
        attrs: Dict[str, Any],
        schema_keys: List[str],
        max_attrs: int = MAX_ATTRS_PER_PRODUCT,
    ) -> Dict[str, Any]:
        if not attrs:
            return {}
        result: Dict[str, Any] = {}
        if schema_keys:
            for key in schema_keys:
                if len(result) >= max_attrs:
                    break
                val = attrs.get(key)
                if val is not None and str(val).strip():
                    result[key] = val
        for key, val in attrs.items():
            if len(result) >= max_attrs:
                break
            if key in result:
                continue
            if key.lower() in ATTRS_BLACKLIST:
                continue
            val_str = str(val) if val is not None else ""
            if "<" in val_str and ">" in val_str:
                continue
            if val_str.strip():
                result[key] = val
        return result

    def _normalize_product(self, res: Dict[str, Any]) -> Dict[str, Any]:
        p = res.get("product") or res.get("data") or res

        attrs    = self._parse_attributes(p.get("attributes"))
        variants = self._parse_variants(p.get("variants"))
        first_v  = variants[0] if variants else {}

        raw_price_min = p.get("price_min")
        if raw_price_min is None:
            raw_price_min = first_v.get("price")
        if raw_price_min is None:
            raw_price_min = p.get("price")
        price_min = self._parse_price_limit(raw_price_min)

        raw_price_max = p.get("price_max")
        price_max     = self._parse_price_limit(raw_price_max)

        url = p.get("product_url") or first_v.get("product_url", "")

        return {
            "product_id":   p.get("product_id") or p.get("id", ""),
            "title":        p.get("title") or p.get("name", "Товар"),
            "price_min":    price_min,
            "price_max":    price_max,
            "price":        price_min,
            "brand":        p.get("brand", ""),
            "url":          url,
            "image_url":    p.get("image_url", ""),
            "search_blob":  p.get("search_blob", ""),
            "attributes":   attrs,
            "availability": p.get("availability", "instock"),
        }

    # ── Сборка динамического контекста ────────────────────────────────────────

    def _build_dynamic_context(self, ctx: StoreContext, intent: Dict[str, Any]) -> str:
        profile         = getattr(ctx, "profile", {})
        store_bio       = ctx.get_store_bio()
        schema_keys     = getattr(ctx, "schema_keys", [])
        language        = getattr(ctx, "language", "Ukrainian")
        currency        = getattr(ctx, "currency", "грн")

        price_analytics = profile.get("price_analytics", {})
        price_min_store = price_analytics.get("min", 0)
        price_max_store = price_analytics.get("max", 0)
        price_avg_store = price_analytics.get("avg", 0)

        brand_matrix = profile.get("brand_matrix", {})
        top_brands   = brand_matrix.get("top_brands", [])

        lang_instruction = (
            "Відповідай ВИКЛЮЧНО УКРАЇНСЬКОЮ МОВОЮ. Жодного слова іншою мовою."
            if language == "Ukrainian"
            else "Отвечай ИСКЛЮЧИТЕЛЬНО НА РУССКОМ ЯЗЫКЕ. Никакого другого языка."
        )

        entities        = intent.get("entities", {})
        detected_cat    = entities.get("category", "")
        detected_brand  = entities.get("brand", "") or entities.get("Виробник", "")
        raw_price_limit = entities.get("price_limit") or entities.get("max_price", "")
        price_limit_f   = self._parse_price_limit(raw_price_limit)
        price_limit_str = f"{price_limit_f} {currency}" if price_limit_f else ""

        schema_block = (
            f"Доступні фільтри товарів: {', '.join(schema_keys[:10])}"
            if schema_keys
            else "Фільтри: категорія, бренд, ціна"
        )
        brands_block = (
            f"Топ-бренди магазину (виділяти в презентації): {', '.join(top_brands[:7])}"
            if top_brands
            else ""
        )

        intent_parts: List[str] = []
        if detected_cat:
            intent_parts.append(f"Категорія: {detected_cat}")
        if detected_brand:
            intent_parts.append(f"Бренд: {detected_brand}")
        if price_limit_str:
            intent_parts.append(f"Ліміт ціни: {price_limit_str}")
        intent_block = (
            "Запит клієнта: " + " | ".join(intent_parts)
            if intent_parts
            else ""
        )

        parts = [f"МАГАЗИН: {store_bio}", schema_block]
        if brands_block:
            parts.append(brands_block)
        parts.append(
            f"Ціновий діапазон магазину: {price_min_store}–{price_max_store} {currency} "
            f"(середня: {price_avg_store} {currency})"
        )
        if intent_block:
            parts.append(intent_block)
        parts.append(f"МОВА ВІДПОВІДІ: {lang_instruction}")

        return "\n".join(parts)

    # ── Динамический промпт ───────────────────────────────────────────────────

    def _build_dynamic_prompt(
        self,
        ctx: StoreContext,
        products: List[Dict[str, Any]],
        intent: Dict[str, Any],
        query: str,
    ) -> Tuple[str, str]:
        profile     = getattr(ctx, "profile", {})
        currency    = getattr(ctx, "currency", "грн")
        schema_keys = getattr(ctx, "schema_keys", [])
        language    = getattr(ctx, "language", "Ukrainian")
        prompts     = getattr(ctx, "kernel_prompts", self._default_prompts)
        top_brands  = [
            b.lower()
            for b in profile.get("brand_matrix", {}).get("top_brands", [])
        ]
        price_avg = float(profile.get("price_analytics", {}).get("avg", 0) or 0)

        raw_price_limit = (
            intent.get("entities", {}).get("price_limit")
            or intent.get("entities", {}).get("max_price")
        )
        price_limit_f = self._parse_price_limit(raw_price_limit)

        store_prompts = getattr(ctx, "prompts", {})
        view_label    = store_prompts.get("view_button", "Переглянути")

        product_cards: List[str] = []
        for i, res in enumerate(products[:MAX_PRODUCTS_IN_PROMPT], 1):
            p     = self._normalize_product(res)
            title = p["title"]
            brand = p["brand"]
            price = p["price"]
            url   = p["url"]

            filtered_attrs = self._normalize_attrs_for_prompt(
                p["attributes"], schema_keys, MAX_ATTRS_PER_PRODUCT
            )

            price_tags: List[str] = []
            if price is not None:
                try:
                    price_f = float(price)
                    if price_avg and price_f < price_avg * 0.9:
                        price_tags.append("💰 Вигідна ціна")
                    if price_limit_f is not None and price_f <= price_limit_f:
                        price_tags.append("✅ В межах бюджету")
                except (ValueError, TypeError):
                    pass

            brand_tag = " 🏆 Перевірений бренд" if (brand and brand.lower() in top_brands) else ""

            attrs_parts: List[str] = []
            for k, v in filtered_attrs.items():
                if v:
                    val_str = ", ".join(v) if isinstance(v, list) else str(v)
                    attrs_parts.append(f"{k}: {val_str}")

            price_str  = f"{price} {currency}" if price is not None else "---"
            card_lines = [
                f"{i}. {title}{brand_tag}",
                f"   Ціна: {price_str}" + (
                    f" ({', '.join(price_tags)})" if price_tags else ""
                ),
            ]
            if brand:
                card_lines.append(f"   Бренд: {brand}")
            if attrs_parts:
                card_lines.append(f"   Характеристики: {'; '.join(attrs_parts)}")
            if url:
                card_lines.append(f"   Посилання: {url}")
            product_cards.append("\n".join(card_lines))

        products_block      = "\n\n".join(product_cards) if product_cards else "Товарів не знайдено."
        store_context_block = self._build_dynamic_context(ctx, intent)

        role  = prompts.get("dynamic_prompt_role", "You are an expert sales consultant.")
        rules = prompts.get("dynamic_prompt_rules", [])
        rules_text = "\n".join(f"- {r.format(max_products=MAX_PRODUCTS_IN_PROMPT, view_label=view_label)}" for r in rules)

        system_prompt = (
            f"{store_context_block}\n\n"
            f"РОЛЬ: {role}\n\n"
            f"ПРАВИЛА:\n{rules_text}\n\n"
            f"ЗНАЙДЕНІ ТОВАРИ:\n{products_block}"
        )

        user_message = query or "Покажи знайдені товари."
        return system_prompt, user_message

    # ── Валидация ─────────────────────────────────────────────────────────────

    def _validate_llm_response(self, text: str, ctx: StoreContext) -> bool:
        if not text or len(text.strip()) < LLM_MIN_RESPONSE_LEN:
            return False
        has_price = bool(re.search(r'\d{2,}', text))
        has_url   = bool(re.search(r'https?://', text))
        if not has_price and not has_url:
            return False
        language = getattr(ctx, "language", "Ukrainian")
        if language == "Ukrainian":
            ukr_chars = len(re.findall(r'[іїєґ]', text, re.IGNORECASE))
            ru_chars  = len(re.findall(r'[ыэёъ]', text, re.IGNORECASE))
            total_lang_markers = ukr_chars + ru_chars
            if total_lang_markers > 0:
                ru_ratio = ru_chars / total_lang_markers
                if ru_ratio > LANG_RU_RATIO_THRESHOLD:
                    return False
        return True

    def _count_valid_products(self, products: List[Dict[str, Any]]) -> int:
        count = 0
        for res in products:
            p     = self._normalize_product(res)
            title = p.get("title", "").strip()
            url   = p.get("url", "").strip()
            price = p.get("price")
            if title and title != "Товар" and (url or price is not None):
                count += 1
        return count

    # ── format_products ───────────────────────────────────────────────────────

    async def format_products(
        self,
        raw_text: str,
        products: List[Dict[str, Any]],
        user_id: Any,
        ctx: Optional[StoreContext] = None,
        slug: Optional[str] = None,
        intent: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        slug_for_log = getattr(ctx, "slug", "?") if ctx else (slug or "?")

        if not raw_text or not raw_text.strip():
            logger.warning(f"[{slug_for_log}] format_products: empty raw_text.")
            return None

        if ctx is None:
            target_slug = slug or (self.registry.get_all_slugs() or [None])[0]
            if not target_slug:
                logger.error("[kernel] format_products: no active store found.")
                return None
            ctx = self.registry.get_context(target_slug)
            if not ctx:
                logger.error(f"[kernel] format_products: context not found for {target_slug}.")
                return None
            slug_for_log = target_slug

        if not products:
            logger.info(f"[{slug_for_log}] format_products: empty products → Light Path (template).")
            fallback_reason = intent.get("fallback_reason") if intent else None
            return await self._build_template_response(ctx, [], raw_text, fallback_reason=fallback_reason)

        if intent is None:
            intent = {"action": "SEARCH", "entities": {}}

        entities   = intent.get("entities", {}) or {}
        n_entities = len([
            v for v in entities.values()
            if v and str(v).lower() not in ("none", "null", "any", "")
        ])
        is_complex = (
            n_entities >= COMPLEX_ENTITIES_THRESHOLD
            or len(raw_text.strip()) > COMPLEX_QUERY_LENGTH
        )

        valid_count = self._count_valid_products(products)
        if valid_count < MIN_VALID_PRODUCTS_FOR_LLM:
            logger.info(
                f"[{slug_for_log}] format_products: only {valid_count} valid products "
                f"(< {MIN_VALID_PRODUCTS_FOR_LLM}) → Light Path (template)."
            )
            return await self._build_template_response(ctx, products, raw_text)

        if is_complex:
            try:
                client, model = await asyncio.wait_for(
                    self.selector.get_heavy(), timeout=5.0
                )
                system_prompt, user_message = self._build_dynamic_prompt(
                    ctx, products, intent, raw_text
                )
                max_tokens = min(
                    BASE_MAX_TOKENS + len(products[:MAX_PRODUCTS_IN_PROMPT]) * TOKENS_PER_PRODUCT,
                    MAX_TOKENS_CAP,
                )
                logger.info(
                    f"[{slug_for_log}] format_products: Heavy Path "
                    f"({model}), products={len(products)}, valid={valid_count}, "
                    f"entities={n_entities}, prompt_chars={len(system_prompt)}, "
                    f"max_tokens={max_tokens}."
                )
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": user_message},
                        ],
                        temperature=0.4,
                        max_tokens=max_tokens,
                    ),
                    timeout=25.0,
                )
                result = response.choices[0].message.content or ""

                if self._validate_llm_response(result, ctx):
                    logger.info(f"[{slug_for_log}] format_products: Heavy Path OK, response_chars={len(result)}.")
                    return result.strip()

                logger.warning(f"[{slug_for_log}] format_products: Heavy Path failed validation → template.")

            except asyncio.TimeoutError:
                logger.warning(f"[{slug_for_log}] format_products: Heavy Path timeout → template.")
            except Exception as e:
                logger.warning(f"[{slug_for_log}] format_products: Heavy Path error: {e} → template.")
        else:
            logger.info(
                f"[{slug_for_log}] format_products: Light Path "
                f"(entities={n_entities}, query_len={len(raw_text.strip())})."
            )

        result = await self._build_template_response(
            ctx, products, raw_text,
            fallback_reason=intent.get("fallback_reason") if intent else None,
        )
        logger.info(f"[{slug_for_log}] format_products: Light Path (template), response_chars={len(result)}.")
        return result

    # ── handle_message ────────────────────────────────────────────────────────

    async def handle_message(
        self, text: str, chat_id: str, slug: Optional[str] = None
    ) -> Dict[str, Any]:
        """Industrial Entry Point for test_chat.py and CLI."""
        if not self.llm_ready.is_set():
            await asyncio.wait_for(self.llm_ready.wait(), timeout=10.0)

        slugs = self.registry.get_all_slugs()
        if not slugs:
            return {"text": "Ошибка: Магазины не загружены", "status": "ERROR"}

        target_slug = slug or slugs[0]
        ctx         = self.registry.get_context(target_slug)

        if not ctx:
            return {"text": f"Ошибка: Магазин {target_slug} не найден", "status": "ERROR"}

        await self.wait_for_store(target_slug)
        start_time_perf = time.perf_counter()

        try:
            if hasattr(ctx, "dialog_manager"):
                intent = await ctx.dialog_manager.analyze_intent(
                    user_text=text, chat_id=chat_id,
                )
            else:
                intent = await ctx.analyzer.extract_intent(text, chat_context="")

            action   = intent.get("action", "SEARCH")
            language = getattr(ctx, "language", "Ukrainian")
            prompts  = getattr(ctx, "kernel_prompts", self._default_prompts)

            def _save(role: str, content: str):
                if hasattr(ctx, "dialog_manager"):
                    asyncio.create_task(ctx.dialog_manager.save_history(chat_id, role, content))

            def _result(response_text: str, intent_applied: str, count: int = 0) -> Dict:
                llm_status   = self.selector.get_status()
                active_model = next((k for k, v in llm_status.items() if v != "OFFLINE"), "N/A")
                return {
                    "text":           response_text,
                    "intent_applied": intent_applied,
                    "status":         "SUCCESS",
                    "model":          active_model,
                    "ms":             int((time.perf_counter() - start_time_perf) * 1000),
                    "trace":          {"entities": intent.get("entities", {}), "count": count, "chat_id": chat_id, "slug": target_slug},
                }

            if action == "CONSULT":
                response_text = await self._build_advisory_response(ctx, text, prompts)
                _save("user", text); _save("assistant", response_text)
                return _result(response_text, "CONSULT")

            if action == "TROLL":
                response_text = intent.get("witty_text") or await self._handle_troll_response(ctx, prompts)
                _save("user", text); _save("assistant", response_text)
                return _result(response_text, "TROLL")

            search_results = {"products": [], "status": "EMPTY"}
            if action not in ("OFF_TOPIC", "CHAT"):
                search_results = await ctx.retrieval.search(
                    query=text, entities=intent.get("entities", {}), top_k=5,
                )

            status = search_results.get("status")

            if status == "NO_CATEGORY":
                response_text = self._p(prompts, "no_category", language,
                    categories=', '.join(getattr(ctx, 'profile', {}).get('expertise_fields', [])[:4]))
                _save("user", text); _save("assistant", response_text)
                return _result(response_text, action)

            if status == "NOT_FOUND_SECURE":
                response_text = self._p(prompts, "not_found_secure", language)
                _save("user", text); _save("assistant", response_text)
                return _result(response_text, action)

            products = search_results.get("products", [])

            response_text = await self.format_products(
                raw_text=text, products=products, user_id=chat_id, ctx=ctx, intent=intent,
            )

            if not response_text:
                if action in ("OFF_TOPIC", "CHAT"):
                    response_text = self._p(prompts, "off_topic", language)
                else:
                    response_text = await self._build_template_response(ctx, products, text)

            _save("user", text); _save("assistant", response_text)
            return _result(response_text, action, len(products))

        except Exception as e:
            logger.error(f"Kernel handle_message error: {e}")
            traceback.print_exc()
            language = getattr(ctx, "language", "Ukrainian") if ctx else "Ukrainian"
            return {
                "text":   self._p(self._default_prompts, "error", language),
                "status": "ERROR",
            }

    # ── handle_webhook ────────────────────────────────────────────────────────

    async def handle_webhook(self, slug: str, update: dict) -> bool:
        if not self.llm_ready.is_set():
            await asyncio.wait_for(self.llm_ready.wait(), timeout=20.0)

        await self.wait_for_store(slug)
        message = update.get("message", {})
        user_id = message.get("from", {}).get("id")
        text    = message.get("text", "")

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

    # ── process_request (legacy) ──────────────────────────────────────────────

    async def process_request(
        self, user_text: str, chat_id: str, ctx: Any, limit: int = 5
    ) -> str:
        try:
            u_id = int(chat_id) if str(chat_id).isdigit() else 999
        except Exception:
            u_id = 999
        return await self.get_recommendations(
            ctx=ctx, query=user_text, top_k=limit, user_id=u_id
        )

    # ── get_recommendations ───────────────────────────────────────────────────

    async def get_recommendations(
        self,
        ctx: StoreContext,
        query: str,
        filters: Dict = None,
        top_k: int = 5,
        user_id: int = None,
    ) -> str:
        start_time   = time.time()
        slug_for_log = getattr(ctx, "slug", "?")
        language     = getattr(ctx, "language", "Ukrainian")
        prompts      = getattr(ctx, "kernel_prompts", self._default_prompts)

        _cached_answer_candidate = None
        if hasattr(ctx, 'semantic_cache'):
            cached_answer = ctx.semantic_cache.get_answer(query)
            if cached_answer and not isinstance(cached_answer, (dict, list)):
                _cached_answer_candidate = str(cached_answer)

        dialog_context = ""
        if hasattr(ctx, "dialog_manager") and ctx.dialog_manager:
            dialog_context = await ctx.dialog_manager.get_chat_context(user_id, minutes=5)

        intent = await ctx.analyzer.extract_intent(query, chat_context=dialog_context)

        if filters:
            if not intent.get("entities"):
                intent["entities"] = {}
            intent["entities"].update(filters)

        _intent_entities   = intent.get("entities", {}) or {}
        _nonempty_entities = [
            v for v in _intent_entities.values()
            if v and str(v).lower() not in ("none", "null", "any", "")
        ]
        if _cached_answer_candidate and not _nonempty_entities:
            log_pipeline_step("SEMANTIC_CACHE_HIT", time.time() - start_time)
            logger.info(f"[{slug_for_log}] get_recommendations: cache hit (no entities).")
            return _cached_answer_candidate

        log_pipeline_step("INTENT_ANALYSIS", time.time() - start_time, extra={"intent": intent})

        action = intent.get("action", "SEARCH")

        if action == "CONSULT":
            return await self._build_advisory_response(ctx, query, prompts)

        if action == "TROLL":
            return intent.get("witty_text") or await self._handle_troll_response(ctx, prompts)

        if action in ("CHAT", "OFF_TOPIC"):
            return str(intent.get("response_text") or self._p(prompts, "off_topic", language))

        search_results = await ctx.retrieval.search(
            query=query, entities=intent.get("entities", {}), top_k=top_k * 3,
        )

        if search_results.get("status") == "NO_CATEGORY":
            return self._p(prompts, "no_category", language,
                categories=', '.join(getattr(ctx, 'profile', {}).get('expertise_fields', [])[:4]))

        if search_results.get("status") == "NOT_FOUND_SECURE":
            return self._p(prompts, "not_found_secure", language)

        if search_results.get("fallback_reason"):
            intent["fallback_reason"] = search_results["fallback_reason"]

        final_products = await ctx.dialog_manager.process_search_pipeline(
            chat_id=str(user_id), search_response=search_results, intent=intent, top_k=top_k,
        )

        confidence_result = confidence_evaluate(
            search_result={**search_results, "products": final_products},
            intent=intent,
            user_query=query,
            profile=getattr(ctx, "profile", None),
            category_map=getattr(ctx, "category_map", None),
        )
        log_pipeline_step("CONFIDENCE", time.time() - start_time, extra=confidence_result)

        response_text = await self.format_products(
            raw_text=query, products=final_products, user_id=user_id, ctx=ctx, intent=intent,
        )

        if not response_text:
            logger.warning(f"[{slug_for_log}] get_recommendations: format_products returned None → fallback.")
            response_text = self._p(prompts, "off_topic", language)

        if hasattr(ctx, 'semantic_cache') and final_products and len(response_text) > 20:
            ctx.semantic_cache.add(query, response_text)

        return response_text

    # ── Шаблонный ответ ───────────────────────────────────────────────────────

    async def _build_template_response(
        self, ctx: StoreContext, products: List[Dict], query: str,
        fallback_reason: Optional[str] = None,
    ) -> str:
        language    = getattr(ctx, "language", "Ukrainian")
        prompts     = getattr(ctx, "kernel_prompts", self._default_prompts)
        store_p     = getattr(ctx, "prompts", {})
        view_label  = store_p.get("view_button", "Переглянути")
        price_label = store_p.get("price_label", "Ціна")
        currency    = getattr(ctx, "currency", "грн")

        if not products:
            return self._p(prompts, "no_results", language)

        if fallback_reason in ("no_results_with_properties", "no_results_without_properties",
                                "no_results_faiss_only"):
            intro = self._p(prompts, "similar_found", language)
        else:
            intro = self._p(prompts, "results_found", language)

        parts = [intro, ""]

        for i, res in enumerate(products[:MAX_PRODUCTS_IN_PROMPT], 1):
            p         = self._normalize_product(res)
            price_str = f"{p['price']} {currency}" if p['price'] is not None else "---"
            line      = f"{i}. 🛍 **{p['title']}**\n   {price_label}: {price_str}"
            if p['url']:
                line += f"\n   [{view_label}]({p['url']})"
            parts.append(line)

        parts.append(f"\n{self._p(prompts, 'results_footer', language)}")
        return "\n".join(parts)

    # ── Вспомогательные методы ────────────────────────────────────────────────

    async def _build_advisory_response(
        self, ctx: StoreContext, query: str, prompts: Dict
    ) -> str:
        language  = getattr(ctx, "language", "Ukrainian")
        store_bio = ctx.get_store_bio() if hasattr(ctx, "get_store_bio") else ""
        lang_note = "Пиши українською." if language == "Ukrainian" else "Пиши на русском."

        prompt_template = prompts.get("advisory_prompt", self._default_prompts.get("advisory_prompt", ""))
        prompt = prompt_template.format(store_bio=store_bio, lang_note=lang_note)

        try:
            client, model = await self.selector.get_fast()
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user",   "content": query},
                    ],
                    temperature=0.5,
                    max_tokens=200,
                ),
                timeout=10.0,
            )
            result = response.choices[0].message.content
            if result and result.strip():
                return result.strip()
        except Exception as e:
            logger.warning(f"[kernel] _build_advisory_response error: {e}")

        return self._p(prompts, "advisory_fallback", language)

    async def _handle_troll_response(self, ctx: StoreContext, prompts: Dict) -> str:
        language  = getattr(ctx, "language", "Ukrainian")
        store_bio = ctx.get_store_bio() if hasattr(ctx, "get_store_bio") else ""
        lang_note = "Пиши українською." if language == "Ukrainian" else "Пиши на русском."

        prompt_template = prompts.get("troll_prompt", self._default_prompts.get("troll_prompt", ""))
        prompt = prompt_template.format(store_bio=store_bio, lang_note=lang_note)

        try:
            client, model = await self.selector.get_fast()
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.7,
                    max_tokens=120,
                ),
                timeout=8.0,
            )
            result = response.choices[0].message.content
            if result and result.strip():
                return result.strip()
        except Exception as e:
            logger.warning(f"[kernel] _handle_troll_response error: {e}")

        return self._p(prompts, "troll_fallback", language)

    def _load_store_negative_examples(self, base_path: str) -> list:
        patch_path = os.path.join(base_path, "fsm_soft_patch.json")
        if os.path.exists(patch_path):
            try:
                with open(patch_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else data.get("troll_patterns", [])
            except Exception:
                pass
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
        message    = update.get("message", {})
        text       = message.get("text")
        if not text or len(text.strip()) < 2:
            return
        store_lang = getattr(engine.ctx, "language", "Ukrainian")
        if store_lang == "Ukrainian":
            try:
                if await self.translator.detect_language(text) == "ru":
                    translated = await self.translator.translate(text, target_lang="uk")
                    if translated:
                        update["message"]["original_text"] = text
                        update["message"]["text"]          = translated
            except Exception:
                pass

    async def wait_for_store(self, slug: str, timeout: float = 30.0) -> bool:
        ctx = self.registry.get_context(slug)
        if not ctx:
            return False
        if hasattr(ctx, 'data_ready'):
            try:
                await asyncio.wait_for(ctx.data_ready.wait(), timeout=timeout)
                return True
            except asyncio.TimeoutError:
                logger.error(f"❌ Timeout waiting for store {slug}")
        return True

    async def close(self):
        logger.info("⏳ Closing Kernel resources...")
        if self.selector:
            try:
                if asyncio.iscoroutinefunction(self.selector.close):
                    await self.selector.close()
                else:
                    self.selector.close()
            except Exception:
                pass
        logger.info("✅ Kernel resources released.")

    def get_all_active_slugs(self) -> List[str]:
        return self.registry.get_all_slugs() if self.registry else []

    def __repr__(self):
        return f"<UkrSellKernel v8.6.0 stores={len(self.get_all_active_slugs())}>"