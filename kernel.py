# /root/ukrsell_v4/kernel.py v8.5.4
"""
UkrSell AI Kernel v8.5.4.

v8.5.4:

"""

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
from engine.base import StoreEngine


# ── Константы ─────────────────────────────────────────────────────────────────

BASE_MAX_TOKENS    = 800
TOKENS_PER_PRODUCT = 120
MAX_TOKENS_CAP     = 2000

MAX_PRODUCTS_IN_PROMPT = 5
MAX_ATTRS_PER_PRODUCT  = 3

COMPLEX_ENTITIES_THRESHOLD = 2
COMPLEX_QUERY_LENGTH       = 40

_ATTRS_BLACKLIST = frozenset({
    "id", "sku", "uid", "uuid", "product_id", "item_id",
    "created_at", "updated_at", "last_updated",
    "html", "description_html", "meta", "seo",
    "position", "sort_order", "weight_g", "volume_ml",
})

LLM_MIN_RESPONSE_LEN    = 60
LANG_RU_RATIO_THRESHOLD = 0.35
MIN_VALID_PRODUCTS_FOR_LLM = 2


class UkrSellKernel:
    """AI Kernel v8.5.3."""

    def __init__(self, model_name: str = 'intfloat/multilingual-e5-small'):
        if HF_TOKEN:
            os.environ["HF_TOKEN"] = HF_TOKEN
            logger.info("HF_TOKEN set in environment.")

        self.model_name = model_name
        self._request_cache: Dict[str, float] = {}
        self._cache_ttl = 5

        self.model      = None
        self.translator = TextTranslator()
        self.selector   = LLMSelector()
        self.registry   = None
        self.llm_ready  = asyncio.Event()

    # ── Инициализация ─────────────────────────────────────────────────────────

    async def initialize(self):
        start_time = time.time()
        logger.info(f"🚀 Initializing UkrSell Kernel v8.5.4. Model: {self.model_name}")

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
            if key.lower() in _ATTRS_BLACKLIST:
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

        products_for_prompt = products[:MAX_PRODUCTS_IN_PROMPT]

        product_cards: List[str] = []
        for i, res in enumerate(products_for_prompt, 1):
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
        prompts             = getattr(ctx, "prompts", {})
        view_label          = prompts.get("view_button", "Переглянути")

        system_prompt = (
            f"{store_context_block}\n\n"
            "РОЛЬ: Ти — експертний продавець-консультант. "
            "Твоє завдання — коротко, переконливо і доброзичливо презентувати знайдені товари.\n\n"
            "ПРАВИЛА:\n"
            "- Виділяй переваги 🏆 Перевірений бренд та 💰 Вигідна ціна якщо вони є.\n"
            f"- Максимум {MAX_PRODUCTS_IN_PROMPT} позицій у відповіді.\n"
            "- Кожна позиція: назва, ціна, короткий аргумент чому варто.\n"
            f"- Кнопка «{view_label}» або пряме посилання для кожного товару.\n"
            "- Завершуй коротким питанням для продовження діалогу.\n"
            "- СУВОРА ЗАБОРОНА: не вигадуй назви, ціни або бренди яких немає в списку.\n"
            "- СУВОРА ЗАБОРОНА: не перемикай мову відповіді.\n\n"
            f"ЗНАЙДЕНІ ТОВАРИ:\n{products_block}"
        )

        user_message = query or "Покажи знайдені товари."
        return system_prompt, user_message

    # ── Валидация и вспомогательные методы форматирования ────────────────────

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

    # ── format_products — единственная точка LLM для ответа ──────────────────

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
                prompt_size = len(system_prompt)
                logger.info(
                    f"[{slug_for_log}] format_products: Heavy Path "
                    f"({model}), products={len(products)}, valid={valid_count}, "
                    f"entities={n_entities}, prompt_chars={prompt_size}, "
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
                    logger.info(
                        f"[{slug_for_log}] format_products: Heavy Path OK, "
                        f"response_chars={len(result)}."
                    )
                    return result.strip()

                logger.warning(
                    f"[{slug_for_log}] format_products: Heavy Path response failed validation "
                    f"(len={len(result)}) → Fallback to template."
                )

            except asyncio.TimeoutError:
                logger.warning(
                    f"[{slug_for_log}] format_products: Heavy Path timeout → "
                    f"Fallback to template."
                )
            except Exception as e:
                logger.warning(
                    f"[{slug_for_log}] format_products: Heavy Path error: {e} → "
                    f"Fallback to template."
                )
        else:
            logger.info(
                f"[{slug_for_log}] format_products: Light Path "
                f"(entities={n_entities}, query_len={len(raw_text.strip())})."
            )

        result = await self._build_template_response(
            ctx, products, raw_text,
            fallback_reason=intent.get("fallback_reason") if intent else None,
        )
        logger.info(
            f"[{slug_for_log}] format_products: Light Path (template), "
            f"response_chars={len(result)}."
        )
        return result

    # ── handle_message ────────────────────────────────────────────────────────

    async def handle_message(
        self, text: str, chat_id: str, slug: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Industrial Entry Point for test_chat.py and CLI.

        FIX v8.5.3:
          1. Використовує dialog_manager.analyze_intent замість analyzer.extract_intent.
             Це активує _normalize_entities, few-shot промпт, L1+L2 intent cache.
          2. Зберігає user + assistant повідомлення в chat_history.
             Виправляє "Zero Memory" — follow-up запити бачать контекст.
          3. Використовує format_products для форматування замість synthesize_response.
        """
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
            # FIX: використовуємо dialog_manager.analyze_intent — активує всі фікси v7.7.3
            # (few-shot промпт, _normalize_entities, L1+L2 cache, follow-up merge)
            if hasattr(ctx, "dialog_manager"):
                intent = await ctx.dialog_manager.analyze_intent(
                    user_text=text,
                    chat_id=chat_id,
                )
            else:
                # fallback для сторів без dialog_manager
                history_context = ""
                intent = await ctx.analyzer.extract_intent(
                    text, chat_context=history_context
                )

            action = intent.get("action", "SEARCH")

            # FIX #1: Intent gate — CONSULT не идёт в retrieval
            if action == "CONSULT":
                response_text = await self._build_advisory_response(ctx, text)
                if hasattr(ctx, "dialog_manager"):
                    asyncio.create_task(ctx.dialog_manager.save_history(chat_id, "user", text))
                    asyncio.create_task(ctx.dialog_manager.save_history(chat_id, "assistant", response_text))
                llm_status   = self.selector.get_status()
                active_model = next((k for k, v in llm_status.items() if v != "OFFLINE"), "N/A")
                return {
                    "text":           response_text,
                    "intent_applied": "CONSULT",
                    "status":         "SUCCESS",
                    "model":          active_model,
                    "ms":             int((time.perf_counter() - start_time_perf) * 1000),
                    "trace":          {"entities": intent.get("entities", {}), "count": 0, "chat_id": chat_id, "slug": target_slug},
                }

            search_results = {"products": [], "status": "EMPTY"}
            if action not in ("OFF_TOPIC", "TROLL"):
                search_results = await ctx.retrieval.search(
                    query=text,
                    entities=intent.get("entities", {}),
                    top_k=5,
                )

            # FIX #2: NO_CATEGORY → честный ответ без мусора
            if search_results.get("status") == "NO_CATEGORY":
                response_text = self._build_no_category_response(ctx, text)
                if hasattr(ctx, "dialog_manager"):
                    asyncio.create_task(ctx.dialog_manager.save_history(chat_id, "user", text))
                    asyncio.create_task(ctx.dialog_manager.save_history(chat_id, "assistant", response_text))
                llm_status   = self.selector.get_status()
                active_model = next((k for k, v in llm_status.items() if v != "OFFLINE"), "N/A")
                return {
                    "text":           response_text,
                    "intent_applied": action,
                    "status":         "SUCCESS",
                    "model":          active_model,
                    "ms":             int((time.perf_counter() - start_time_perf) * 1000),
                    "trace":          {"entities": intent.get("entities", {}), "count": 0, "chat_id": chat_id, "slug": target_slug},
                }

            products = search_results.get("products", [])

            # FIX: format_products — єдина точка LLM для форматування відповіді
            response_text = await self.format_products(
                raw_text=text,
                products=products,
                user_id=chat_id,
                ctx=ctx,
                intent=intent,
            )

            # Якщо format_products повернув None (пустий текст) — fallback
            if not response_text:
                if action == "TROLL":
                    response_text = await self._handle_troll_response(ctx)
                elif action in ("OFF_TOPIC", "CHAT"):
                    response_text = self._get_off_topic_response(ctx)
                else:
                    response_text = await self._build_template_response(ctx, products, text)

            # FIX: зберігаємо обидва повідомлення для контексту follow-up запитів
            if hasattr(ctx, "dialog_manager"):
                asyncio.create_task(
                    ctx.dialog_manager.save_history(chat_id, "user", text)
                )
                asyncio.create_task(
                    ctx.dialog_manager.save_history(chat_id, "assistant", response_text)
                )

            llm_status   = self.selector.get_status()
            active_model = "N/A"
            for k, v in llm_status.items():
                if v != "OFFLINE":
                    active_model = k
                    break

            return {
                "text":           response_text,
                "intent_applied": action,
                "status":         "SUCCESS",
                "model":          active_model,
                "ms":             int((time.perf_counter() - start_time_perf) * 1000),
                "trace": {
                    "entities": intent.get("entities", {}),
                    "count":    len(products),
                    "chat_id":  chat_id,
                    "slug":     target_slug,
                },
            }

        except Exception as e:
            logger.error(f"Kernel handle_message error: {e}")
            traceback.print_exc()
            return {
                "text":   "Виникла технічна помилка. Спробуйте пізніше.",
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

        # Semantic cache — пропускаємо якщо є entities
        _cached_answer_candidate = None
        if hasattr(ctx, 'semantic_cache'):
            cached_answer = ctx.semantic_cache.get_answer(query)
            if cached_answer and not isinstance(cached_answer, (dict, list)):
                _cached_answer_candidate = str(cached_answer)

        # Intent через dialog_manager (всі фікси v7.7.3 активні)
        dialog_context = ""
        if hasattr(ctx, "dialog_manager") and ctx.dialog_manager:
            dialog_context = await ctx.dialog_manager.get_chat_context(user_id, minutes=5)

        intent = await ctx.analyzer.extract_intent(
            query, chat_context=dialog_context
        )

        if filters:
            if not intent.get("entities"):
                intent["entities"] = {}
            intent["entities"].update(filters)

        _intent_entities  = intent.get("entities", {}) or {}
        _nonempty_entities = [
            v for v in _intent_entities.values()
            if v and str(v).lower() not in ("none", "null", "any", "")
        ]
        if _cached_answer_candidate and not _nonempty_entities:
            log_pipeline_step("SEMANTIC_CACHE_HIT", time.time() - start_time)
            logger.info(f"[{slug_for_log}] get_recommendations: cache hit (no entities).")
            return _cached_answer_candidate

        log_pipeline_step(
            "INTENT_ANALYSIS",
            time.time() - start_time,
            extra={"intent": intent},
        )

        action = intent.get("action", "SEARCH")

        # FIX #1: CONSULT intent gate — advisory без retrieval
        if action == "CONSULT":
            return await self._build_advisory_response(ctx, query)

        if action == "TROLL":
            return await self._handle_troll_response(ctx)
        if action in ("CHAT", "OFF_TOPIC"):
            return str(
                intent.get("response_text")
                or self._get_off_topic_response(ctx)
            )

        search_results = await ctx.retrieval.search(
            query=query,
            entities=intent.get("entities", {}),
            top_k=top_k * 3,
        )

        # FIX #2: NO_CATEGORY → честный ответ
        if search_results.get("status") == "NO_CATEGORY":
            return self._build_no_category_response(ctx, query)

        # Пробрасываем fallback_reason в intent для _build_template_response
        if search_results.get("fallback_reason"):
            intent["fallback_reason"] = search_results["fallback_reason"]

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
        log_pipeline_step("CONFIDENCE", time.time() - start_time, extra=confidence_result)

        response_text = await self.format_products(
            raw_text=query,
            products=final_products,
            user_id=user_id,
            ctx=ctx,
            intent=intent,
        )

        if not response_text:
            logger.warning(
                f"[{slug_for_log}] get_recommendations: format_products returned None → "
                f"using empty fallback."
            )
            response_text = self._get_off_topic_response(ctx)

        if (
            hasattr(ctx, 'semantic_cache')
            and final_products
            and len(response_text) > 20
        ):
            ctx.semantic_cache.add(query, response_text)

        return response_text

    # ── Шаблонный ответ (fallback) ────────────────────────────────────────────

    async def _build_template_response(
        self, ctx: StoreContext, products: List[Dict], query: str,
        fallback_reason: Optional[str] = None,
    ) -> str:
        language    = getattr(ctx, "language", "Ukrainian")
        view_label  = ctx.prompts.get("view_button", "Переглянути")
        price_label = ctx.prompts.get("price_label", "Ціна")
        currency    = getattr(ctx, "currency", "грн")

        # FIX #5/#7: контекстный заголовок вместо "Чудовий вибір!"
        if not products:
            intro = (
                "На жаль, нічого не знайдено за вашим запитом. 😔"
                if language == "Ukrainian"
                else "К сожалению, по вашему запросу ничего не найдено. 😔"
            )
            return intro
        elif fallback_reason in ("no_results_with_properties", "no_results_without_properties",
                                  "no_results_faiss_only"):
            intro = (
                "Саме такого товару немає, але ось схожі варіанти:"
                if language == "Ukrainian"
                else "Именно такого товара нет, но вот похожие варианты:"
            )
        else:
            intro = (
                "Ось що вдалося знайти:"
                if language == "Ukrainian"
                else "Вот что удалось найти:"
            )

        parts = [intro, ""]

        for i, res in enumerate(products[:MAX_PRODUCTS_IN_PROMPT], 1):
            p     = self._normalize_product(res)
            name  = p["title"]
            price = p["price"]
            url   = p["url"]

            price_str = f"{price} {currency}" if price is not None else "---"
            line      = f"{i}. 🛍 **{name}**\n   {price_label}: {price_str}"
            if url:
                line += f"\n   [{view_label}]({url})"
            parts.append(line)

        footer = (
            "Підсказати щось конкретне щодо цих моделей?"
            if language == "Ukrainian"
            else "Подсказать детали по этим моделям?"
        )
        parts.append(f"\n{footer}")
        return "\n".join(parts)

    # ── Вспомогательные методы ────────────────────────────────────────────────

    async def _build_advisory_response(self, ctx: StoreContext, query: str) -> str:
        """CONSULT intent — советник без показа товаров через retrieval."""
        language  = getattr(ctx, "language", "Ukrainian")
        store_bio = ctx.get_store_bio() if hasattr(ctx, "get_store_bio") else ""
        lang_note = "Пиши українською." if language == "Ukrainian" else "Пиши на русском."
        prompt = (
            f"{store_bio}\n{lang_note}\n"
            "Покупець просить консультацію. "
            "Дай коротку фахову пораду (2-3 речення). "
            "Якщо товару немає в асортименті — чесно скажи і запропонуй альтернативу з магазину. "
            "Завершуй питанням для уточнення потреби."
        )
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
        language = getattr(ctx, "language", "Ukrainian")
        return (
            "На жаль, такого товару зараз немає в нашому асортименті. "
            "Чим ще можу допомогти? 🙂"
            if language == "Ukrainian"
            else "К сожалению, такого товара сейчас нет в нашем ассортименте. Чем ещё помочь? 🙂"
        )

    def _build_no_category_response(self, ctx: StoreContext, query: str) -> str:
        """NO_CATEGORY статус — честный ответ без мусора."""
        language = getattr(ctx, "language", "Ukrainian")
        if language == "Ukrainian":
            return (
                "На жаль, такого товару немає в нашому асортименті. 😔\n"
                "Можу підібрати щось із наявних категорій: "
                f"{', '.join(getattr(ctx, 'profile', {}).get('expertise_fields', [])[:4])}. "
                "Що вас цікавить?"
            )
        return (
            "К сожалению, такого товара нет в нашем ассортименте. 😔\n"
            "Могу предложить из доступных категорий: "
            f"{', '.join(getattr(ctx, 'profile', {}).get('expertise_fields', [])[:4])}. "
            "Что вас интересует?"
        )

    async def _handle_troll_response(self, ctx: StoreContext) -> str:
        language  = getattr(ctx, "language", "Ukrainian")
        store_bio = ctx.get_store_bio() if hasattr(ctx, "get_store_bio") else ""
        lang_note = (
            "Пиши українською." if language == "Ukrainian" else "Пиши на русском."
        )
        prompt = (
            f"{store_bio}\n{lang_note}\n"
            "Користувач написав щось абсурдне або тролить. "
            "Відповідь: дотепна, тепла, 1–2 речення. "
            "М'яко поверни до реальних товарів магазину."
        )
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
        return "🙃 Цікавий запит! Може, підберемо щось реальне для вас?"

    def _get_off_topic_response(self, ctx: StoreContext) -> str:
        language = getattr(ctx, "language", "Ukrainian")
        if language == "Ukrainian":
            return "Вибачте, це поза нашою спеціалізацією. Чим ще можу допомогти? 🙂"
        return "Извините, это не по нашей специализации. Чем ещё могу помочь? 🙂"

    def _load_store_negative_examples(self, base_path: str) -> list:
        patch_path = os.path.join(base_path, "fsm_soft_patch.json")
        if os.path.exists(patch_path):
            try:
                with open(patch_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return (
                        data
                        if isinstance(data, list)
                        else data.get("troll_patterns", [])
                    )
            except Exception:
                pass
        return []

    def _is_duplicate(self, user_id: Any, text: str) -> bool:
        if not text:
            return False
        request_hash = hashlib.md5(f"{user_id}:{text}".encode()).hexdigest()
        now = time.time()
        self._request_cache = {
            k: v for k, v in self._request_cache.items()
            if now - v < self._cache_ttl
        }
        if request_hash in self._request_cache:
            return True
        self._request_cache[request_hash] = now
        return False

    async def _normalize_update_language(
        self, engine: StoreEngine, update: Dict[str, Any]
    ):
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
        return f"<UkrSellKernel v8.5.4 stores={len(self.get_all_active_slugs())}>"