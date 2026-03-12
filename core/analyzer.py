# /root/ukrsell_v4/core/analyzer.py v7.4.0
"""
Layer 3 – Semantic Intent Parser & Generative Synthesizer.

Changelog:
    v7.3.3  ASK_CLARIFICATION: catalog_context вставляется в промпт,
            варианты только из CATALOG, убран запрет «Do NOT show products».
    v7.4.0  [ФИКС галлюцинаций]
            - ASK_CLARIFICATION: усилены правила анти-галлюцинации.
              Явный запрет инвентить бренды, модели, характеристики не из CATALOG.
            - ASK_CLARIFICATION: если CATALOG содержит точный match — ПОКАЗЫВАЕМ товар,
              не задаём уточняющий вопрос.
            - SHOW_PRODUCTS: добавлен запрет «Do NOT add products not in CATALOG».
            - NO_RESULTS: добавлено «Do NOT suggest products — only categories».
            - _prepare_products_context: передаёт product_url корректно.
            - extract_intent: возвращает entities в поле "entities" (не в корне)
              для совместимости с confidence.py v1.7.0.
"""

import logging
import json
import asyncio
import re
from typing import List, Dict, Any, Union, Optional
from core.llm_selector import LLMSelector
from core.logger import logger, log_event, log_llm_request, log_llm_response, log_llm_error, log_model_selected
from core.intelligence import safe_extract_json, semantic_guard, deduplicate_products


class Analyzer:
    """
    Analyzer v7.4.0 — Semantic Intent Parser & Generative Synthesizer.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.slug      = getattr(ctx, 'slug', 'unknown')
        self.language  = getattr(ctx, 'language', 'Ukrainian')
        self.currency  = getattr(ctx, 'currency', 'грн')
        self.llm_selector = getattr(ctx, 'selector', None)

        if not self.llm_selector:
            logger.error(f"[{self.slug}] Analyzer CRITICAL: Shared LLMSelector not found in ctx.")

        self.profile   = getattr(ctx, 'profile', {})
        self.store_bio = ctx.get_store_bio() if hasattr(ctx, 'get_store_bio') else ""
        self.store_context_prompt = self.profile.get("store_context_prompt", "")

        logger.info(f"🛠️ [{self.slug}] Analyzer v7.4.0 initialized.")

    async def _wait_for_ready(self):
        if hasattr(self.ctx, 'data_ready'):
            if not self.ctx.data_ready.is_set():
                logger.info(f"[{self.slug}] Analyzer: Waiting for assets...")
                try:
                    await asyncio.wait_for(self.ctx.data_ready.wait(), timeout=15.0)
                except asyncio.TimeoutError:
                    logger.error(f"[{self.slug}] Analyzer: Assets load timeout.")

    def _get_current_schema(self) -> List[str]:
        try:
            if hasattr(self.ctx, 'retrieval') and hasattr(self.ctx.retrieval, 'schema_keys'):
                return self.ctx.retrieval.schema_keys
        except:
            pass
        return []

    async def extract_intent(self, user_query: str, chat_context: str = "") -> Dict[str, Any]:
        """
        STAGE 1: Entity extraction.

        v7.4.0: результат всегда содержит поле "entities" для совместимости
        с confidence.py v1.7.0 (_attr_match_score читает entities.get("entities")).
        """
        await self._wait_for_ready()

        if not self.llm_selector:
            return {"entities": {"target": user_query}, "action": "SEARCH", "properties": {}}

        client, model = await self.llm_selector.get_fast()
        log_model_selected(self.slug, tier="fast", model=model, provider="selector")
        schema_keys = self._get_current_schema()

        schema_instruction = (
            f"Keys: {schema_keys}"
            if schema_keys
            else "Keys: color, price_limit, brand, category, animal."
        )
        context_block = f"\nRECENT CONVERSATION:\n{chat_context}\n" if chat_context else ""

        prompt = (
            "Extract search entities for e-commerce. Return ONLY JSON.\n"
            "Fields:\n"
            "- entities: object with extracted fields\n"
            f"  - {schema_instruction}\n"
            "  - category: obvious shop category (string)\n"
            "  - brand: brand name if mentioned (string)\n"
            "  - animal: dog/cat/both if mentioned (string)\n"
            "  - price_limit: max price as number if mentioned\n"
            "- action: SEARCH | INFO | TROLL | EMOTION | CHAT | OFF_TOPIC\n"
            "- response_text: short reply if action is CHAT/TROLL/OFF_TOPIC (else null)\n"
            f"{context_block}\n"
            f"QUERY: {user_query}"
        )

        log_llm_request(
            self.slug, tier="fast", model=model, provider="selector",
            prompt_preview=f"QUERY: {user_query}", max_tokens=400,
        )
        try:
            import time as _time
            _t0 = _time.perf_counter()
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise JSON intent parser. Output ONLY valid JSON."},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=400,
                temperature=0.0,
            )
            _dur_ms = (_time.perf_counter() - _t0) * 1000
            content = response.choices[0].message.content
            intent  = safe_extract_json(content)

            # Нормализация: если LLM вернул flat dict без "entities" — оборачиваем
            if "entities" not in intent:
                entities_keys = {"category", "brand", "animal", "price_limit", "target", "color"}
                extracted = {k: v for k, v in intent.items() if k in entities_keys}
                rest = {k: v for k, v in intent.items() if k not in entities_keys}
                intent = {"entities": extracted, **rest}

            log_llm_response(
                self.slug, model=model, duration_ms=_dur_ms,
                response_preview=content,
                finish_reason=response.choices[0].finish_reason or "",
                tokens_used=getattr(response.usage, "total_tokens", 0),
            )
            logger.info(
                f"[{self.slug}] INTENT | entities={intent.get('entities')} action={intent.get('action')}"
            )
            return intent

        except Exception as e:
            log_llm_error(
                self.slug, model=model, provider="selector",
                error_type=type(e).__name__, error_msg=str(e),
            )
            return {"entities": {"target": user_query}, "action": "SEARCH", "properties": {}}

    def _prepare_products_context(self, products: List[Dict[str, Any]]) -> str:
        """
        Format product list for LLM context.
        v7.4.0: корректная передача product_url.
        """
        if not products:
            return (
                "Товари не знайдено."
                if self.language.lower() == "ukrainian"
                else "Товары не найдены."
            )

        clean_products = deduplicate_products(products, top_k=5)
        context_lines  = []

        for idx, res in enumerate(clean_products, 1):
            p = (
                res.get("data")
                if isinstance(res, dict) and "data" in res
                else (
                    res.get("product")
                    if isinstance(res, dict) and "product" in res
                    else res
                )
            )

            name      = p.get("title") or p.get("name") or "Товар без назви"
            raw_price = p.get("price", "N/A")
            clean_price = re.sub(r'[^\d.,]', '', str(raw_price)).replace(',', '.')
            display_price = (
                f"{clean_price} {self.currency}"
                if clean_price and clean_price != "."
                else "Ціна за запитом"
            )

            # v7.4.0: product_url приоритет над url/link
            link = (
                p.get("product_url") or
                p.get("link") or
                p.get("url") or
                "#"
            )
            desc  = p.get("description") or p.get("short_description", "Опис відсутній")
            brand = p.get("brand", "")

            attrs = p.get("attributes", [])
            if isinstance(attrs, list):
                attr_str = ", ".join([
                    f"{a.get('key', 'Attr')}: {a.get('value', 'N/A')}"
                    for a in attrs
                    if isinstance(a, dict)
                ])
            elif isinstance(attrs, dict):
                attr_str = ", ".join([f"{k}: {v}" for k, v in attrs.items()])
            else:
                attr_str = str(attrs)

            context_lines.append(
                f"ITEM #{idx} | {name} | {display_price}"
                + (f" | Brand: {brand}" if brand else "")
                + f"\nDESC: {str(desc)[:200]}\nSPECS: {attr_str}\nURL: {link}\n---"
            )

        return "\n".join(context_lines)

    def _check_semantic_conflict(self, category: str) -> bool:
        if not category or not self.profile:
            return False
        expertise = self.profile.get("expertise_fields", [])
        if not expertise:
            return False
        cat_lower = str(category).lower().strip()
        return not any(
            cat_lower in ex.lower() or ex.lower() in cat_lower
            for ex in expertise
        )

    def _build_prompt(
        self,
        mode: str,
        catalog_context: str,
        chat_context: str,
        entities: Dict[str, Any],
        top_categories: Optional[List] = None,
    ) -> str:
        """
        Build dynamic system prompt based on Confidence Engine mode.

        v7.4.0:
          SHOW_PRODUCTS     — жёсткий запрет добавлять товары не из CATALOG
          ASK_CLARIFICATION — если CATALOG содержит точный match — показываем товар,
                              иначе задаём ОДИН уточняющий вопрос только с вариантами из CATALOG.
                              Явный запрет инвентить бренды, модели, характеристики.
          NO_RESULTS        — только категории, никаких выдуманных товаров
        """
        lang_target = "РУССКОМ" if self.language.lower() == "russian" else "УКРАЇНСЬКОЮ"
        is_ua       = self.language.lower() == "ukrainian"

        # category из entities (поддержка нового формата с вложенным "entities")
        ent_inner = entities.get("entities", {}) or {}
        category  = ent_inner.get("category") or entities.get("category", "")

        store_ctx_block = (
            f"\nSTORE PROFILE:\n{self.store_context_prompt}\n"
            if self.store_context_prompt else ""
        )

        has_real_products = "ITEM #" in catalog_context

        # ── SHOW_PRODUCTS ────────────────────────────────────────────────────
        if mode == "SHOW_PRODUCTS":
            logger.info(f"[{self.slug}] PROMPT mode=SHOW_PRODUCTS category={category!r}")
            return (
                f"You are a helpful sales consultant for: {self.store_bio}\n"
                f"{store_ctx_block}"
                f"Language: {lang_target}\n"
                "RULES:\n"
                "1. Present ONLY the products from CATALOG below. NEVER add products not in CATALOG.\n"
                "2. HTML FORMATTING: Use <b>Name</b> and <a href='URL'>Переглянути</a>.\n"
                "3. Be concise. One sentence per product. No pressure.\n"
                "4. If catalog has 1 item — present it, ask if they want other options.\n"
                "5. NEVER invent brands, models, prices, or specs not shown in CATALOG.\n"
                f"CATALOG:\n{catalog_context}\n"
                f"CONVERSATION CONTEXT:\n{chat_context}"
            )

        # ── ASK_CLARIFICATION ────────────────────────────────────────────────
        if mode == "ASK_CLARIFICATION":
            logger.info(
                f"[{self.slug}] PROMPT mode=ASK_CLARIFICATION "
                f"category={category!r} top_categories={top_categories} "
                f"catalog_items={catalog_context.count('ITEM #')} "
                f"has_real_products={has_real_products}"
            )

            clarify_hint = ""
            if category:
                if is_ua:
                    clarify_hint = f"Покупець шукає щось у категорії «{category}». "
                else:
                    clarify_hint = f"Покупатель ищет что-то в категории «{category}». "

            top_cat_block = ""
            if top_categories:
                cat_str = ", ".join([f"{cat} ({prob:.0%})" for cat, prob in top_categories])
                if is_ua:
                    top_cat_block = f"\nНайбільш популярні категорії магазину: {cat_str}\n"
                else:
                    top_cat_block = f"\nНаиболее популярные категории магазина: {cat_str}\n"
                logger.debug(f"[{self.slug}] top_cat_block injected: {cat_str}")

            if has_real_products:
                # Есть реальные товары — показываем И задаём уточнение
                return (
                    f"You are a helpful sales consultant for: {self.store_bio}\n"
                    f"{store_ctx_block}"
                    f"{top_cat_block}"
                    f"Language: {lang_target}\n"
                    f"{clarify_hint}\n"
                    "RULES (STRICT — violations cause hallucinations):\n"
                    "1. The query is unclear or broad — ask ONE clarifying question.\n"
                    "2. Show 2-3 options as bullet points using ONLY items from CATALOG below.\n"
                    "   - Copy name, price and URL exactly from CATALOG. Do NOT paraphrase names.\n"
                    "3. NEVER invent product names, brands, models, or characteristics.\n"
                    "4. NEVER mention products not listed in CATALOG.\n"
                    "5. If CATALOG has a product that closely matches the query — show it first.\n"
                    "6. Be friendly and concise (2-4 sentences total).\n"
                    f"CATALOG (use ONLY these products):\n{catalog_context}\n"
                    f"CONVERSATION CONTEXT:\n{chat_context}"
                )
            else:
                # Нет товаров — только уточняем категорию
                expertise_str = ", ".join(
                    self.profile.get("expertise_fields", [])[:6]
                )
                return (
                    f"You are a helpful sales consultant for: {self.store_bio}\n"
                    f"{store_ctx_block}"
                    f"{top_cat_block}"
                    f"Language: {lang_target}\n"
                    f"{clarify_hint}\n"
                    "RULES:\n"
                    "1. No products found — ask ONE clarifying question to narrow down the search.\n"
                    "2. Suggest 2-3 available CATEGORIES (not products) from the list below.\n"
                    "3. NEVER invent product names, brands, or prices.\n"
                    f"4. Available categories: {expertise_str}\n"
                    "5. Be friendly and short.\n"
                    f"CONVERSATION CONTEXT:\n{chat_context}"
                )

        # ── NO_RESULTS ───────────────────────────────────────────────────────
        logger.info(f"[{self.slug}] PROMPT mode=NO_RESULTS category={category!r}")
        expertise = self.profile.get("expertise_fields", [])
        expertise_str = ", ".join(expertise[:6]) if expertise else ""
        return (
            f"You are a helpful sales consultant for: {self.store_bio}\n"
            f"{store_ctx_block}"
            f"Language: {lang_target}\n"
            "RULES:\n"
            "1. This product or category is NOT available in the store.\n"
            "2. Apologize briefly and honestly.\n"
            "3. NEVER suggest specific products — suggest 1-2 available CATEGORIES only.\n"
            "4. NEVER invent products, brands, or prices.\n"
            f"5. Available categories: {expertise_str}\n"
            "6. Be short and friendly (2-3 sentences max).\n"
            f"CONVERSATION CONTEXT:\n{chat_context}"
        )

    async def synthesize_response(
        self,
        search_results: Dict[str, Any],
        entities: Dict[str, Any],
        user_query: str,
        chat_context: str = "",
        mode: Optional[str] = None,
        top_categories: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        STAGE 2: Final text generation (v7.4.0).
        mode= from Confidence Engine drives dynamic prompt selection.
        top_categories= from confidence_result used in ASK_CLARIFICATION.
        """
        if not self.llm_selector:
            return {"text": "Offline (No Selector)", "status": "ERROR", "count": 0}

        r_data       = search_results or {"status": "NO_RESULTS", "products": []}
        raw_products = r_data.get("products", [])
        intent_action = entities.get("action", "SEARCH")

        # Semantic conflict check
        ent_inner = entities.get("entities", {}) or {}
        if (
            self._check_semantic_conflict(ent_inner.get("category") or entities.get("category"))
            or r_data.get("status") == "SEMANTIC_REJECT"
        ):
            raw_products  = []
            intent_action = "OFF_TOPIC"
            mode          = "NO_RESULTS"
            logger.info(f"[{self.slug}] Semantic conflict detected → mode forced to NO_RESULTS")

        # Resolve effective mode
        if mode is None:
            effective_mode = "NO_RESULTS" if not raw_products else "SHOW_PRODUCTS"
            use_heavy      = intent_action in ["EMOTION", "TROLL", "OFF_TOPIC"] or not raw_products
            logger.info(f"[{self.slug}] mode=None → legacy routing, effective_mode={effective_mode}")
        else:
            effective_mode = mode
            use_heavy      = effective_mode in ("NO_RESULTS", "ASK_CLARIFICATION") or \
                             intent_action in ["EMOTION", "TROLL", "OFF_TOPIC"]

        logger.info(
            f"[{self.slug}] SYNTH START | mode={effective_mode} action={intent_action} "
            f"products={len(raw_products)} use_heavy={use_heavy}"
        )

        client, model = await (
            self.llm_selector.get_heavy()
            if use_heavy
            else self.llm_selector.get_fast()
        )
        log_model_selected(
            self.slug,
            tier="heavy" if use_heavy else "fast",
            model=model, provider="selector",
        )

        catalog_context = self._prepare_products_context(raw_products)
        instruction     = self._build_prompt(
            mode            = effective_mode,
            catalog_context = catalog_context,
            chat_context    = chat_context,
            entities        = entities,
            top_categories  = top_categories,
        )

        log_llm_request(
            self.slug,
            tier="heavy" if use_heavy else "fast",
            model=model, provider="selector",
            prompt_preview=user_query, max_tokens=1000,
        )
        try:
            import time as _time
            _t0 = _time.perf_counter()
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user",   "content": user_query},
                ],
                max_tokens=1000,
                temperature=0.3,
            )
            _dur_ms     = (_time.perf_counter() - _t0) * 1000
            content     = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason or ""
            tokens_used = getattr(response.usage, "total_tokens", 0)

            log_llm_response(
                self.slug, model=model, duration_ms=_dur_ms,
                response_preview=content,
                finish_reason=finish_reason,
                tokens_used=tokens_used,
            )
            logger.info(
                f"[{self.slug}] SYNTH DONE | mode={effective_mode} action={intent_action} "
                f"products={len(raw_products)} model={model} tokens={tokens_used} finish={finish_reason!r}"
            )
            return {
                "text":        content,
                "status":      "SUCCESS" if raw_products else effective_mode,
                "count":       len(raw_products),
                "model":       model,
                "finish_reason": finish_reason,
                "tokens_used": tokens_used,
                "action_type": intent_action,
                "mode":        effective_mode,
            }

        except Exception as e:
            log_llm_error(
                self.slug, model=model, provider="selector",
                error_type=type(e).__name__, error_msg=str(e),
            )
            msg = (
                "Вибачте, сталася технічна помилка при генерації відповіді."
                if self.language.lower() == "ukrainian"
                else "Извините, произошла техническая ошибка."
            )
            return {"text": msg, "status": "CRASH", "count": 0}

    def __repr__(self):
        return f"<Analyzer v7.4.0 slug={self.slug}>"