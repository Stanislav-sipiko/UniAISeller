# /root/ukrsell_v4/core/analyzer.py v7.8.6
import logging
import json
import asyncio
import re
import time as _time
from typing import List, Dict, Any, Union, Optional
from core.llm_selector import LLMSelector
from core.logger import (
    logger, 
    log_event, 
    log_llm_request, 
    log_llm_response, 
    log_llm_error, 
    log_model_selected,
    log_intent
)
from core.intelligence import (
    safe_extract_json, 
    semantic_guard, 
    deduplicate_products, 
    entity_filter
)


class Analyzer:
    """
    Analyzer v7.8.6 — Taxonomy-Aware Semantic Intent Parser & Generative Synthesizer.
    
    Changelog v7.8.6:
        - FIX: Added 'history' and 'query' to synthesize_response signature to prevent TypeError.
        - SYNC: Compatible with Kernel v7.8.4 calls passing unexpected keyword arguments.
        - STABILITY: Full source restoration for CI/CD compatibility.
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
        
        if hasattr(ctx, 'get_store_bio'):
            self.store_bio = ctx.get_store_bio()
        else:
            self.store_bio = self.profile.get("description", "Спеціалізований інтернет-магазин.")

        self.store_context_prompt = self.profile.get("store_context_prompt", "")
        
        self.intent_mapping = self.profile.get("intent_mapping", {})
        self.category_map   = self.profile.get("category_map", {})
        
        self.allowed_brands = self.profile.get("allowed_brands", [])

        logger.info(f"🛠️ [{self.slug}] Analyzer v7.8.6 (Full Source) initialized.")

    async def _wait_for_ready(self):
        """Ожидание готовности ассетов (FAISS и метаданных)."""
        if hasattr(self.ctx, 'data_ready'):
            if not self.ctx.data_ready.is_set():
                logger.info(f"[{self.slug}] Analyzer: Waiting for assets...")
                try:
                    await asyncio.wait_for(self.ctx.data_ready.wait(), timeout=15.0)
                except asyncio.TimeoutError:
                    logger.error(f"[{self.slug}] Analyzer: Assets load timeout.")

    def _get_current_schema(self) -> List[str]:
        """Получение списка ключей атрибутов из текущей БД товаров для подсказки LLM."""
        try:
            if hasattr(self.ctx, 'schema_keys') and self.ctx.schema_keys:
                return self.ctx.schema_keys
            if hasattr(self.ctx, 'retrieval') and hasattr(self.ctx.retrieval, 'schema_keys'):
                return self.ctx.retrieval.schema_keys
        except Exception as e:
            logger.debug(f"[{self.slug}] Schema detection skip: {e}")
        return ["color", "size", "material", "animal"]

    def _get_taxonomy_hint(self) -> str:
        """Формирует строку-подсказку из доступных категорий и брендов для промпта."""
        hint_parts = []
        if self.category_map:
            cats = list(self.category_map.keys())[:40] 
            hint_parts.append(f"ALLOWED CATEGORIES: {', '.join(cats)}")
        
        if self.allowed_brands:
            brands = self.allowed_brands[:30]
            hint_parts.append(f"ALLOWED BRANDS: {', '.join(brands)}")
        
        return "\n".join(hint_parts) if hint_parts else "No specific taxonomy constraints."

    async def extract_intent(self, user_query: str, chat_context: str = "") -> Dict[str, Any]:
        """
        STAGE 1: Entity extraction с учетом таксономии магазина.
        """
        await self._wait_for_ready()

        if not self.llm_selector:
            return {"entities": {"category": None, "brand": None, "price_limit": None}, "action": "SEARCH"}

        client, model = await self.llm_selector.get_fast()
        log_model_selected(self.slug, tier="fast", model=model, provider="selector")
        
        schema_keys = self._get_current_schema()
        taxonomy_hint = self._get_taxonomy_hint()

        schema_instruction = (
            f"Detect these dynamic attributes if present: {', '.join(schema_keys)}"
            if schema_keys else "Detect standard attributes: category, brand, price_limit."
        )
        
        context_block = f"\nRECENT CONVERSATION (Context):\n{chat_context}\n" if chat_context else ""

        system_instr = (
            "You are a precise JSON-only e-commerce intent parser.\n"
            "Your task: Extract search parameters and user intent from the query.\n"
            "Never use markdown formatting. Output ONLY raw JSON."
        )

        prompt = (
            f"STORE TAXONOMY:\n{taxonomy_hint}\n\n"
            "STRICT EXTRACTION RULES:\n"
            "1. CATEGORY: Must match 'ALLOWED CATEGORIES' exactly. If it is a synonym, map it to the allowed name. If no match, return null.\n"
            "2. BRAND: Extract ONLY if it is a real brand name. Use 'ALLOWED BRANDS' as primary reference.\n"
            "3. ACTION: 'SEARCH' for shopping, 'INFO' for questions about store/delivery, 'TROLL' for insults, 'CHAT' for greetings.\n"
            "4. PRICE: Extract only numbers into 'price_limit'.\n\n"
            f"{schema_instruction}\n"
            f"{context_block}"
            f"USER QUERY: {user_query}\n\n"
            "RETURN JSON STRUCTURE:\n"
            "{\n"
            '  "action": "SEARCH | INFO | TROLL | EMOTION | CHAT | OFF_TOPIC",\n'
            '  "entities": {\n'
            '    "category": "string or null",\n'
            '    "brand": "string or null",\n'
            '    "price_limit": number or null,\n'
            '    "properties": { "key": "value" }\n'
            '  }\n'
            "}"
        )

        log_llm_request(
            slug=self.slug, 
            tier="fast", 
            model=model, 
            provider="selector", 
            prompt_preview=f"Intent Extraction for: {user_query}",
            max_tokens=500
        )
        
        try:
            t0 = _time.perf_counter()
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_instr},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=500,
                temperature=0.0,
            )
            dur_ms = (_time.perf_counter() - t0) * 1000
            content = response.choices[0].message.content
            tokens = getattr(response.usage, "total_tokens", 0)

            log_llm_response(
                slug=self.slug, 
                model=model, 
                duration_ms=dur_ms, 
                response_preview=content, 
                finish_reason=getattr(response.choices[0], "finish_reason", "stop"),
                tokens_used=tokens
            )
            
            intent = safe_extract_json(content)

            entities = intent.get("entities", {})
            
            raw_cat = entities.get("category")
            if raw_cat and self.category_map:
                cat_lower = str(raw_cat).lower().strip()
                if cat_lower not in [k.lower() for k in self.category_map.keys()]:
                    found = False
                    for valid_cat in self.category_map.keys():
                        if cat_lower == valid_cat.lower() or cat_lower in valid_cat.lower():
                            entities["category"] = valid_cat
                            found = True
                            break
                    if not found:
                        logger.warning(f"[{self.slug}] Taxonomy Enforcement: '{raw_cat}' discarded.")
                        entities["category"] = None
            
            intent["entities"] = entities

            log_intent(
                slug=self.slug, 
                chat_id="N/A", 
                model=model, 
                action=intent.get("action", "UNKNOWN"), 
                entities=entities, 
                duration_ms=dur_ms
            )

            return intent

        except Exception as e:
            log_llm_error(
                slug=self.slug, 
                model=model, 
                provider="selector", 
                error_type=type(e).__name__, 
                error_msg=str(e)
            )
            return {"entities": {"category": None, "brand": None}, "action": "SEARCH"}

    def _prepare_products_context(self, products: List[Dict[str, Any]]) -> str:
        """Подготовка данных о товарах для инъекции в промпт синтезатора."""
        if not products:
            return "Товари не знайдені в каталозі." if self.language.lower() == "ukrainian" else "Товары не найдены."

        clean_products = deduplicate_products(products, top_k=6)
        context_lines  = []

        for idx, res in enumerate(clean_products, 1):
            p = res.get("data") or res.get("product") or res
            
            p_id  = p.get("id") or p.get("sku") or p.get("product_id", "N/A")
            name  = p.get("title") or p.get("name") or "Товар"
            brand = p.get("brand", "Unknown")
            price = p.get("price", "---")
            link  = p.get("product_url") or p.get("link") or "#"
            desc  = p.get("description") or p.get("short_description", "")
            
            attrs = p.get("attributes", {})
            attr_str = ""
            if isinstance(attrs, dict):
                attr_str = ", ".join([f"{k}: {v}" for k, v in attrs.items() if v])
            elif isinstance(attrs, list):
                attr_str = ", ".join([str(x) for x in attrs])

            context_lines.append(
                f"ITEM #{idx} [ID: {p_id}]\n"
                f"TITLE: {name}\n"
                f"BRAND: {brand} | PRICE: {price} {self.currency}\n"
                f"SPECS: {attr_str}\n"
                f"DESC: {str(desc)[:200]}...\n"
                f"URL: {link}\n"
                f"---"
            )

        return "\n".join(context_lines)

    def _check_semantic_conflict(self, category: str) -> bool:
        """Проверка, входит ли категория в сферу компетенции магазина."""
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
        """Динамическая сборка системной инструкции для генерации финального ответа."""
        lang_name = "УКРАЇНСЬКОЮ" if self.language.lower() == "ukrainian" else "РУССКОМ"
        
        personality = (
            f"\nВАШ ХАРАКТЕР ТА ПРАВИЛА:\n{self.store_context_prompt}\n"
            if self.store_context_prompt else ""
        )

        if mode == "SHOW_PRODUCTS":
            return (
                f"Ви — професійний асистент магазину: {self.store_bio}.\n"
                f"{personality}"
                f"МОВА ВІДПОВІДІ: {lang_name}.\n\n"
                "ПРАВИЛА ГЕНЕРАЦІЇ:\n"
                "1. Використовуйте ТІЛЬКИ товары з блоку CATALOG. Не вигадуйте неіснуючі товары.\n"
                "2. Для кожного товару обов'язково вказуйте назву, ціну та посилання.\n"
                "3. Обов'язково додавайте [ID: ...] для каждого товару.\n"
                "4. Форматування: HTML (<b>назва</b>, <a href='url'>текст</a>).\n\n"
                f"CATALOG:\n{catalog_context}\n\n"
                f"CONVERSATION HISTORY:\n{chat_context}"
            )

        if mode == "ASK_CLARIFICATION":
            cat_hints = ""
            if top_categories:
                cat_hints = "\nСхожі категорії у нас: " + ", ".join([str(c[0]) for c in top_categories[:3]])

            return (
                f"Ви — консультант магазину: {self.store_bio}.\n"
                f"{personality}{cat_hints}\n"
                f"МОВА ВІДПОВІДІ: {lang_name}.\n\n"
                "СИТУАЦІЯ: Запит користувача занадто широкий або неоднозначний.\n"
                "ЗАВДАННЯ:\n"
                "1. Коротко покажіть 1-2 приклади з CATALOG, якщо они хоч трохи підходять.\n"
                "2. Поставте уточнююче питання.\n"
                f"\nCATALOG:\n{catalog_context}\n\n"
                f"CONTEXT:\n{chat_context}"
            )

        expertise = self.profile.get("expertise_fields", [])
        exp_str = ", ".join(expertise) if expertise else "товари для дому"
        
        return (
            f"Ви — представник магазину: {self.store_bio}.\n"
            f"{personality}"
            f"МОВА ВІДПОВІДІ: {lang_name}.\n\n"
            "СИТУАЦІЯ: Ми не знайшли товарів.\n"
            "ЗАВДАННЯ:\n"
            "1. Ввічливо повідомте, що таких товарів зараз немає.\n"
            f"2. Нагадайте, що ваша спеціалізація: {exp_str}.\n"
            "3. Запропонуйте пошукати щось інше.\n"
            f"\nCONTEXT:\n{chat_context}"
        )

    async def synthesize_response(
        self,
        search_results: Dict[str, Any],
        user_query: str = None,
        entities: Optional[Dict[str, Any]] = None,
        intent: Optional[Dict[str, Any]] = None,
        chat_context: str = "",
        mode: Optional[str] = None,
        top_categories: Optional[List] = None,
        products: List[Dict[str, Any]] = None,
        query: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        STAGE 2: Генерация финального текста.
        v7.8.6: Added 'query' and '**kwargs' (to capture 'history') for Kernel compatibility.
        """
        if not self.llm_selector:
            return {"text": "Помилка зв'язку з сервером.", "status": "ERROR", "count": 0}

        # Resolve the query: use 'user_query' (old) or 'query' (new Kernel)
        final_query = user_query or query or ""

        working_intent = intent or entities or {}
        working_entities = working_intent.get("entities", working_intent)
        intent_action = working_intent.get("action", "SEARCH")

        r_data = search_results or {"status": "NO_RESULTS", "products": []}
        all_vector_hits = products if products is not None else r_data.get("products", [])
        
        filtered_products = entity_filter(
            all_vector_hits, 
            working_entities, 
            intent_mapping=self.intent_mapping,
            category_map=self.category_map
        )
        
        if mode is None:
            if not filtered_products:
                mode = "NO_RESULTS"
            elif len(filtered_products) < 2 and len(all_vector_hits) > 10:
                mode = "ASK_CLARIFICATION"
            else:
                mode = "SHOW_PRODUCTS"

        cat_to_check = working_entities.get("category")
        if self._check_semantic_conflict(cat_to_check):
            filtered_products = []
            mode = "NO_RESULTS"
            intent_action = "OFF_TOPIC"

        use_heavy = mode in ("NO_RESULTS", "ASK_CLARIFICATION") or intent_action != "SEARCH"
        client, model = await (self.llm_selector.get_heavy() if use_heavy else self.llm_selector.get_fast())
        log_model_selected(self.slug, tier="heavy" if use_heavy else "fast", model=model, provider="selector")

        catalog_context = self._prepare_products_context(filtered_products)
        instruction = self._build_prompt(mode, catalog_context, chat_context, working_entities, top_categories)

        log_llm_request(
            slug=self.slug, 
            tier="heavy" if use_heavy else "fast", 
            model=model, 
            provider="selector",
            prompt_preview=f"Synthesizing (Mode: {mode})",
            max_tokens=1000
        )
        
        try:
            t0 = _time.perf_counter()
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user",   "content": final_query},
                ],
                max_tokens=1000,
                temperature=0.3,
            )
            dur_ms = (_time.perf_counter() - t0) * 1000
            content = response.choices[0].message.content
            tokens = getattr(response.usage, "total_tokens", 0)

            log_llm_response(
                slug=self.slug, 
                model=model, 
                duration_ms=dur_ms, 
                response_preview=content, 
                finish_reason=getattr(response.choices[0], "finish_reason", "stop"),
                tokens_used=tokens
            )
            
            return {
                "text":        content,
                "status":      "SUCCESS" if filtered_products else mode,
                "count":       len(filtered_products),
                "model":       model,
                "action_type": intent_action,
                "mode":        mode,
                "tokens_used": tokens
            }

        except Exception as e:
            log_llm_error(
                slug=self.slug, 
                model=model, 
                provider="selector", 
                error_type=type(e).__name__, 
                error_msg=str(e)
            )
            msg = "Вибачте, сталася помилка." if self.language.lower() == "ukrainian" else "Извините, произошла ошибка."
            return {"text": msg, "status": "CRASH", "count": 0}

    def __repr__(self):
        return f"<Analyzer v7.8.6 slug={self.slug} lang={self.language}>"