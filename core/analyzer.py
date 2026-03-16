# /root/ukrsell_v4/core/analyzer.py v8.3.4
import logging
import json
import asyncio
import re
import os
import time as _time
from typing import List, Dict, Any, Optional
from core.llm_selector import LLMSelector
from core.logger import logger, log_intent, log_llm_error, log_model_selected

# Ядро аналитики и логики
from core.intelligence import safe_extract_json, deduplicate_products, entity_filter, semantic_guard
from core.smart_ranking import SmartRanking
from core.recommendation_brain import RecommendationBrain

# ВНИМАНИЕ: ConversationRepair временно отключен из-за отсутствия модуля на диске
# from core.conversation_repair import ConversationRepair

class Analyzer:
    """
    Analyzer v8.3.4 — Industrial AI-Seller Core (Final Hybrid).
    
    Особенности:
    - Контроль контекста (Drift Guard).
    - Маппинг сущностей (Схема <-> JSON).
    - Ранжирование Sales Score (Популярность, Рейтинг, Маржа).
    - Агностические фильтры (Нишевые ограничения).
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.slug = getattr(ctx, 'slug', 'unknown')
        self.language = getattr(ctx, 'language', 'Ukrainian')
        self.currency = getattr(ctx, 'currency', 'грн')
        self.llm_selector = getattr(ctx, 'selector', None)
        self.base_path = getattr(ctx, 'base_path', f"/root/ukrsell_v4/stores/{self.slug}")
        
        self.profile = getattr(ctx, 'profile', {})
        self.store_bio = self.profile.get("description", "Спеціалізований магазин")
        self.store_context_prompt = self.profile.get("store_context_prompt", "")
        
        # Настройки Sales Mode
        self.max_products_list = 3

        # СЛОВАРЬ МАППИНГА (Schema Key -> JSON Key)
        self.entity_map = {
            "Тварина": "animal",
            "Вид виробу": "subtype",
            "Виробник": "brand",
            "Колір": "color",
            "Розмір одягу для тварин": "size"
        }
        
        # Загрузка локальных данных магазина
        self.store_hints = self._load_local_json("intent_hints.json")
        self.store_filters = self._load_local_json("store_filters.json")
        
        # Инициализация подсистем
        self.repair = None # ConversationRepair() отключен
        self.ranking = SmartRanking()
        self.recommendation = RecommendationBrain()

        logger.info(f"🚀 [{self.slug}] Analyzer v8.3.4 (Hybrid Core) Active.")

    def _load_local_json(self, filename: str) -> Dict[str, Any]:
        path = os.path.join(self.base_path, filename)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"[{self.slug}] Failed to load {filename}: {e}")
        return {}

    async def _wait_for_ready(self):
        if hasattr(self.ctx, 'data_ready') and not self.ctx.data_ready.is_set():
            try:
                await asyncio.wait_for(self.ctx.data_ready.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.error(f"[{self.slug}] Assets load timeout.")

    # --- СЕКЦИЯ КОНТЕКСТА И ПАМЯТИ ---

    def _resolve_context(self, current_entities: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        memory = getattr(self.ctx, "memory", {})
        current_entities.setdefault("properties", {})
        
        # 1. Conversation Repair (Безопасный вызов)
        if self.repair:
            current_entities = self.repair.repair(user_query, current_entities, memory)

        last_entities = memory.get("last_entities", {})
        if not last_entities or len(user_query) > 45:
            return current_entities

        # 2. Наследование при коротких уточняющих запросах (Drift Guard)
        last_props = last_entities.get("properties", {})
        curr_props = current_entities.get("properties", {})
        
        if not current_entities.get("category") and last_entities.get("category"):
            current_entities["category"] = last_entities["category"]

        # Наследование кастомных свойств
        for k, v in last_props.items():
            if k not in curr_props and v:
                curr_props[k] = v
        
        # Наследование базовых сущностей из динамической схемы маппинга
        for schema_key in self.entity_map.keys():
            if not current_entities.get(schema_key) and last_entities.get(schema_key):
                current_entities[schema_key] = last_entities[schema_key]

        current_entities["properties"] = curr_props
        return current_entities

    def _update_memory(self, user_query: str, entities: Dict[str, Any], products: List[Dict]):
        if not hasattr(self.ctx, "memory"): self.ctx.memory = {}
        self.ctx.memory["last_query"] = user_query
        self.ctx.memory["last_entities"] = entities
        self.ctx.memory["last_products"] = products[:3]

    def _is_out_of_niche(self, entities: Dict[str, Any]) -> bool:
        if not self.store_filters: return False
        
        for k, allowed in self.store_filters.items():
            if k in ["business_niche", "mode"]: continue
            
            val = entities.get("properties", {}).get(k) or entities.get(k)
            
            if not val:
                for s_key, j_key in self.entity_map.items():
                    if j_key == k: 
                        val = entities.get(s_key)
            
            if val and allowed:
                allowed_list = [str(a).lower() for a in allowed]
                if str(val).lower() not in allowed_list:
                    logger.warning(f"[{self.slug}] Out of niche triggered: {k}={val}")
                    return True
        return False

    def _apply_fast_hints(self, user_query: str, intent: Dict) -> Dict:
        if not self.store_hints: return intent
        q = user_query.lower()
        
        for rule in self.store_hints.get("fuzzy_mappings", []):
            if rule.get("pattern") and re.search(rule["pattern"], q):
                if rule.get("category"):
                    intent["entities"]["category"] = rule["category"]
                    intent["action"] = "SEARCH"
        
        if any(t.lower() in q for t in self.store_hints.get("troll_examples", [])):
            intent["action"] = "TROLL"
            
        return intent

    # --- СЕКЦИЯ INTENT (Извлечение намерений) ---

    async def extract_intent(self, user_query: str, chat_context: str = "") -> Dict:
        await self._wait_for_ready()
        client, model = await self.llm_selector.get_fast()
        
        memory = getattr(self.ctx, "memory", {})
        schema_keys = list(self.entity_map.keys()) 
        
        system_instr = "You are a precise JSON intent parser for e-commerce. Return ONLY JSON."
        prompt = (
            f"PREVIOUS USER QUERY: {memory.get('last_query', 'None')}\n"
            f"ALLOWED ENTITIES: {', '.join(schema_keys)}\n"
            f"CONTEXT:\n{chat_context}\n"
            f"USER QUERY: {user_query}\n\n"
            "STRICT JSON FORMAT:\n"
            "{\n"
            "  \"action\": \"SEARCH\" или \"OFF_TOPIC\" или \"INFO\",\n"
            "  \"entities\": {\n"
            "    \"category\": \"назва категорії\",\n"
            "    \"properties\": { \"додатковий_ключ\": \"значення\" },\n"
            "    \"...люба сутність з ALLOWED ENTITIES...\": \"значення\"\n"
            "  }\n"
            "}"
        )

        try:
            t0 = _time.perf_counter()
            response = await client.chat.completions.create(
                model=model, messages=[{"role": "system", "content": system_instr}, {"role": "user", "content": prompt}],
                temperature=0.0
            )
            intent = safe_extract_json(response.choices[0].message.content)
            
            # Применяем локальные правила и наследование
            intent = self._apply_fast_hints(user_query, intent)
            intent["entities"] = self._resolve_context(intent.get("entities", {}), user_query)
            
            # Валидация ниши
            if intent.get("action") not in ["OFF_TOPIC", "TROLL"] and self._is_out_of_niche(intent["entities"]):
                intent["action"] = "OFF_TOPIC"
            
            log_intent(self.slug, "N/A", model, intent["action"], intent["entities"], (_time.perf_counter()-t0)*1000)
            return intent
        except Exception as e:
            log_llm_error(self.slug, model, "intent_parser", type(e).__name__, str(e))
            return {"action": "SEARCH", "entities": {}}

    # --- СЕКЦИЯ СИНТЕЗА (Генерация ответа) ---

    def _calculate_sales_score(self, product: Dict) -> float:
        """Ранжирование товара: Популярность 40%, Рейтинг 30%, Маржа 30%."""
        d = product.get("product") or product.get("data") or product
        try:
            score = (float(d.get("popularity", 0)) * 0.4) + \
                    (float(d.get("rating", 0)) * 0.3) + \
                    (float(d.get("margin", 0)) * 0.3)
            return round(score, 4)
        except (ValueError, TypeError):
            return 0.0

    def _prepare_summary(self, products: List[Dict]) -> str:
        """Сборка контекста товаров для LLM с использованием маппинга."""
        if not products: return "Товари не знайдені."
        items = []
        for p in products:
            d = p.get("product") or p.get("data") or p
            
            extra = []
            for schema_key, json_key in self.entity_map.items():
                val = d.get(json_key)
                if val:
                    val_str = ", ".join(val) if isinstance(val, list) else str(val)
                    extra.append(f"{schema_key}: {val_str}")
            
            info = f"ID: {d.get('id')} | {d.get('title')} | Ціна: {d.get('price')} {self.currency}"
            if extra:
                info += f" | Характеристики: {' ; '.join(extra)}"
            
            items.append(info)
        return "\n".join(items)

    def _build_prompt(self, mode: str, catalog_context: str, chat_context: str, is_advice: bool) -> str:
        lang = "УКРАЇНСЬКОЮ МОВОЮ" if self.language.lower() == "ukrainian" else "НА РУССКОМ ЯЗЫКЕ"
        
        base_style = (
            f"Ти — привітний експерт-консультант магазину '{self.store_bio}'.\n"
            f"Твоя мова: {lang}. {self.store_context_prompt}\n"
            "- Пиши живо, коротко, без канцелярщини.\n"
            "- Використовуй емодзі для дружньої атмосфери.\n"
            "- ПРАВИЛО: Завжди закінчуй відповідь коротким питанням, щоб продовжити діалог.\n"
        )

        if mode == "OFF_TOPIC":
            return f"{base_style}\nСИТУАЦІЯ: Користувач запитав про щось поза нашою нішею. Ввічливо відмов та запропонуй допомогу з товарами нашого профілю."

        if is_advice:
            return (
                f"{base_style}\nСИТУАЦІЯ: Клієнт просить поради. Ти вибрав найкращий варіант (CHAMPION).\n"
                f"ЗАВДАННЯ: Переконай клієнта купити саме цей товар, професійно аргументуючи його переваги.\n\n"
                f"ТОВАР ДЛЯ ПРЕЗЕНТАЦІЇ:\n{catalog_context}\n\nКОНТЕКСТ ДІАЛОГУ:\n{chat_context}"
            )

        return (
            f"{base_style}\nСИТУАЦІЯ: Знайдено товари за запитом.\n"
            f"ЗАВДАННЯ: Коротко презентуй ці позиції (макс {self.max_products_list}), підкреслюючи вигоду.\n\n"
            f"СПИСОК ТОВАРІВ:\n{catalog_context}\n\nКОНТЕКСТ ДІАЛОГУ:\n{chat_context}"
        )

    async def synthesize_response(self, search_results: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """STAGE 2: Final Synthesis."""
        t_start = _time.perf_counter()
        intent = kwargs.get("intent") or {"action": "SEARCH", "entities": {}}
        user_query = kwargs.get("user_query", "")
        chat_context = kwargs.get("chat_context", "")
        
        products = search_results.get("products", [])
        
        # 1. Сортировка по Sales Score
        for p in products:
            p["sales_score"] = self._calculate_sales_score(p)
        products.sort(key=lambda x: x.get("sales_score", 0), reverse=True)
        
        # 2. Определение режима работы
        is_advice = self.recommendation.detect_advice(user_query)
        
        if is_advice and products:
            final_selection = self.recommendation.pick_best(products)
            mode = "SHOW_PRODUCTS"
        elif products:
            final_selection = products[:self.max_products_list]
            mode = "SHOW_PRODUCTS"
        else:
            final_selection = []
            mode = "NO_RESULTS"

        if intent.get("action") == "OFF_TOPIC": mode = "OFF_TOPIC"

        # 3. Обновление памяти
        if mode == "SHOW_PRODUCTS":
            self._update_memory(user_query, intent.get("entities", {}), final_selection)

        # 4. Генерация финального текста
        client, model = await self.llm_selector.get_fast()
        catalog_summary = self._prepare_summary(final_selection)
        prompt = self._build_prompt(mode, catalog_summary, chat_context, is_advice)

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_query}],
                temperature=0.4,
                max_tokens=800
            )
            
            return {
                "text": response.choices[0].message.content.strip(),
                "status": "SUCCESS",
                "model": model,
                "ms": round((_time.perf_counter() - t_start) * 1000, 1),
                "intent_applied": intent.get("action")
            }
        except Exception as e:
            log_llm_error(self.slug, model, "synthesizer", type(e).__name__, str(e))
            return {"text": "Вибачте, виникла технічна заминка. Будь ласка, спробуйте ще раз.", "status": "ERROR"}

    def __repr__(self):
        return f"<Analyzer v8.3.4 slug={self.slug} mode=Hybrid_Industrial_Final>"