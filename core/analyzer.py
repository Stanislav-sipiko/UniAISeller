# /root/ukrsell_v4/core/analyzer.py v8.4.2
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

class Analyzer:
    """
    Analyzer v8.4.2 — Industrial AI-Seller Core.
    
    Changelog:
    - v8.4.2: FIXED logic for TROLL/ABSURD detection. Added Sanity Check to prevent 
             pointless retrieval for impossible requests (e.g., 50kg cats, alcoholic dogs).
    - v8.3.4: Added reasoning clean-up and niche validation fixes.
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
        self.repair = None 
        self.ranking = SmartRanking()
        self.recommendation = RecommendationBrain()

        logger.info(f"🚀 [{self.slug}] Analyzer v8.4.2 (Sanity Core) Active.")

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

    def _clean_reasoning(self, text: str) -> str:
        """Удаляет блоки <think>...</think> из ответа модели."""
        if not text:
            return ""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _resolve_context(self, current_entities: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        memory = getattr(self.ctx, "memory", {})
        current_entities.setdefault("properties", {})
        
        if self.repair:
            current_entities = self.repair.repair(user_query, current_entities, memory)

        last_entities = memory.get("last_entities", {})
        if not last_entities or len(user_query) > 45:
            return current_entities

        last_props = last_entities.get("properties", {})
        curr_props = current_entities.get("properties", {})
        
        if not current_entities.get("category") and last_entities.get("category"):
            current_entities["category"] = last_entities["category"]

        for k, v in last_props.items():
            if k not in curr_props and v:
                curr_props[k] = v
        
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
        
        category = entities.get("category")
        if category and str(category).lower() in ["одяг", "комбінезон", "куртка", "жилет"]:
            return False

        for k, allowed in self.store_filters.items():
            if k in ["business_niche", "mode"]: continue
            
            val = entities.get("properties", {}).get(k) or entities.get(k)
            
            if not val:
                for s_key, j_key in self.entity_map.items():
                    if j_key == k: 
                        val = entities.get(s_key)
            
            if val and allowed:
                allowed_list = [str(a).lower() for a in allowed]
                if k in ["animal", "Тварина"] and category:
                    continue
                    
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

    async def extract_intent(self, user_query: str, chat_context: str = "") -> Dict:
        """
        STAGE 1: Intent Extraction with Sanity Check.
        Determines if the request is logical before enabling search.
        """
        await self._wait_for_ready()
        client, model = await self.llm_selector.get_fast()
        
        memory = getattr(self.ctx, "memory", {})
        schema_keys = list(self.entity_map.keys()) 
        
        system_instr = (
            "You are a precise JSON intent parser for e-commerce. "
            "Your main task is to distinguish between a real buyer and a prankster/troll. "
            "Return ONLY JSON. Do not include reasoning tags."
        )
        
        prompt = (
            f"STRICT RULES:\n"
            f"1. If the query is physically impossible (e.g., a 50kg cat, a dog drinking alcohol, a golden phone for 1 dollar), "
            f"set action to 'TROLL'.\n"
            f"2. If the query is a joke, provocation, or absolute nonsense, set action to 'TROLL'.\n"
            f"3. If the user is asking about things not related to our store niche, set action to 'OFF_TOPIC'.\n"
            f"4. Otherwise, set action to 'SEARCH' and extract entities.\n\n"
            f"STORE DESCRIPTION: {self.store_bio}\n"
            f"ALLOWED ENTITIES: {', '.join(schema_keys)}\n"
            f"CONTEXT:\n{chat_context}\n"
            f"USER QUERY: {user_query}\n\n"
            "STRICT JSON FORMAT:\n"
            "{\n"
            "  \"action\": \"SEARCH\" | \"OFF_TOPIC\" | \"TROLL\" | \"INFO\",\n"
            "  \"reason\": \"short explanation of why this is a troll or search\",\n"
            "  \"entities\": {\n"
            "    \"category\": \"category name\",\n"
            "    \"properties\": { \"weight\": \"50kg\", \"material\": \"silk\" },\n"
            "    \"Тварина\": \"animal type\"\n"
            "  }\n"
            "}"
        )

        try:
            t0 = _time.perf_counter()
            response = await client.chat.completions.create(
                model=model, messages=[{"role": "system", "content": system_instr}, {"role": "user", "content": prompt}],
                temperature=0.0
            )
            
            raw_content = self._clean_reasoning(response.choices[0].message.content)
            intent = safe_extract_json(raw_content)
            
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

    def _calculate_sales_score(self, product: Dict) -> float:
        d = product.get("product") or product.get("data") or product
        try:
            score = (float(d.get("popularity", 0)) * 0.4) + \
                    (float(d.get("rating", 0)) * 0.3) + \
                    (float(d.get("margin", 0)) * 0.3)
            return round(score, 4)
        except (ValueError, TypeError):
            return 0.0

    def _prepare_summary(self, products: List[Dict]) -> str:
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
            "- Використовуй емодії для дружньої атмосфери.\n"
            "- ПРАВИЛО: Завжди закінчуй відповідь коротким питанням, щоб продовжити діалог.\n"
            "- СУВОРА ЗАБОРОНА: Ніколи не вигадуй назви товарів, брендів або цін, якщо їх немає в наданому списку.\n"
        )

        if mode == "TROLL":
            return (
                f"{base_style}\nСИТУАЦІЯ: Користувач жартує, тролить або ставить абсурдне запитання (наприклад, про котів вагою 50кг).\n"
                f"ЗАВДАННЯ: Дай дотепну, іронічну, але ввічливу відповідь. "
                f"Покажи, що ти розумієш жарт, але м'яко поверни розмову до реальних товарів нашого магазину."
            )

        if mode == "OFF_TOPIC":
            return f"{base_style}\nСИТУАЦІЯ: Користувач запитав про щось поза нашою нішею. Ввічливо відмов та запропонуй допомогу з товарами нашого профілю."

        if mode == "NO_RESULTS" or "Товари не знайдені" in catalog_context:
            return f"{base_style}\nСИТУАЦІЯ: За запитом клієнта нічого не знайдено.\nЗАВДАННЯ: Чесно скажи, що таких товарів зараз немає, та запропонуй змінити параметри пошуку."

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
        
        if intent.get("action") == "TROLL":
            mode = "TROLL"
            final_selection = []
        elif intent.get("action") == "OFF_TOPIC":
            mode = "OFF_TOPIC"
            final_selection = []
        elif is_advice and products:
            final_selection = self.recommendation.pick_best(products)
            mode = "SHOW_PRODUCTS"
        elif products:
            final_selection = products[:self.max_products_list]
            mode = "SHOW_PRODUCTS"
        else:
            final_selection = []
            mode = "NO_RESULTS"

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
            
            clean_text = self._clean_reasoning(response.choices[0].message.content)
            
            return {
                "text": clean_text,
                "status": "SUCCESS",
                "model": model,
                "ms": round((_time.perf_counter() - t_start) * 1000, 1),
                "intent_applied": intent.get("action")
            }
        except Exception as e:
            log_llm_error(self.slug, model, "synthesizer", type(e).__name__, str(e))
            return {"text": "Вибачте, виникла технічна заминка. Будь ласка, спробуйте ще раз.", "status": "ERROR"}

    def __repr__(self):
        return f"<Analyzer v8.4.2 slug={self.slug} mode=Sanity_Guard_Industrial>"