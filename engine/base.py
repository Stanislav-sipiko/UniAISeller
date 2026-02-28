import logging
import aiohttp
import json
import os
from typing import Any, Dict, Optional
from core.store_context import StoreContext
from core.retrieval import RetrievalEngine
from core.dialog_manager import DialogManager
from core.llm_selector import LLMSelector

logger = logging.getLogger("UkrSell_StoreEngine")

class StoreEngine:
    """
    Универсальный оркестратор бизнес-логики магазина.
    Поддерживает мультивалютность и динамическую локализацию через конфиги.
    """
    def __init__(self, ctx: StoreContext):
        self.ctx = ctx
        self.slug = ctx.slug
        self.token = ctx.config.get("bot_token")
        
        # Динамические параметры из конфига магазина
        self.currency = ctx.config.get("currency", "грн") # По умолчанию грн для Украины
        self.api_url = f"https://api.telegram.org/bot{self.token}"
        
        self.llm_selector = LLMSelector()
        self.retrieval = RetrievalEngine(ctx)
        self.dialog_manager = DialogManager(ctx, self.llm_selector)
        
        self._session: Optional[aiohttp.ClientSession] = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info(f"Engine [{self.slug}] session closed.")

    async def handle_update(self, update: Dict[str, Any]):
        message = update.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        text = message.get("text")
        
        if not chat_id or not text:
            return

        logger.info(f"[{self.slug}] Processing message: {text[:20]}...")

        # --- Шаг 0: Анализ интента и защита (Troll Buffer) ---
        decision = await self.dialog_manager.analyze_intent(text)
        
        if decision.get("action") == "TROLL":
            # Берем шутливый ответ от LLM или стандартную заглушку
            troll_msg = decision.get("response_text", "🧐")
            await self.send_message(chat_id, troll_msg)
            return

        # --- Шаг 1: Семантический поиск по базе ---
        search_result = await self.retrieval.search(text)

        if search_result["status"] == "ABSENT_CATEGORY":
            options = ", ".join(search_result["suggested_categories"])
            # Берем шаблон ответа из промптов магазина
            tpl = self.ctx.prompts.get("wrong_category", "Этого раздела нет. Доступны: {options}")
            await self.send_message(chat_id, tpl.format(options=options))

        elif search_result["status"] == "SUCCESS":
            await self.send_products(chat_id, search_result["products"])

        else:
            fail_msg = self.ctx.prompts.get("not_found", "Ничего не найдено.")
            await self.send_message(chat_id, fail_msg)

    async def send_message(self, chat_id: int, text: str, parse_mode: str = "HTML"):
        url = f"{self.api_url}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
        session = self._get_session()
        try:
            async with session.post(url, json=payload) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"[{self.slug}] Error: {e}")

    async def send_products(self, chat_id: int, products: list):
        """Форматирует список товаров, используя валюту и язык магазина."""
        lines = []
        
        # Заголовки из промптов магазина (локализация)
        header = self.ctx.prompts.get("search_header", "Результаты поиска:")
        view_label = self.ctx.prompts.get("view_button", "Смотреть")
        price_label = self.ctx.prompts.get("price_label", "Цена")

        for idx, p in enumerate(products[:5], 1):
            product_data = p["product"]
            name = product_data.get("name", "---")
            price = product_data.get("price", "???")
            link = product_data.get("link") or product_data.get("url", "")
            
            # Универсальная строка товара
            item_str = f"{idx}. <b>{name}</b>\n{price_label}: {price} {self.currency}"
            if link:
                item_str += f"\n<a href='{link}'>{view_label}</a>"
            
            lines.append(item_str)
        
        full_text = f"<b>{header}</b>\n\n" + "\n\n".join(lines)
        await self.send_message(chat_id, full_text)