# /root/ukrsell_v4/engine/base.py v5.1.6
import aiohttp
import json
import os
import sqlite3
import datetime
from typing import Any, Dict, Optional

# Unified platform logging
from core.logger import logger, log_event
from core.store_context import StoreContext
from core.retrieval import RetrievalEngine
from core.dialog_manager import DialogManager
from core.analyzer import Analyzer

class StoreEngine:
    """
    Universal store business logic orchestrator v5.1.6.
    Fixed: Full restoration of send_products (fallback mechanism).
    Fixed: Support for both 'price_limit' and 'max_price' from LLM.
    Added: AI Persona priority (Profile > Context > Default).
    Implemented: Zero Omission (Full script).
    Fixed: parse_mode changed to HTML for unified recommendation support.
    """
    def __init__(self, ctx: StoreContext):
        self.ctx = ctx
        self.slug = ctx.slug
        
        # Protection against missing config in StoreContext
        self.config = getattr(ctx, 'config', {})
        self.token = self.config.get("bot_token", "NO_TOKEN")
        
        # Dynamic parameters from store config
        self.currency = self.config.get("currency", "грн")
        self.api_url = f"https://api.telegram.org/bot{self.token}"
        
        # Dependencies - EXTRACT FROM CONTEXT
        self.llm_selector = getattr(ctx, 'selector', None)
        self.kernel = getattr(ctx, 'kernel', None) 
        
        if not self.llm_selector:
            logger.error(f"[{self.slug}] CRITICAL: LLMSelector not found in context!")
            
        # Компоненты берём из ctx — они созданы в kernel.py/engine_factory().
        # StoreEngine НЕ создаёт своих копий, чтобы не было дублей в памяти.
        self.retrieval      = getattr(ctx, 'retrieval', None)
        self.dialog_manager = getattr(ctx, 'dialog_manager', None)
        self.analyzer       = getattr(ctx, 'analyzer', None)

        # Защита: если kernel не заполнил ctx (прямой инстанс вне платформы) — создаём сами
        if self.retrieval is None:
            logger.warning(f"[{self.slug}] retrieval not in ctx — creating fallback instance.")
            self.retrieval = RetrievalEngine(ctx, None, None)
        if self.dialog_manager is None:
            logger.warning(f"[{self.slug}] dialog_manager not in ctx — creating fallback instance.")
            self.dialog_manager = DialogManager(ctx, self.llm_selector)
        if self.analyzer is None:
            logger.warning(f"[{self.slug}] analyzer not in ctx — creating fallback instance.")
            self.analyzer = Analyzer(ctx)
        
        # Database paths
        self.db_path = os.path.join(ctx.base_path, "users.db")
        self.session_db_path = os.path.join(ctx.base_path, "sessions.db")
        self.patch_path = os.path.join(ctx.base_path, "fsm_soft_patch.json")
        self.profile_path = os.path.join(ctx.base_path, "store_profile.json")
        
        # Session and DB initialization
        self._session: Optional[aiohttp.ClientSession] = None


    def _get_ai_welcome(self) -> Optional[str]:
        """Retrieves AI persona message from store profile with fallback."""
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("profile", {}).get("ai_welcome_message")
            except Exception as e:
                logger.error(f"[{self.slug}] Error reading profile: {e}")
        return None

    def record_fsm_error(self, user_query: str, reason: str):
        """Records critical logic errors for Negative Examples training."""
        try:
            data = {"troll_patterns": [], "fsm_errors": []}
            if os.path.exists(self.patch_path):
                with open(self.patch_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        pass

            data.setdefault("fsm_errors", []).append({
                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "query": user_query,
                "reason": reason,
                "type": "FSM_ERROR"
            })

            with open(self.patch_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            log_event("INSPECTOR_LOG", {"slug": self.slug, "event": "FSM_ERROR_RECORDED"})
        except Exception as e:
            logger.error(f"[{self.slug}] Failed to record FSM error: {e}")

    def _log_user_activity(self, user_id: str, username: str, query: str, res_type: str, cat: str):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO users_stats (user_id, username, query, response_type, category, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, username, query, res_type, cat, datetime.datetime.now())
                )
        except Exception as e:
            logger.error(f"[{self.slug}] Activity logging error: {e}")

    def _log_chat_history(self, chat_id: str, message_id: str, role: str, content: str):
        try:
            with sqlite3.connect(self.session_db_path) as conn:
                conn.execute(
                    "INSERT INTO chat_history (chat_id, message_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (chat_id, message_id, role, content, datetime.datetime.now())
                )
        except Exception as e:
            logger.error(f"[{self.slug}] History logging error: {e}")

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def handle_update(self, update: Dict[str, Any]):
        """Main processing loop v5.1.6."""
        message = update.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        user_info = message.get("from", {})
        user_id = user_info.get("id")
        username = user_info.get("username", "anonymous")
        text = message.get("text", "").strip()
        msg_id = message.get("message_id")
        
        if not chat_id or not text:
            return

        log_event("MESSAGE_RECEIVED", {"slug": self.slug, "chat_id": chat_id, "text": text[:50]})
        self._log_chat_history(str(chat_id), str(msg_id), "user", text)

        # --- Step 0: System Commands ---
        if text.startswith("/"):
            if text == "/start":
                ai_welcome = self._get_ai_welcome()
                welcome = ai_welcome or getattr(self.ctx, 'prompts', {}).get("welcome", "Привіт! Чим я можу вам допомогти?")
                await self.send_message(chat_id, welcome, parse_mode="HTML")
                self._log_user_activity(str(user_id), username, text, "START", "none")
                return

        # --- Step 1: Intent Analysis ---
        decision = {"action": "SEARCH", "entities": {}, "response_text": None}
        try:
            decision = await self.dialog_manager.analyze_intent(text, str(chat_id))
            action = decision.get("action", "SEARCH")

            if action == "TROLL":
                await self.send_message(chat_id, decision.get("response_text", "🧐"), parse_mode="HTML")
                self._log_user_activity(str(user_id), username, text, "TROLL", "none")
                return
            
            if action == "CHAT":
                await self.send_message(chat_id, decision.get("response_text", "Я вас слухаю."), parse_mode="HTML")
                self._log_user_activity(str(user_id), username, text, "CHAT", "none")
                return
                
        except Exception as e:
            logger.error(f"[{self.slug}] Intent Analysis Error: {e}")
            self.record_fsm_error(text, f"Intent Error: {str(e)}")

        # --- Step 2: Unified Search ---
        try:
            if not self.kernel:
                raise Exception("Kernel reference missing")

            entities = decision.get("entities", {})
            # Integration with intelligence.py filters
            filters = {
                "brand": entities.get("brand"),
                "animal": entities.get("animal"),
                "category": entities.get("category"),
                "price_limit": entities.get("price_limit") or entities.get("max_price")
            }
            filters = {k: v for k, v in filters.items() if v is not None}

            final_response = await self.kernel.get_recommendations(
                ctx=self.ctx,
                query=text,
                filters=filters,
                top_k=5,
                user_id=user_id
            )

            # --- Step 3: Response Delivery (Fallback Logic) ---
            if final_response:
                # FIXED: changed from Markdown to HTML to support kernel output
                await self.send_message(chat_id, final_response, parse_mode="HTML")
            else:
                # Fallback to manual list if synthesis failed
                fallback_result = await self.retrieval.search(
                    query=text, entities=filters or {}, top_k=5
                )
                products = fallback_result.get("products", [])
                if products:
                    await self.send_products(chat_id, products)
                else:
                    fail_msg = getattr(self.ctx, 'prompts', {}).get("not_found", "На жаль, нічого не знайдено.")
                    await self.send_message(chat_id, fail_msg, parse_mode="HTML")
            
            self._log_user_activity(str(user_id), username, text, "SUCCESS", entities.get("category", "general"))

        except Exception as e:
            logger.error(f"[{self.slug}] Processing Error: {e}")
            self.record_fsm_error(text, str(e))
            fail_msg = getattr(self.ctx, 'prompts', {}).get("not_found", "Вибачте, сталася помилка.")
            await self.send_message(chat_id, fail_msg, parse_mode="HTML")

    async def send_message(self, chat_id: int, text: str, parse_mode: str = "HTML"):
        """Sends message with flexible parse_mode support. Default changed to HTML."""
        url = f"{self.api_url}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
        session = self._get_session()
        try:
            async with session.post(url, json=payload) as response:
                result = await response.json()
                if result.get("ok"):
                    msg_id = result.get("result", {}).get("message_id")
                    self._log_chat_history(str(chat_id), str(msg_id), "assistant", text)
                return result
        except Exception as e:
            logger.error(f"[{self.slug}] Telegram Send Error: {e}")

    async def send_products(self, chat_id: int, products: list):
        """LEGACY Fallback: Manual product formatting if Synthesis fails."""
        lines = []
        prompts = getattr(self.ctx, 'prompts', {})
        header = prompts.get("search_header", "Ось що я знайшов:")
        price_label = prompts.get("price_label", "Ціна")

        for idx, p in enumerate(products[:5], 1):
            data = p.get("data", p)
            name = data.get("title", data.get("name", "Товар"))
            price = data.get("price", "---")
            link = data.get("link", "")
            # Using HTML tags for consistency in fallback
            item = f"{idx}. <b>{name}</b>\n{price_label}: {price} {self.currency}"
            if link: item += f" — <a href='{link}'>Купити</a>"
            lines.append(item)
        
        await self.send_message(chat_id, f"{header}\n\n" + "\n\n".join(lines), parse_mode="HTML")