# /root/ukrsell_v4/core/dialog_manager.py v8.0.0

import json
import os
import asyncio
import re
import time
import logging
import aiosqlite
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union

from core.logger import logger, log_event





class DialogManager:
    """Dialog Manager v8.0.0 — история диалога, troll handling. Store-agnostic."""

    def __init__(self, ctx, llm_selector):
        self.ctx             = ctx
        self.selector        = llm_selector
        self.language        = getattr(ctx, 'language', 'Ukrainian')
        self.slug            = getattr(ctx, 'slug', 'Store')
        self.base_path       = getattr(ctx, 'base_path', '/root/ukrsell_v4')
        self.patch_path      = os.path.join(self.base_path, "fsm_soft_patch.json")
        self.session_db_path = os.path.join(self.base_path, "sessions.db")

        self._intent_cache: Dict[str, Dict] = {}
        self._intent_hints  = self._load_intent_hints()
        self._search_config = self._load_search_config()
        self._negative_examples_cache: Optional[List] = None
        self._patch_lock = asyncio.Lock()  # защита от race condition на fsm_soft_patch.json

        self._init_db()
        logger.info(f"✅ [DM_INIT] DialogManager v8.0.0 Active. Store: {self.slug}")

    # ── DB Init ───────────────────────────────────────────────────────────────

    def _init_db(self):
        """Синхронная инициализация DDL при старте — вызывается до event loop."""
        try:
            import sqlite3 as _sqlite3
            conn   = _sqlite3.connect(self.session_db_path, timeout=10.0)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id         INTEGER  PRIMARY KEY AUTOINCREMENT,
                    chat_id    TEXT     NOT NULL,
                    role       TEXT     NOT NULL,
                    content    TEXT     NOT NULL,
                    session_id TEXT,
                    timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"[{self.slug}] Database Init Error: {e}")

    # ── Config ────────────────────────────────────────────────────────────────

    def _load_intent_hints(self) -> dict:
        path = os.path.join(self.base_path, "intent_hints.json")
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ [DM_INIT] intent_hints.json load error: {e}")
        return {}

    def _load_search_config(self) -> dict:
        path = os.path.join(self.base_path, "search_config.json")
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ [DM_INIT] search_config.json load error: {e}")
        return {}

    def get_negative_examples(self) -> list:
        if self._negative_examples_cache is not None:
            return self._negative_examples_cache
        try:
            if os.path.exists(self.patch_path):
                with open(self.patch_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._negative_examples_cache = data.get("troll_patterns", [])
                    return self._negative_examples_cache
        except Exception as e:
            logger.error(f"❌ [DM_ERROR] Failed to load negative patterns: {e}")
        self._negative_examples_cache = []
        return self._negative_examples_cache

    # ── Intent Cache ──────────────────────────────────────────────────────────

    async def get_chat_context(self, chat_id: str, minutes: int = None) -> str:
        if minutes is None:
            minutes = self._search_config.get("context_ttl_min", 15)
        try:
            if not os.path.exists(self.session_db_path):
                logger.warning(f"⚠️ [DB_TRACE] Session DB missing at {self.session_db_path}")
                return ""

            time_threshold = (datetime.now() - timedelta(minutes=minutes)).strftime(
                '%Y-%m-%d %H:%M:%S'
            )
            logger.debug(f"📅 [DB_TRACE] Fetching history for {chat_id} since {time_threshold}")

            async with aiosqlite.connect(self.session_db_path, timeout=10.0) as db:
                async with db.execute(
                    """
                    SELECT role, content FROM chat_history
                    WHERE chat_id = ? AND timestamp > ?
                    ORDER BY timestamp DESC LIMIT 10
                    """,
                    (str(chat_id), time_threshold),
                ) as cursor:
                    rows = await cursor.fetchall()

            if not rows:
                logger.debug(f"ℹ️ [DB_TRACE] No history found for chat_id: {chat_id}")
                return ""

            # Метки ролей берутся из search_config магазина
            buyer_label     = self._search_config.get("label_buyer", "Buyer")
            consultant_label = self._search_config.get("label_consultant", "Consultant")

            history = []
            for row in reversed(rows):
                label = buyer_label if row[0] == "user" else consultant_label
                history.append(f"{label}: {row[1]}")

            full_context = "\n".join(history)
            logger.debug(f"📜 [DB_TRACE] Retrieved {len(rows)} messages. Context snippet: {full_context[:100]}...")
            return full_context

        except Exception as e:
            logger.error(f"❌ [DM_ERROR] SQL Context retrieval error: {e}")
            return ""

    async def save_history(
        self,
        chat_id: str,
        role: str,
        content: str,
        session_id: Optional[str] = None,
    ):
        try:
            async with aiosqlite.connect(self.session_db_path, timeout=10.0) as db:
                await db.execute(
                    """
                    INSERT INTO chat_history (chat_id, role, content, session_id, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(chat_id),
                        role,
                        content,
                        session_id,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    ),
                )
                await db.commit()
            logger.debug(f"💾 [DB_TRACE] Saved {role} message for {chat_id} (session={session_id})")
        except Exception as e:
            logger.error(f"[{self.slug}] History Save Error: {e}")

    async def record_troll_pattern(self, user_text: str):
        async with self._patch_lock:
            try:
                data: Dict[str, Any] = {"troll_patterns": [], "fsm_errors": []}
                if os.path.exists(self.patch_path):
                    with open(self.patch_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                        except Exception:
                            pass
                clean_text = str(user_text).lower().strip()[:200]  # защита от overflow
                if not clean_text:
                    return
                if clean_text not in data.get("troll_patterns", []):
                    data.setdefault("troll_patterns", []).append(clean_text)
                    data.setdefault("fsm_errors", []).append({
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "request":   user_text[:200],
                        "type":      "negative_example",
                    })
                    with open(self.patch_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    self._negative_examples_cache = None
                    logger.warning(f"⚠️ [TROLL_DETECTED] Pattern saved: {clean_text}")
            except Exception as e:
                logger.error(f"❌ [DM_ERROR] Failed to record troll pattern: {e}")

    # ── Prompt building ───────────────────────────────────────────────────────

    async def _handle_troll_response(self, user_text: str) -> str:
        import random
        troll_responses = self._search_config.get("troll_responses", [])
        prompt_path_raw = self._search_config.get("troll_prompt", "")

        if prompt_path_raw:
            prompt_path = prompt_path_raw if os.path.isabs(prompt_path_raw) else os.path.join(self.base_path, prompt_path_raw)
        else:
            prompt_path = ""

        if prompt_path and os.path.exists(prompt_path):
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    prompt_template = f.read()
                prompt = prompt_template.format(
                    store_name=self.slug,
                    language=self.language,
                )
                result_obj = await self.selector.get_fast()
                if isinstance(result_obj, tuple):
                    client, model = result_obj
                else:
                    client = result_obj
                    model  = getattr(result_obj, 'model_name', None)
                if not model:
                    raise ValueError("model is None — skipping LLM troll response")
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user",   "content": user_text},
                        ],
                        temperature=0.7,
                        max_tokens=80,
                    ),
                    timeout=8.0,
                )
                text = ""
                if response.choices:
                    text = getattr(response.choices[0].message, 'content', '').strip()
                if text:
                    return text
            except Exception as e:
                logger.warning(f"[{self.slug}] Troll LLM failed: {e} → using fallback")

        if troll_responses:
            return random.choice(troll_responses)
        return "🙃"

    def __repr__(self):
        return f"<DialogManager v8.0.0 slug={self.slug}>"


def get_version():
    return "8.0.0"