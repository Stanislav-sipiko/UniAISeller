# /root/ukrsell_v4/base.py v7.8.0
import aiohttp
import json
import os
import re
import datetime
import aiosqlite
import asyncio
import hashlib
import math
import html
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

from core.logger import logger, log_event
from core.store_context import StoreContext
from core.retrieval import RetrievalEngine
from core.dialog_manager import DialogManager
from core.analyzer import Analyzer

try:
    from engine.keyboards import get_main_menu
except ImportError:
    def get_main_menu(ctx):
        return {"keyboard": [[{"text": "🔍 Пошук"}]], "resize_keyboard": True}

# ── Константы конфигурации ────────────────────────────────────────────────────
DB_TIMEOUT              = 15.0
HTTP_TIMEOUT            = aiohttp.ClientTimeout(total=25, connect=5)
LOG_BATCH_INTERVAL      = 3.0
LOG_BATCH_SIZE          = 50
SEMANTIC_CACHE_TTL_DAYS = 7
GLOBAL_UPDATE_TIMEOUT   = 40.0
SEND_TIMEOUT            = 18.0
MAX_CACHE_TEXT_SIZE     = 1024 * 128
MAX_TG_MESSAGE_SIZE     = 4000
DEDUPLICATE_BATCH_SIZE  = 10
MAX_RETRY_WAIT          = 30

# ── Пороги качества ───────────────────────────────────────────────────────────
MIN_RESULTS_THRESHOLD = 3
MAX_DUPLICATES_SCORE  = 0.94
MIN_QUERY_LENGTH      = 2

STOP_WORDS = {
    "мені", "треба", "хочу", "купити", "для", "знайти", "покажи", "будь", "ласка",
    "мне", "надо", "хочу", "купить", "покажи", "пожалуйста", "на", "в", "с", "и", "та",
}


# ── Инициализация БД ──────────────────────────────────────────────────────────

async def init_store_db(db_path: str) -> bool:
    try:
        async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute("PRAGMA synchronous=NORMAL;")
            await db.execute("PRAGMA foreign_keys=ON;")
            await db.execute("PRAGMA busy_timeout=15000;")

            await db.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    product_id   TEXT PRIMARY KEY,
                    title        TEXT,
                    category     TEXT,
                    category_ukr TEXT,
                    subtype      TEXT,
                    brand        TEXT,
                    animal       TEXT,
                    price_min    REAL,
                    price_max    REAL,
                    image_url    TEXT,
                    search_blob  TEXT,
                    attributes   TEXT,
                    variants     TEXT
                );
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS faiss_map (
                    position   INTEGER PRIMARY KEY,
                    product_id TEXT NOT NULL,
                    FOREIGN KEY(product_id) REFERENCES products(product_id) ON DELETE CASCADE
                );
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    query_hash    TEXT PRIMARY KEY,
                    query_text    TEXT NOT NULL,
                    response_json TEXT,
                    timestamp     DATETIME DEFAULT (DATETIME('now', 'utc')),
                    expires_at    DATETIME
                );
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS fsm_errors (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT (DATETIME('now', 'utc')),
                    query     TEXT,
                    reason    TEXT,
                    type      TEXT
                );
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id    TEXT,
                    message_id TEXT,
                    role       TEXT,
                    content    TEXT,
                    timestamp  DATETIME DEFAULT (DATETIME('now', 'utc'))
                );
            """)

            await db.execute("CREATE INDEX IF NOT EXISTS idx_faiss_pid     ON faiss_map(product_id);")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_cache_expires ON semantic_cache(expires_at);")
            await db.commit()
            return True
    except Exception as e:
        logger.error(f"DB Init Fatal error for {db_path}: {e}")
        return False


# ── Утилиты ───────────────────────────────────────────────────────────────────

def _jaccard_similarity(a: str, b: str) -> float:
    set_a = set(re.findall(r"\w+", a.lower()))
    set_b = set(re.findall(r"\w+", b.lower()))
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _parse_variants(raw_variants: Any) -> List[Dict]:
    if isinstance(raw_variants, list):
        return raw_variants
    if isinstance(raw_variants, str) and raw_variants.strip():
        try:
            parsed = json.loads(raw_variants)
            if isinstance(parsed, list):
                return parsed
        except Exception as e:
            logger.debug(f"_parse_variants: failed to parse variants JSON: {e}")
    return []


def _safe_truncate_html(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text

    truncated = text[: max_len - 120]
    truncated = re.sub(r"<[^>]*$", "", truncated)

    tag_pattern = re.compile(
        r"<(/?)(?P<tag>b|i|u|s|a|code|pre)(?:\s[^>]*)?>",
        flags=re.IGNORECASE,
    )
    tag_stack: List[str] = []
    for m in tag_pattern.finditer(truncated):
        is_closing = bool(m.group(1))
        tag_name = m.group("tag").lower()
        if is_closing:
            if tag_stack and tag_stack[-1] == tag_name:
                tag_stack.pop()
        else:
            tag_stack.append(tag_name)

    closing = "".join(f"</{t}>" for t in reversed(tag_stack))
    return truncated + closing + "\n\n<i>(текст скорочено через ліміти)</i>"


def _build_safe_url(raw_url: str) -> str:
    if not isinstance(raw_url, str) or not raw_url.startswith("http"):
        return ""
    try:
        parsed = urllib.parse.urlparse(raw_url)
        safe_path = urllib.parse.quote(parsed.path, safe="/")
        safe_query = urllib.parse.urlencode(
            urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        )
        result = f"{parsed.scheme}://{parsed.netloc}{safe_path}"
        if safe_query:
            result += f"?{safe_query}"
        return result
    except Exception:
        return ""


class SimpleEmbedding:
    @staticmethod
    async def get_vec(text: str) -> List[float]:
        if not text:
            return [0.0] * 64
        clean_text = re.sub(r"[^\w\s]", "", text.lower())
        words = clean_text.split()
        vec = [0.0] * 64
        if not words:
            return vec
        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            for i in range(64):
                vec[i] += float((h >> i) & 1)
        norm = math.sqrt(sum(x * x for x in vec))
        if norm < 1e-12:
            return [0.0] * 64
        return [x / norm for x in vec]


# ── StoreEngine ───────────────────────────────────────────────────────────────

class StoreEngine:

    def __init__(self, ctx: StoreContext):
        self.ctx = ctx
        self.slug = ctx.slug
        self.config = getattr(ctx, "config", {})
        self.token = self.config.get("bot_token", "NO_TOKEN")
        self.currency = self.config.get("currency", "грн")
        self.api_url = f"https://api.telegram.org/bot{self.token}"

        self.retrieval = getattr(ctx, "retrieval", None) or RetrievalEngine(ctx, None, None)
        self.dialog_manager = getattr(ctx, "dialog_manager", None) or DialogManager(ctx, None)
        self.kernel = getattr(ctx, "kernel", None)
        self.analyzer = getattr(ctx, "analyzer", None) or Analyzer(ctx)

        self.db_path = os.path.join(ctx.base_path, "store.db")
        self.profile_path = os.path.join(ctx.base_path, "store_profile.json")

        self._session: Optional[aiohttp.ClientSession] = None
        self.db: Optional[aiosqlite.Connection] = None
        self._log_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._log_worker_task: Optional[asyncio.Task] = None
        self._is_closing = False
        self._session_lock = asyncio.Lock()
        self._db_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()

    async def __aenter__(self):
        await init_store_db(self.db_path)
        async with self._session_lock:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(timeout=HTTP_TIMEOUT)

        self.db = await aiosqlite.connect(self.db_path, timeout=DB_TIMEOUT)
        self.db.row_factory = aiosqlite.Row

        if self._log_worker_task is None or self._log_worker_task.done():
            self._log_worker_task = asyncio.create_task(self._log_worker())

        if hasattr(self.ctx, "data_ready") and isinstance(self.ctx.data_ready, asyncio.Event):
            try:
                await asyncio.wait_for(self.ctx.data_ready.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                logger.warning(f"[{self.slug}] Timeout waiting for data_ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def _normalize_query(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+-\s+", " ", text)
        text = re.sub(r"[^\w\s\-]", " ", text, flags=re.UNICODE)
        words = text.split()
        filtered = [w for w in words if w not in STOP_WORDS]
        return " ".join(filtered) if filtered else text

    async def _log_worker(self):
        buffer: List[Dict[str, Any]] = []
        last_flush = asyncio.get_running_loop().time()
        try:
            while True:
                should_stop = self._is_closing and self._log_queue.empty() and not buffer
                now = asyncio.get_running_loop().time()
                
                try:
                    item = await asyncio.wait_for(self._log_queue.get(), timeout=0.5)
                    buffer.append(item)
                except asyncio.TimeoutError:
                    pass
                except asyncio.CancelledError:
                    if buffer:
                        await self._flush_logs(buffer)
                        buffer.clear()
                    break

                time_passed = now - last_flush
                if buffer and (len(buffer) >= LOG_BATCH_SIZE or time_passed >= LOG_BATCH_INTERVAL or self._is_closing):
                    await self._flush_logs(buffer)
                    buffer.clear()
                    last_flush = now

                if should_stop:
                    break
        finally:
            self._shutdown_event.set()

    async def _flush_logs(self, logs: List[Dict[str, Any]]):
        if not self.db:
            return
        async with self._db_lock:
            try:
                chat_rows: List[Tuple] = []
                fsm_rows: List[Tuple] = []
                for log in logs:
                    raw_ts = log.get("timestamp") or datetime.datetime.now(datetime.timezone.utc)
                    ts = raw_ts.isoformat() if isinstance(raw_ts, datetime.datetime) else str(raw_ts)
                    if log["type"] == "chat":
                        chat_rows.append((log["chat_id"], log["message_id"], log["role"], log["content"], ts))
                    elif log["type"] == "fsm":
                        fsm_rows.append((ts, log["query"], log["reason"], "FSM_ERROR"))

                if chat_rows:
                    await self.db.executemany(
                        "INSERT INTO chat_history (chat_id, message_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
                        chat_rows,
                    )
                if fsm_rows:
                    await self.db.executemany(
                        "INSERT INTO fsm_errors (timestamp, query, reason, type) VALUES (?, ?, ?, ?)",
                        fsm_rows,
                    )
                await self.db.commit()
            except Exception as e:
                logger.error(f"[{self.slug}] DB Flush Failure: {e}")

    def _queue_log(self, data: Dict[str, Any]):
        if not self._is_closing:
            if "timestamp" not in data:
                data["timestamp"] = datetime.datetime.now(datetime.timezone.utc)
            try:
                self._log_queue.put_nowait(data)
            except asyncio.QueueFull:
                logger.warning(f"[{self.slug}] Log queue overflow, dropping record.")

    async def _get_cached_response(self, query: str) -> Optional[str]:
        if not self.db:
            return None
        qh = hashlib.sha256(query.encode()).hexdigest()
        try:
            async with self.db.execute(
                "SELECT response_json FROM semantic_cache WHERE query_hash = ? AND expires_at > DATETIME('now', 'utc')",
                (qh,),
            ) as cur:
                row = await cur.fetchone()
                if row is None or row["response_json"] is None:
                    return None
                return json.loads(row["response_json"])
        except Exception as e:
            logger.error(f"[{self.slug}] Cache retrieval error: {e}")
            return None

    async def _set_cache(self, query: str, response: str):
        if not self.db or len(response) > MAX_CACHE_TEXT_SIZE:
            return
        qh = hashlib.sha256(query.encode()).hexdigest()
        expires_at = (
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(days=SEMANTIC_CACHE_TTL_DAYS)
        ).isoformat()
        try:
            async with self._db_lock:
                await self.db.execute(
                    "INSERT OR REPLACE INTO semantic_cache (query_hash, query_text, response_json, expires_at) VALUES (?, ?, ?, ?)",
                    (qh, query, json.dumps(response, ensure_ascii=False), expires_at),
                )
                await self.db.commit()
        except Exception as e:
            logger.error(f"[{self.slug}] Cache storage error: {e}")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    self._session = aiohttp.ClientSession(timeout=HTTP_TIMEOUT)
        return self._session

    async def close(self):
        async with self._close_lock:
            if self._is_closing:
                return
            self._is_closing = True

        try:
            if self._log_worker_task:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=7.0)
        except asyncio.TimeoutError:
            logger.warning(f"[{self.slug}] Log worker shutdown timeout.")
            if self._log_worker_task:
                self._log_worker_task.cancel()

        try:
            async with self._db_lock:
                if self.db:
                    await self.db.close()
        except Exception as e:
            logger.error(f"[{self.slug}] Error closing DB: {e}")

        try:
            async with self._session_lock:
                if self._session and not self._session.closed:
                    await self._session.close()
        except Exception as e:
            logger.error(f"[{self.slug}] Error closing session: {e}")

    async def _deduplicate(self, products: List[Dict]) -> List[Dict]:
        if not products:
            return []
        unique: List[Dict] = []
        unique_titles: List[str] = []

        for p in products:
            title = str(p.get("product", p).get("title", ""))
            is_dup = any(
                _jaccard_similarity(title, t) > MAX_DUPLICATES_SCORE
                for t in unique_titles[-50:]
            )
            if not is_dup:
                unique.append(p)
                unique_titles.append(title)
        return unique

    async def handle_update(self, update: Dict[str, Any]):
        chat_id = update.get("message", {}).get("chat", {}).get("id")
        if not chat_id:
            return

        final_products: List[Dict] = []
        final_response: Optional[str] = None

        try:
            final_products, final_response = await asyncio.wait_for(
                self._run_pipeline(update), timeout=GLOBAL_UPDATE_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning(f"[{self.slug}] Pipeline timeout for chat {chat_id}")
            if final_products:
                await asyncio.wait_for(self._send_fallback_list(chat_id, final_products), timeout=SEND_TIMEOUT)
                return
            await asyncio.wait_for(
                self.send_message(chat_id, "⚠️ Запит обробляється довше ніж зазвичай. Спробуйте ще раз через хвилину."),
                timeout=SEND_TIMEOUT
            )
            return
        except Exception as e:
            logger.error(f"[{self.slug}] Handle Update Crash: {e}", exc_info=True)
            return

        try:
            if final_response:
                await asyncio.wait_for(self.send_message(chat_id, final_response), timeout=SEND_TIMEOUT)
            elif final_products:
                await asyncio.wait_for(self._send_fallback_list(chat_id, final_products), timeout=SEND_TIMEOUT)
        except Exception as e:
            logger.error(f"[{self.slug}] Final send failed: {e}")

    async def _run_pipeline(self, update: Dict[str, Any]) -> Tuple[List[Dict], Optional[str]]:
        message  = update.get("message", {})
        chat_id  = message.get("chat", {}).get("id")
        raw_text = message.get("text", "").strip()
        user_id  = message.get("from", {}).get("id")

        if not raw_text:
            return [], None

        norm_query = self._normalize_query(raw_text)
        self._queue_log({"type": "chat", "chat_id": str(chat_id), "message_id": str(message.get("message_id")), "role": "user", "content": raw_text})

        if raw_text.startswith("/start"):
            await self.send_message(chat_id, "Вітаю! Я допоможу знайти потрібні товари. Що саме Ви шукаєте?", reply_markup=get_main_menu(self.ctx))
            return [], None

        try:
            decision = await asyncio.wait_for(self.dialog_manager.analyze_intent(norm_query, str(chat_id)), timeout=10.0)
        except Exception as e:
            logger.warning(f"[{self.slug}] Intent analysis failed: {e}")
            decision = {"action": "SEARCH", "entities": {}}

        action = decision.get("action", "SEARCH")
        if action in ["TROLL", "CHAT"]:
            await self.send_message(chat_id, decision.get("response_text", "🧐 Опишіть товар детальніше."))
            return [], None

        cached = await self._get_cached_response(norm_query)
        if cached:
            await self.send_message(chat_id, cached)
            return [], None

        start_time = asyncio.get_running_loop().time()
        entities = decision.get("entities", {})
        elapsed = asyncio.get_running_loop().time() - start_time
        search_timeout = max(5.0, 20.0 - elapsed)

        search_result = await asyncio.wait_for(self.retrieval.search(query=norm_query, entities=entities, top_k=15), timeout=search_timeout)
        products: List[Dict] = search_result.get("products", [])

        if not products:
            await self.send_message(chat_id, "😔 На жаль, за Вашим запитом нічого не знайдено.")
            return [], None

        products = await self._deduplicate(products)
        products = products[:5]

        if self.kernel:
            elapsed = asyncio.get_running_loop().time() - start_time
            format_timeout = max(5.0, 28.0 - elapsed)
            try:
                formatted = await asyncio.wait_for(self.kernel.format_products(raw_text, products, user_id), timeout=format_timeout)
                if formatted and len(formatted) > 20:
                    await self._set_cache(norm_query, formatted)
                    return products, formatted
            except Exception as e:
                logger.warning(f"[{self.slug}] Kernel format_products failed, using fallback: {e}")

        return products, None

    async def _send_fallback_list(self, chat_id: int, products: List[Dict]):
        if not products:
            await self.send_message(chat_id, "😔 Товари тимчасово недоступні.")
            return

        lines = ["<b>Знайдені товари:</b>"]
        for idx, p in enumerate(products, 1):
            data = p.get("product", p)
            title = html.escape(str(data.get("title", "Товар")))
            variants = _parse_variants(data.get("variants"))
            first_variant = variants[0] if variants else {}
            price_val = data.get("price_min") or first_variant.get("price")
            price_str = html.escape(str(price_val)) if price_val is not None else "---"
            raw_url = data.get("product_url") or first_variant.get("product_url", "")
            safe_url = _build_safe_url(raw_url)

            item = f"{idx}. <b>{title}</b>\nЦіна: {price_str} {self.currency}"
            if safe_url:
                item += f" — <a href='{html.escape(safe_url)}'>Перейти</a>"
            lines.append(item)

        await self.send_message(chat_id, "\n\n".join(lines))

    async def send_message(self, chat_id: int, text: str, parse_mode: str = "HTML", reply_markup: Optional[Dict] = None) -> Optional[Dict]:
        text = _safe_truncate_html(text, MAX_TG_MESSAGE_SIZE)
        url = f"{self.api_url}/sendMessage"
        payload: Dict[str, Any] = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
        if reply_markup:
            payload["reply_markup"] = reply_markup

        session = await self._get_session()
        for attempt in range(4):
            try:
                async with session.post(url, json=payload, timeout=15) as resp:
                    data = await resp.json()
                    if data.get("ok"):
                        m_id = data.get("result", {}).get("message_id")
                        self._queue_log({"type": "chat", "chat_id": str(chat_id), "message_id": str(m_id), "role": "assistant", "content": text[:500]})
                        return data

                    if resp.status == 429:
                        raw_wait = data.get("parameters", {}).get("retry_after", 3 * (attempt + 1))
                        wait = min(int(raw_wait), MAX_RETRY_WAIT)
                        logger.warning(f"[{self.slug}] TG Rate Limit, waiting {wait}s (attempt {attempt + 1})")
                        await asyncio.sleep(wait)
                        continue

                    logger.warning(f"[{self.slug}] TG API Error Response: {data}")
                    if "can't parse entities" in str(data.get("description", "")):
                        payload["parse_mode"] = None
                        continue
                    return data
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[{self.slug}] Send Attempt {attempt+1} failed: {e}")
                if attempt < 3:
                    await asyncio.sleep(2 * (attempt + 1))
        return None