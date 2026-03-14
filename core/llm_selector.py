# /root/ukrsell_v4/core/llm_selector.py v7.6.0
import time
import asyncio
import threading
import openai
import aiohttp
import json
import os
from pathlib import Path
from typing import Tuple, Any, Optional, Dict, List
from core.logger import logger, log_event
from core.config import GROQ_KEYS, LLM_TIERS

# --- Non-LLM model filter ---
_NON_LLM_PATTERNS = [
    "whisper", "tts", "dall-e", "stable-diffusion", "embedding",
    "rerank", "transcription", "ocr", "vision-encoder", "speech",
    "guard", "moderation", "classifier", "detector",
    "orpheus", "bark", "xtts", "kokoro", "safeguard",
]

def _is_llm_model(model_id: str) -> bool:
    ml = model_id.lower()
    return not any(p in ml for p in _NON_LLM_PATTERNS)


BLACKLIST_DURATION = 180  # seconds
RATING_FILE = Path("/root/ukrsell_v4/core/llm_rating.json")


class LLMSelector:
    """
    LLM Selector v7.6.0
    Groq-only. Integrated with central config.py LLM_TIERS and llm_rating.json.
    Changes vs v7.5.4:
      - INTELLIGENCE-FIRST: Added support for llm_rating.json to prioritize "smartest" models.
      - HEAVY-LOGIC: In 'heavy' tier, ignores latency and selects by rank.
      - FALLBACK: Reverts to latency-based selection if rating file is missing or invalid.
      - PERSISTENCE: Rating is loaded during initialization.
    """

    def __init__(self):
        # threading.Lock ONLY for key-index rotation (sync, never held across awaits)
        self._key_lock = threading.Lock()
        self._key_indices: Dict[str, int] = {"groq": 0}

        # asyncio primitives — created lazily to avoid event loop issues
        self._async_lock: Optional[asyncio.Lock] = None
        self._refresh_event: Optional[asyncio.Event] = None

        self._last_refresh_time = 0.0
        self._refresh_ttl = 1800
        self._blacklisted_models: Dict[str, float] = {}
        self._current_loop_id: Optional[int] = None

        self._session: Optional[aiohttp.ClientSession] = None

        self.stacks: Dict[str, List[Dict]] = {"light": [], "fast": [], "heavy": []}
        self.active: Dict[str, Optional[Dict]] = {"light": None, "fast": None, "heavy": None}
        
        # Load intelligence rating
        self.model_ranks: Dict[str, int] = self._load_ranks()

    def _load_ranks(self) -> Dict[str, int]:
        if not RATING_FILE.exists():
            logger.warning(f"[LLMSelector] Rating file not found at {RATING_FILE}. Using latency logic.")
            return {}
        try:
            with open(RATING_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[LLMSelector] Failed to load {RATING_FILE}: {e}")
            return {}

    # -------------------------------------------------------------------------
    # Lazy async primitives
    # -------------------------------------------------------------------------
    def _ensure_async_primitives(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
            
        loop_id = id(loop)
        if self._async_lock is None or self._current_loop_id != loop_id:
            self._async_lock = asyncio.Lock()
            self._refresh_event = asyncio.Event()
            self._refresh_event.set()
            self._current_loop_id = loop_id

    # -------------------------------------------------------------------------
    # Client initialisation
    # -------------------------------------------------------------------------
    async def _init_clients_async(self):
        self._ensure_async_primitives()
        if self._async_lock is None:
            return

        async with self._async_lock:
            loop_id = id(asyncio.get_running_loop())
            if self._session and not self._session.closed and self._current_loop_id == loop_id:
                return

            if self._session and not self._session.closed:
                try:
                    await self._session.close()
                except Exception:
                    pass

            self._session = aiohttp.ClientSession()
            self._current_loop_id = loop_id
            log_event("LLM_CLIENTS_INIT", {"status": "success", "provider": "groq", "ver": "7.6.0"})

    # -------------------------------------------------------------------------
    # Key rotation
    # -------------------------------------------------------------------------
    def _next_key(self) -> Tuple[str, int]:
        with self._key_lock:
            idx = self._key_indices["groq"]
            key = GROQ_KEYS[idx % len(GROQ_KEYS)]
            num = (idx % len(GROQ_KEYS)) + 1
            self._key_indices["groq"] = (idx + 1) % len(GROQ_KEYS)
            return key, num

    # -------------------------------------------------------------------------
    # Build client
    # -------------------------------------------------------------------------
    def _build_client_ping(self, model: str) -> Optional[Any]:
        """Returns client using fixed key #0. No rotation. For ping only."""
        try:
            key = GROQ_KEYS[0]
            return openai.AsyncOpenAI(
                api_key=key,
                base_url="https://api.groq.com/openai/v1"
            )
        except Exception as e:
            logger.error(f"[LLMSelector] Failed to build ping client for {model}: {e}")
            return None

    def _build_client_dispatch(self, model: str) -> Tuple[Optional[Any], str]:
        """Returns (client, label) with rotated key. For real requests only."""
        try:
            key, n = self._next_key()
            client = openai.AsyncOpenAI(
                api_key=key,
                base_url="https://api.groq.com/openai/v1"
            )
            label = f"[Key #{n}]"
            return client, label
        except Exception as e:
            logger.error(f"[LLMSelector] Failed to build dispatch client for {model}: {e}")
            return None, ""

    # -------------------------------------------------------------------------
    # Static stack build from LLM_TIERS (Config)
    # -------------------------------------------------------------------------
    def _build_stacks(self):
        new_stacks: Dict[str, List[Dict]] = {"light": [], "fast": [], "heavy": []}
        for tier, models in LLM_TIERS.items():
            for model in models:
                if _is_llm_model(model):
                    new_stacks[tier].append({"type": "groq", "model": model})
        self.stacks = new_stacks
        logger.info(
            f"[LLMSelector] Stacks built from config: "
            f"light={len(self.stacks['light'])} "
            f"fast={len(self.stacks['fast'])} "
            f"heavy={len(self.stacks['heavy'])}"
        )

    # -------------------------------------------------------------------------
    # Health-check ping with latency measurement
    # -------------------------------------------------------------------------
    async def _test_ping(self, client: Any, model: str) -> Tuple[bool, float]:
        start = time.time()
        try:
            await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                timeout=5.0,
            )
            lat = round((time.time() - start) * 1000, 2)
            return True, lat
        except Exception as e:
            logger.debug(f"[LLMSelector] Ping failed [{model}]: {e}")
            return False, 0.0

    async def _find_fastest_healthy(self, stack: List[Dict], tier: str = "light") -> Optional[Dict]:
        now = time.time()
        self._ensure_async_primitives()
        if self._async_lock is None: return None

        async with self._async_lock:
            blacklist_snapshot = dict(self._blacklisted_models)

        candidates = []
        for entry in stack:
            if blacklist_snapshot.get(entry["model"], 0) > now:
                continue
            candidates.append(entry)

        if not candidates:
            return None

        async def ping_entry(entry: Dict) -> Optional[Dict]:
            model_id = entry["model"]
            ping_client = self._build_client_ping(model_id)
            if not ping_client:
                return None
            ok, lat = await self._test_ping(ping_client, model_id)
            if ok:
                # Add intelligence rank if available
                rank = self.model_ranks.get(model_id, 999)
                logger.info(f"[LLMSelector] ✅ Healthy: {model_id} latency={lat}ms rank={rank}")
                return {**entry, "active_client": None, "active_model": model_id, "latency_ms": lat, "rank": rank}
            else:
                self.report_failure(model_id, BLACKLIST_DURATION)
                return None

        results = await asyncio.gather(*[ping_entry(e) for e in candidates], return_exceptions=True)
        healthy = [r for r in results if isinstance(r, dict)]
        if not healthy:
            return None

        # --- SELECTION LOGIC ---
        if tier == "heavy" and self.model_ranks:
            # For heavy tier: prioritize Intelligence (Rank) over Latency
            # We pick the model with the lowest Rank (1 is best)
            best_by_rank = min(healthy, key=lambda x: x.get("rank", 999))
            logger.info(f"[LLMSelector] 🧠 Selected HEAVY by Rank: {best_by_rank['model']} (Rank {best_by_rank['rank']})")
            return best_by_rank
        else:
            # For other tiers: keep choosing the fastest healthy model
            fastest = min(healthy, key=lambda x: x.get("latency_ms", 9999))
            return fastest

    # -------------------------------------------------------------------------
    # Prewarm & Refresh
    # -------------------------------------------------------------------------
    async def prewarm(self):
        await self._init_clients_async()
        self._build_stacks()
        logger.info("[LLMSelector] 🔥 Prewarming light tier...")
        result = await self._find_fastest_healthy(self.stacks["light"], "light")
        if self._async_lock:
            async with self._async_lock:
                self.active["light"] = result

    async def ensure_ready(self):
        self._ensure_async_primitives()
        await self._init_clients_async()
        if self._refresh_event is None: return

        needs_refresh = (
            (time.time() - self._last_refresh_time) > self._refresh_ttl
            or not any(self.active.values())
        )
        if not needs_refresh:
            return
        await self._refresh_event.wait()
        if (time.time() - self._last_refresh_time) <= self._refresh_ttl and any(self.active.values()):
            return
        self._refresh_event.clear()
        try:
            await self.refresh()
        finally:
            self._refresh_event.set()

    async def refresh(self, force: bool = False):
        await self._init_clients_async()
        self._build_stacks()
        # Reload ranks on refresh to catch manual changes
        self.model_ranks = self._load_ranks()
        
        logger.info("[LLMSelector] 🧪 Health-checking groq stacks (intelligence-aware)...")
        results = await asyncio.gather(
            self._find_fastest_healthy(self.stacks["light"], "light"),
            self._find_fastest_healthy(self.stacks["fast"], "fast"),
            self._find_fastest_healthy(self.stacks["heavy"], "heavy"),
            return_exceptions=True,
        )
        if self._async_lock:
            async with self._async_lock:
                for tier, res in zip(["light", "fast", "heavy"], results):
                    self.active[tier] = res if isinstance(res, dict) else None
                self._last_refresh_time = time.time()
        logger.info(f"[LLMSelector] ✅ Refresh complete. Active: {self.get_status()}")

    # -------------------------------------------------------------------------
    # Public get_* methods
    # -------------------------------------------------------------------------
    async def _get_tier(self, tier: str) -> Tuple[Any, str]:
        await self.ensure_ready()
        now = time.time()
        self._ensure_async_primitives()
        if self._async_lock is None: raise RuntimeError("Async primitives not ready")
        
        async with self._async_lock:
            entry = self.active.get(tier)
            stack_snapshot = list(self.stacks.get(tier, []))
            blacklist = dict(self._blacklisted_models)

        # 1. Check current active model
        if entry and entry.get("active_model"):
            m_id = entry["active_model"]
            if blacklist.get(m_id, 0) <= now:
                client, label = self._build_client_dispatch(m_id)
                if client:
                    logger.info(f"[LLMSelector] 🟢 DISPATCH tier={tier} model={m_id} {label}")
                    return client, m_id
            else:
                logger.warning(f"[LLMSelector] 🚫 Cached model {m_id} is blacklisted. Forcing fallback.")
                async with self._async_lock:
                    if self.active.get(tier) and self.active[tier].get("active_model") == m_id:
                        self.active[tier] = None

        # 2. Fallback: walk the stack
        # If intelligence ranking exists for heavy, sort stack by rank before falling back
        if tier == "heavy" and self.model_ranks:
            stack_snapshot.sort(key=lambda x: self.model_ranks.get(x["model"], 999))

        logger.warning(f"[LLMSelector] ⚠️ Finding healthy fallback for tier={tier}...")
        for e in stack_snapshot:
            if blacklist.get(e["model"], 0) > now:
                continue
            client, label = self._build_client_dispatch(e["model"])
            if client:
                async with self._async_lock:
                    # Update cache with fallback info
                    rank = self.model_ranks.get(e["model"], 999)
                    self.active[tier] = {**e, "active_client": client, "active_model": e["model"], "rank": rank}
                logger.warning(f"[LLMSelector] ⚠️ Fallback active: tier={tier} model={e['model']} {label}")
                return client, e["model"]
        
        raise RuntimeError(f"[LLMSelector] No available healthy models for tier '{tier}'")

    async def get_light(self) -> Tuple[Any, str]:
        return await self._get_tier("light")

    async def get_fast(self) -> Tuple[Any, str]:
        return await self._get_tier("fast")

    async def get_heavy(self) -> Tuple[Any, str]:
        return await self._get_tier("heavy")

    # -------------------------------------------------------------------------
    # Blacklist management
    # -------------------------------------------------------------------------
    def report_failure(self, model_id: str, duration: int = BLACKLIST_DURATION):
        now = time.time()
        with self._key_lock:
            self._blacklisted_models[model_id] = now + duration
            # Cleanup expired entries
            self._blacklisted_models = {k: v for k, v in self._blacklisted_models.items() if v > now}
        
        # Critical: Clear from active cache immediately
        for tier in ["light", "fast", "heavy"]:
            if self.active[tier] and self.active[tier].get("active_model") == model_id:
                self.active[tier] = None
        
        logger.warning(f"[LLMSelector] 🚫 Model {model_id} blacklisted for {duration}s. Cache cleared.")

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------
    def get_status(self) -> Dict[str, str]:
        return {
            t: (self.active[t]["active_model"] if self.active.get(t) else "OFFLINE")
            for t in ["light", "fast", "heavy"]
        }

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("[LLMSelector] 🔌 HTTP session closed.")