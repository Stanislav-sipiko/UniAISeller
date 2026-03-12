# /root/ukrsell_v4/core/llm_selector.py v7.5.1
import time
import asyncio
import threading
import openai
import aiohttp
from typing import Tuple, Any, Optional, Dict, List
from core.logger import logger, log_event
from core.config import GROQ_KEYS

# --- Groq-only tier ranking with fallbacks ---
TIER_RANKING = {
    "light": [
        "llama-3.1-8b-instant",
        "groq/compound",
        "moonshotai/kimi-k2-instruct",
    ],
    "fast": [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b",
        "qwen/qwen3-32b",
    ],
    "heavy": [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "moonshotai/kimi-k2-instruct",
        "moonshotai/kimi-k2-instruct-0905",
        "openai/gpt-oss-120b",
        "groq/compound",
    ],
}

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


class LLMSelector:
    """
    LLM Selector v7.5.1
    Groq-only. Static tier ranking. No LLM-based model classification.
    Changes vs v7.5.0:
      - light tier has fallbacks: groq/compound, moonshotai/kimi-k2-instruct
      - latency-aware selection: fastest healthy model wins within tier
      - prewarm() method for startup pre-heating of light tier
      - improved log levels: INFO=healthy, WARNING=fallback/blacklist
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

    # -------------------------------------------------------------------------
    # Lazy async primitives
    # -------------------------------------------------------------------------
    def _ensure_async_primitives(self):
        loop = asyncio.get_running_loop()
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
            log_event("LLM_CLIENTS_INIT", {"status": "success", "provider": "groq", "ver": "7.5.1"})

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
    # Build client — two modes:
    #   _build_client_ping()     — fixed key #0, no rotation (for health checks)
    #   _build_client_dispatch() — rotated key (for real requests only)
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
    # Static stack build from TIER_RANKING
    # -------------------------------------------------------------------------
    def _build_stacks(self):
        new_stacks: Dict[str, List[Dict]] = {"light": [], "fast": [], "heavy": []}
        for tier, models in TIER_RANKING.items():
            for model in models:
                if _is_llm_model(model):
                    new_stacks[tier].append({"type": "groq", "model": model})
        self.stacks = new_stacks
        logger.info(
            f"[LLMSelector] Stacks built: "
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

    async def _find_fastest_healthy(self, stack: List[Dict]) -> Optional[Dict]:
        """
        Ping all models in stack concurrently, return the fastest healthy one.
        Blacklisted models are skipped.
        """
        now = time.time()
        async with self._async_lock:
            blacklist_snapshot = dict(self._blacklisted_models)

        candidates = []
        for entry in stack:
            if blacklist_snapshot.get(entry["model"], 0) > now:
                logger.warning(f"[LLMSelector] ⚠️ {entry['model']} blacklisted, skipping.")
                continue
            candidates.append(entry)

        if not candidates:
            logger.warning(f"[LLMSelector] ⚠️ All models in stack are blacklisted.")
            return None

        async def ping_entry(entry: Dict) -> Optional[Dict]:
            model_id = entry["model"]
            ping_client = self._build_client_ping(model_id)
            if not ping_client:
                return None
            ok, lat = await self._test_ping(ping_client, model_id)
            if ok:
                logger.info(f"[LLMSelector] ✅ Healthy: {model_id} latency={lat}ms")
                return {**entry, "active_client": None, "active_model": model_id, "latency_ms": lat}
            else:
                self.report_failure(model_id, BLACKLIST_DURATION)
                logger.warning(f"[LLMSelector] ⚠️ Unhealthy: {model_id} → blacklisted {BLACKLIST_DURATION}s")
                return None

        results = await asyncio.gather(*[ping_entry(e) for e in candidates], return_exceptions=True)

        # Filter healthy results and pick the fastest
        healthy = [r for r in results if isinstance(r, dict)]
        if not healthy:
            logger.warning(f"[LLMSelector] ⚠️ No healthy models found in stack.")
            return None

        fastest = min(healthy, key=lambda x: x.get("latency_ms", 9999))
        logger.info(
            f"[LLMSelector] 🏆 Fastest selected: {fastest['active_model']} "
            f"latency={fastest['latency_ms']}ms "
            f"(from {len(healthy)} healthy)"
        )
        return fastest

    # -------------------------------------------------------------------------
    # Prewarm — call at service startup to pre-heat light tier
    # -------------------------------------------------------------------------
    async def prewarm(self):
        """Pre-heat light tier at startup so first request is fast."""
        await self._init_clients_async()
        self._build_stacks()
        logger.info("[LLMSelector] 🔥 Prewarming light tier...")
        result = await self._find_fastest_healthy(self.stacks["light"])
        async with self._async_lock:
            self.active["light"] = result
        if result:
            logger.info(f"[LLMSelector] ✅ Prewarm done: light={result['active_model']} latency={result['latency_ms']}ms")
        else:
            logger.warning("[LLMSelector] ⚠️ Prewarm failed: no healthy light model found.")

    # -------------------------------------------------------------------------
    # Refresh
    # -------------------------------------------------------------------------
    async def ensure_ready(self):
        self._ensure_async_primitives()
        await self._init_clients_async()

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
        logger.info("[LLMSelector] 🧪 Health-checking groq stacks (latency-aware)...")

        results = await asyncio.gather(
            self._find_fastest_healthy(self.stacks["light"]),
            self._find_fastest_healthy(self.stacks["fast"]),
            self._find_fastest_healthy(self.stacks["heavy"]),
            return_exceptions=True,
        )

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

        async with self._async_lock:
            entry = self.active.get(tier)
            stack_snapshot = list(self.stacks.get(tier, []))

        if entry and entry.get("active_model"):
            client, label = self._build_client_dispatch(entry["active_model"])
            if client:
                logger.info(f"[LLMSelector] 🟢 DISPATCH tier={tier} model={entry['active_model']} {label}")
                return client, entry["active_model"]

        # Fallback: walk the stack sequentially
        logger.warning(f"[LLMSelector] ⚠️ Active entry missing for tier={tier}, walking stack...")
        now = time.time()
        async with self._async_lock:
            blacklist = dict(self._blacklisted_models)

        for e in stack_snapshot:
            if blacklist.get(e["model"], 0) > now:
                logger.warning(f"[LLMSelector] ⚠️ Fallback skip blacklisted: {e['model']}")
                continue
            client, _ = self._build_client_dispatch(e["model"])
            if not client:
                continue
            async with self._async_lock:
                self.active[tier] = {**e, "active_client": client, "active_model": e["model"]}
            logger.warning(f"[LLMSelector] ⚠️ Fallback active: tier={tier} model={e['model']}")
            return client, e["model"]

        raise RuntimeError(f"[LLMSelector] No available model for tier '{tier}'")

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
            self._blacklisted_models = {
                k: v for k, v in self._blacklisted_models.items() if v > now
            }
        logger.warning(f"[LLMSelector] 🚫 Model {model_id} blacklisted for {duration}s")

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
