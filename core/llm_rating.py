# /root/ukrsell_v4/core/llm_rating.py   v1.4
"""
LLM Rating System для UkrSell v4
==================================

Три компонента:
  1. LLMRating      — хранит рейтинг каждой модели в llm_rating.json
  2. KeywordAdvisor — раз в день спрашивает LLM какие keywords → light/heavy
  3. make_rating_sorter — функция-замена _sort_by_tier для LLMSelector

Формула smart_score (умность 85%, скорость 15%):
  quality     = correct_calls / total_calls
  speed_bonus = min(1.0, 1500 / p50_latency_ms)
  smart_score = quality * 0.85 + speed_bonus * 0.15

Структура llm_rating.json:
{
  "version": "1.0",
  "keyword_hints": {
    "light":   ["8b", "mini", ...],
    "heavy":   ["72b", "pro", ...],
    "updated": "2026-03-10"
  },
  "models": {
    "llama-3.1-8b-instant": {
      "provider":       "groq",
      "tier_hint":      "light",
      "total_calls":    142,
      "correct_calls":  98,
      "wrong_calls":    44,
      "avg_latency_ms": 7200.0,
      "p50_latency_ms": 6800.0,
      "p95_latency_ms": 9100.0,
      "quality_score":  0.6901,
      "smart_score":    0.6166,
      "call_types":     {"intent": 120, "synthesis": 22},
      "last_seen":      "2026-03-10",
      "last_updated":   "2026-03-10T13:55:00"
    }
  }
}
"""

import json
import asyncio
import time
import statistics
import re
from pathlib import Path
from datetime import datetime, date
from typing import Optional

try:
    from core.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger("llm_rating")

try:
    from core.config import OPENROUTER_KEYS, GROQ_KEYS, GEMINI_KEYS, BASE_DIR_PATH
except ImportError:
    BASE_DIR_PATH = Path(__file__).resolve().parent.parent
    OPENROUTER_KEYS = []
    GROQ_KEYS = []
    GEMINI_KEYS = []

RATING_FILE    = BASE_DIR_PATH / "llm_rating.json"
SPEED_WEIGHT   = 0.15           # доля скорости в smart_score
FAST_BASELINE_MS = 1500.0       # p50 <= этого значения → speed_bonus = 1.0
MIN_CALLS_RELIABLE = 5          # минимум вызовов для надёжного рейтинга

DEFAULT_KEYWORDS = {
    "light": ["8b", "flash-lite", "small", "haiku", "mini", "nano", "3b", "1b", "2b"],
    "heavy": [
        "30b", "34b", "40b", "56b", "65b", "70b", "72b", "90b",
        "120b", "123b", "180b", "235b", "405b",
        "pro", "ultra", "large", "max", "plus", "opus", "super",
    ],
}


# ═════════════════════════════════════════════════════════════════════════════
#  LLMRating
# ═════════════════════════════════════════════════════════════════════════════

class LLMRating:
    """
    Потокобезопасное хранилище рейтингов моделей.

    Использование:
        rating = LLMRating()

        # После каждого LLM-вызова:
        await rating.record(
            model_id="llama-3.1-8b-instant",
            provider="groq",
            tier_hint="light",
            latency_ms=320.5,
            correct=True,          # правильный action / показаны товары
            call_type="intent",    # "intent" | "synthesis" | "troll" | "test"
        )

        # В стресс-тесте:
        await rating.record(..., correct=result.passed, call_type="test")
    """

    def __init__(self, path: Path = RATING_FILE):
        self.path = path
        self._data: dict = {"version": "1.0", "keyword_hints": {}, "models": {}}
        self._lock = asyncio.Lock()
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
                logger.info(f"[LLMRating] Loaded {len(self._data.get('models', {}))} models")
            except Exception as e:
                logger.warning(f"[LLMRating] Load failed: {e} — starting fresh")

    def _save(self):
        """Атомарная запись через временный файл."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)  # гарантируем что директория существует
            tmp = self.path.with_suffix(".tmp")
            # _latencies — внутренний кэш, не сохраняем в JSON
            clean = json.loads(json.dumps(self._data))
            for rec in clean.get("models", {}).values():
                rec.pop("_latencies", None)
            tmp.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except Exception as e:
            logger.error(f"[LLMRating] Save failed: {e}")

    # ── Запись результата ─────────────────────────────────────────────────────

    async def record(
        self,
        model_id:  str,
        provider:  str,
        tier_hint: str,
        latency_ms: float,
        correct:   bool,
        call_type: str = "intent",
    ):
        async with self._lock:
            models = self._data.setdefault("models", {})

            if model_id not in models:
                models[model_id] = {
                    "provider":       provider,
                    "tier_hint":      tier_hint,
                    "total_calls":    0,
                    "correct_calls":  0,
                    "wrong_calls":    0,
                    "_latencies":     [],
                    "avg_latency_ms": 0.0,
                    "p50_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "quality_score":  0.5,
                    "smart_score":    0.5,
                    "call_types":     {},
                    "last_seen":      str(date.today()),
                    "last_updated":   datetime.now().isoformat(timespec="seconds"),
                }

            rec = models[model_id]
            rec["total_calls"] += 1
            rec["correct_calls"] += int(correct)
            rec["wrong_calls"]   += int(not correct)

            # Скользящее окно latency (200 вызовов)
            lats = rec.setdefault("_latencies", [])
            lats.append(float(latency_ms))
            if len(lats) > 200:
                lats.pop(0)
            sl = sorted(lats)
            rec["avg_latency_ms"] = round(statistics.mean(lats), 1)
            rec["p50_latency_ms"] = round(statistics.median(lats), 1)
            rec["p95_latency_ms"] = round(sl[max(0, int(len(sl) * 0.95) - 1)], 1)

            rec["quality_score"] = round(rec["correct_calls"] / rec["total_calls"], 4)
            rec["smart_score"]   = self._calc_smart(rec)
            rec["call_types"].setdefault(call_type, 0)
            rec["call_types"][call_type] += 1
            rec["last_seen"]    = str(date.today())
            rec["last_updated"] = datetime.now().isoformat(timespec="seconds")
            rec["provider"]     = provider
            rec["tier_hint"]    = tier_hint

            self._save()

        logger.debug(
            f"[LLMRating] {'✅' if correct else '❌'} {model_id} "
            f"| {latency_ms:.0f}ms | smart={rec['smart_score']:.3f} "
            f"| q={rec['quality_score']:.1%} ({rec['total_calls']} calls)"
        )

    @staticmethod
    def _calc_smart(rec: dict) -> float:
        total = rec.get("total_calls", 0)
        if not total:
            return 0.5
        quality     = rec["correct_calls"] / total
        p50         = rec.get("p50_latency_ms") or FAST_BASELINE_MS
        speed_bonus = min(1.0, FAST_BASELINE_MS / max(p50, 100.0))
        return round(quality * (1 - SPEED_WEIGHT) + speed_bonus * SPEED_WEIGHT, 4)

    # ── Чтение ────────────────────────────────────────────────────────────────

    def get_score(self, model_id: str) -> Optional[float]:
        """smart_score если >= MIN_CALLS_RELIABLE вызовов, иначе None."""
        rec = self._data.get("models", {}).get(model_id)
        if rec and rec.get("total_calls", 0) >= MIN_CALLS_RELIABLE:
            return rec["smart_score"]
        return None

    def get_sorted(self, tier: Optional[str] = None, min_calls: int = 0) -> list:
        """Список моделей по убыванию smart_score."""
        out = []
        for mid, rec in self._data.get("models", {}).items():
            if tier and rec.get("tier_hint") != tier:
                continue
            if rec.get("total_calls", 0) < min_calls:
                continue
            out.append({"model": mid, **rec})
        return sorted(out, key=lambda x: x.get("smart_score", 0), reverse=True)

    def get_model_tiers(self) -> dict:
        """Возвращает сохранённую классификацию {"updated": "...", "tiers": {...}}"""
        return self._data.get("model_tiers", {})

    def set_model_tiers(self, tiers: dict):
        """Сохраняет классификацию моделей по провайдерам."""
        self._data["model_tiers"] = {
            "updated": str(date.today()),
            "tiers":   tiers,
        }
        self._save()
        total = sum(
            len(v.get(t, []))
            for v in tiers.values()
            for t in ("light", "heavy", "fast")
        )
        logger.info(f"[LLMRating] Model tiers saved: {len(tiers)} providers, {total} models")

    def needs_classification(self) -> bool:
        """True если классификация устарела (> 1 дня) или отсутствует."""
        mt = self._data.get("model_tiers", {})
        return mt.get("updated", "") != str(date.today())

    # ── обратная совместимость ────────────────────────────────────────────────
    def get_keyword_hints(self) -> dict:
        return self._data.get("keyword_hints", {})

    def needs_keyword_refresh(self) -> bool:
        return self.needs_classification()

    # ── Лидерборд ─────────────────────────────────────────────────────────────

    def print_leaderboard(self, top_n: int = 25):
        rows = self.get_sorted(min_calls=3)[:top_n]
        if not rows:
            print("  [LLMRating] Нет данных (нужно ≥ 3 вызовов на модель)")
            return
        print(f"\n{'═'*72}")
        print(f"  LLM LEADERBOARD   умность {1-SPEED_WEIGHT:.0%} / скорость {SPEED_WEIGHT:.0%}")
        print(f"{'═'*72}")
        print(f"  {'Модель':<38} {'Tier':<7} {'Smart':>6} {'Qual':>6} {'p50мс':>7} {'Calls':>6}")
        print(f"  {'─'*68}")
        for m in rows:
            print(
                f"  {m['model'][:37]:<38} "
                f"{m.get('tier_hint','?'):<7} "
                f"{m.get('smart_score',0):>6.3f} "
                f"{m.get('quality_score',0):>6.1%} "
                f"{m.get('p50_latency_ms',0):>7.0f} "
                f"{m.get('total_calls',0):>6}"
            )
        print(f"{'═'*72}\n")


# ═════════════════════════════════════════════════════════════════════════════
#  KeywordAdvisor
# ═════════════════════════════════════════════════════════════════════════════

_PROMPT = """\
Ты эксперт по классификации LLM-моделей. Нужны keyword-паттерны для автоматического
разделения моделей на два класса по именам.

КЛАССЫ:
- light = маленькие (~1–9B параметров): быстрые, дешёвые, для простых задач
- heavy = большие умные (>30B) или известные флагманы (Pro, Ultra, Max…)
(fast = всё остальное — не нужен, определяем отдельно)

СПИСОК ВСЕХ МОДЕЛЕЙ:
{model_list}

ПРАВИЛА:
- keyword встречается ТОЛЬКО в именах нужного класса
- предпочитай числовые паттерны: "8b", "3b", "72b"
- light: "flash-lite", "nano", "mini", "small", "haiku" и числа ≤9b
- heavy: "pro", "ultra", "large", "max", "plus", "opus" И числовые ≥30b:
  обязательно включи: "30b", "34b", "40b", "56b", "65b", "70b", "72b", "90b", "120b", "235b", "405b"
- НЕ добавляй общие слова: "chat", "instruct", "v1", "latest", "turbo"
- light: 8–14 keywords, heavy: 15–22 keywords (числовые паттерны обязательны)

Ответь СТРОГО JSON (без markdown, без пояснений):
{{"light": ["kw1", "kw2", ...], "heavy": ["kw1", "kw2", ...]}}
"""


class KeywordAdvisor:
    """
    Раз в день обновляет keywords для light/heavy через qwen-2.5-72b-instruct.
    Вызывать из _discover_stacks() после сбора model_ids.
    """

    def __init__(self, rating: LLMRating):
        self.rating = rating

    async def maybe_refresh(self, all_model_ids: list) -> dict:
        """Возвращает актуальные {"light": [...], "heavy": [...]}."""
        if not self.rating.needs_keyword_refresh():
            hints = self.rating.get_keyword_hints()
            if hints.get("light") and hints.get("heavy"):
                logger.debug("[KeywordAdvisor] Keywords fresh")
                return hints

        logger.info(f"[KeywordAdvisor] Refreshing keywords ({len(all_model_ids)} models)…")
        try:
            hints = await self._ask_llm(all_model_ids)
            self.rating.set_keyword_hints(hints["light"], hints["heavy"])
            return hints
        except Exception as e:
            logger.warning(f"[KeywordAdvisor] Refresh failed: {e} — using fallback")

        current = self.rating.get_keyword_hints()
        return current if (current.get("light") and current.get("heavy")) else DEFAULT_KEYWORDS.copy()

    async def _ask_llm(self, model_ids: list) -> dict:
        import openai  # локальный импорт
        if not OPENROUTER_KEYS:
            raise ValueError("OPENROUTER_KEYS empty")

        unique = sorted(set(model_ids))[:300]
        prompt = _PROMPT.format(model_list="\n".join(f"- {m}" for m in unique))

        client = openai.AsyncOpenAI(
            api_key=OPENROUTER_KEYS[0],
            base_url="https://openrouter.ai/api/v1"
        )
        t0 = time.time()
        resp = await client.chat.completions.create(
            model="qwen/qwen-2.5-72b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400,
        )
        ms = (time.time() - t0) * 1000
        content = resp.choices[0].message.content.strip()
        logger.info(f"[KeywordAdvisor] Response {ms:.0f}ms: {content[:80]}…")

        m = re.search(r'\{[^{}]+\}', content, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON: {content[:200]}")
        result = json.loads(m.group(0))

        light = [str(k).lower().strip() for k in result.get("light", []) if k]
        heavy = [str(k).lower().strip() for k in result.get("heavy", []) if k]
        if len(light) < 3 or len(heavy) < 3:
            raise ValueError(f"Too few keywords light={light} heavy={heavy}")

        logger.info(f"[KeywordAdvisor] ✅ light={light[:5]} heavy={heavy[:5]}")
        return {"light": light, "heavy": heavy}


# ═════════════════════════════════════════════════════════════════════════════
#  make_rating_sorter — замена _sort_by_tier
# ═════════════════════════════════════════════════════════════════════════════

def make_rating_sorter(rating: LLMRating, tier_ranking: dict):
    """
    Фабрика функции сортировки стека моделей.
    Подставляется вместо self._sort_by_tier в LLMSelector.

    Приоритет:
      0 — надёжный рейтинг (≥ MIN_CALLS_RELIABLE) → по -smart_score
      1 — нет данных, но модель есть в TIER_RANKING хардкоде → по индексу
      2 — совсем новая модель → в конец (наберёт статистику через fallback)

    Пример подключения в LLMSelector.__init__:
        from core.llm_rating import LLMRating, KeywordAdvisor, make_rating_sorter
        self.rating   = LLMRating()
        self._advisor = KeywordAdvisor(self.rating)
        self._sort_by_tier = make_rating_sorter(self.rating, TIER_RANKING)
    """
    def sort_fn(stack: list, tier: str) -> list:
        ranking = tier_ranking.get(tier, [])

        def key(entry: dict) -> tuple:
            mid   = entry["model"]
            score = rating.get_score(mid)   # None если нет надёжных данных

            if score is not None:
                return (0, -score)          # группа 0: лучшие вверх

            ml = mid.lower()
            for i, target in enumerate(ranking):
                if target in ml:
                    return (1, float(i))    # группа 1: по TIER_RANKING

            return (2, 0.0)                 # группа 2: новые в конец

        return sorted(stack, key=key)

    return sort_fn


# ═════════════════════════════════════════════════════════════════════════════
#  get_current_keywords — хелпер для _discover_stacks
# ═════════════════════════════════════════════════════════════════════════════

# Эти keywords ВСЕГДА в heavy — нельзя перебить через KeywordAdvisor.
# Reasoning-модели и флагманы без числового признака размера.
# 70b НЕ здесь — он в FAST_PRIORITY в llm_selector.py (llama-70b = fast).
ALWAYS_HEAVY_KEYWORDS = [
    "deepseek-r1",   # DeepSeek R1 reasoning (без distill)
    "o1", "o3",      # OpenAI reasoning серии
    "72b",           # qwen-2.5-72b, llama-2-72b — реально тяжёлые
    "671b", "236b",  # DeepSeek MoE полные версии
]

def get_current_keywords(rating: LLMRating, fast_keywords: list) -> dict:
    """
    Актуальные keywords для всех трёх тиров.
    light/heavy — из llm_rating.json (или DEFAULT_KEYWORDS при первом запуске).
    fast — всегда передаётся явно (управляется вручную).
    ALWAYS_HEAVY_KEYWORDS — добавляются к heavy всегда, KeywordAdvisor не может убрать.
    """
    hints = rating.get_keyword_hints()
    heavy = list(hints.get("heavy") or DEFAULT_KEYWORDS["heavy"])
    for kw in ALWAYS_HEAVY_KEYWORDS:
        if kw not in heavy:
            heavy.append(kw)
    return {
        "light": hints.get("light") or DEFAULT_KEYWORDS["light"],
        "fast":  fast_keywords,
        "heavy": heavy,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  CLI — python core/llm_rating.py [команда]
# ═════════════════════════════════════════════════════════════════════════════

def _cli():
    import sys, asyncio

    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    rating = LLMRating()

    # ── status: активные модели в каждом тире ────────────────────────────────
    if cmd == "status":
        print("\n📡 АКТИВНЫЕ МОДЕЛИ (из llm_selector через selector.active):")
        print("   Запусти: python core/llm_rating.py stacks\n")

        hints = rating.get_keyword_hints()
        print(f"🔑 KEYWORDS (обновлены: {hints.get('updated', 'никогда')})")
        print(f"   light : {hints.get('light', DEFAULT_KEYWORDS['light'])}")
        print(f"   heavy : {hints.get('heavy', DEFAULT_KEYWORDS['heavy'])}")

        models = rating._data.get("models", {})
        if not models:
            print("\n⚠️  llm_rating.json пуст — данных ещё нет.")
            print("   После первых LLM-вызовов появится статистика.")
        else:
            rating.print_leaderboard()

    # ── top: топ моделей по тиру ─────────────────────────────────────────────
    elif cmd == "top":
        tier = sys.argv[2] if len(sys.argv) > 2 else None
        label = f"тир={tier}" if tier else "все тиры"
        rows = rating.get_sorted(tier=tier, min_calls=1)
        if not rows:
            print(f"  Нет данных для {label}")
            return
        print(f"\n🏆 ТОП МОДЕЛЕЙ ({label}):\n")
        print(f"  {'Модель':<42} {'Smart':>6} {'Qual':>6} {'p50мс':>7} {'Calls':>6} {'Провайдер'}")
        print(f"  {'─'*75}")
        for m in rows[:15]:
            print(
                f"  {m['model'][:41]:<42} "
                f"{m.get('smart_score',0):>6.3f} "
                f"{m.get('quality_score',0):>6.1%} "
                f"{m.get('p50_latency_ms',0):>7.0f} "
                f"{m.get('total_calls',0):>6}  "
                f"{m.get('provider','?')}"
            )

    # ── stacks: что сейчас в стеках selector ─────────────────────────────────
    elif cmd == "stacks":
        print("\n📦 СТЕКИ МОДЕЛЕЙ В SELECTOR:\n")
        try:
            import sys as _sys
            import os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from core.llm_selector import LLMSelector

            # Модели которые НЕ являются LLM — фильтруем из отображения
            _NON_LLM = {"whisper", "tts", "dall-e", "stable-diffusion",
                        "embedding", "rerank", "transcription", "ocr"}

            def _is_llm(model_id: str) -> bool:
                ml = model_id.lower()
                return not any(x in ml for x in _NON_LLM)

            async def _show():
                sel = LLMSelector()
                try:
                    await sel.ensure_ready()

                    for tier in ["light", "fast", "heavy"]:
                        active = sel.active.get(tier)
                        stack  = [e for e in sel.stacks.get(tier, []) if _is_llm(e["model"])]
                        total  = len(sel.stacks.get(tier, []))
                        filtered = total - len(stack)
                        suffix = f", отфильтровано non-LLM: {filtered}" if filtered else ""
                        print(f"  ── {tier.upper()} ({len(stack)} LLM-моделей{suffix}) ────────────")
                        if active:
                            print(f"  ✅ ACTIVE : {active.get('active_model','?')} [{active.get('type','?')}] {active.get('latency_ms',0):.0f}мс")
                        else:
                            print(f"  ⚠️  ACTIVE : OFFLINE")

                        for i, e in enumerate(stack[:5]):
                            mid       = e["model"]
                            score     = rating.get_score(mid)
                            score_str = f"smart={score:.3f}" if score else "нет данных"
                            marker    = "▶" if (active and active.get("active_model") == mid) else " "
                            print(f"  {marker} {i+1}. {mid:<45} [{e['type']:<12}] {score_str}")
                        if len(stack) > 5:
                            print(f"       … ещё {len(stack)-5} LLM-моделей")
                        print()
                finally:
                    await sel.close()   # закрываем aiohttp сессию

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_show())
            finally:
                loop.run_until_complete(asyncio.sleep(0.1))  # даём httpx закрыть соединения
                loop.close()

        except Exception as e:
            print(f"  ❌ Не удалось загрузить LLMSelector: {e}")
            print("     Запусти из директории /root/ukrsell_v4/")

    # ── keywords: текущие keywords и история ─────────────────────────────────
    elif cmd == "keywords":
        hints = rating.get_keyword_hints()
        print(f"\n🔑 KEYWORDS ДЛЯ КЛАССИФИКАЦИИ МОДЕЛЕЙ")
        print(f"   Обновлены : {hints.get('updated', 'никогда')}")
        print(f"   light     : {hints.get('light', DEFAULT_KEYWORDS['light'])}")
        print(f"   heavy     : {hints.get('heavy', DEFAULT_KEYWORDS['heavy'])}")
        print(f"\n   (fast управляется вручную через FAST_KEYWORDS в llm_selector.py)")
        needs = rating.needs_keyword_refresh()
        print(f"\n   Нужно обновление сегодня: {'ДА' if needs else 'нет'}")

    # ── worst: худшие модели (кандидаты на блеклист) ─────────────────────────
    elif cmd == "worst":
        rows = rating.get_sorted(min_calls=5)
        worst = [r for r in rows if r.get("quality_score", 1) < 0.6]
        if not worst:
            print("\n✅ Моделей с quality < 60% нет (или < 5 вызовов)")
            return
        print(f"\n⚠️  СЛАБЫЕ МОДЕЛИ (quality < 60%, вызовов ≥ 5):\n")
        for m in worst:
            print(
                f"  {m['model'][:45]:<46} "
                f"smart={m.get('smart_score',0):.3f} "
                f"q={m.get('quality_score',0):.1%} "
                f"p50={m.get('p50_latency_ms',0):.0f}мс "
                f"({m.get('total_calls',0)} calls) [{m.get('provider','?')}]"
            )

    # ── help ─────────────────────────────────────────────────────────────────
    else:
        print("""
📖 llm_rating.py — CLI для мониторинга рейтинга моделей

КОМАНДЫ (из /root/ukrsell_v4/):
  python core/llm_rating.py                  → лидерборд + keywords
  python core/llm_rating.py status           → то же самое
  python core/llm_rating.py top              → топ всех моделей
  python core/llm_rating.py top light        → топ light-тира
  python core/llm_rating.py top heavy        → топ heavy-тира
  python core/llm_rating.py stacks           → активные модели в selector прямо сейчас
  python core/llm_rating.py keywords         → текущие keywords light/heavy
  python core/llm_rating.py worst            → модели с quality < 60%

ФАЙЛ ДАННЫХ:
  /root/ukrsell_v4/llm_rating.json           → рейтинг всех моделей

В ЛОГАХ (в продакшне):
  grep "LLMRating" logs/ukrsell.log          → все записи рейтинга
  grep "KeywordAdvisor" logs/ukrsell.log     → обновления keywords
""")


if __name__ == "__main__":
    _cli()# ═════════════════════════════════════════════════════════════════════════════
#  ModelClassifier — LLM классифицирует модели напрямую (раз в день)
# ═════════════════════════════════════════════════════════════════════════════

_CLASSIFY_PROMPT = """Ты эксперт по LLM-моделям. Тебе дан список моделей от одного провайдера.
Распредели каждую модель по трём категориям:

- light  = маленькие и быстрые (~1–9B параметров): для простых задач (intent, troll-detection)
- heavy  = большие умные модели (>72B) или reasoning-флагманы (o1, o3, deepseek-r1, claude-opus, kimi-k2)
- fast   = всё остальное: рабочие лошадки 10–72B

ВАЖНО:
- 70b модели (llama-3.3-70b и подобные) → fast, НЕ heavy (они быстрые рабочие лошадки)
- 72b модели (qwen-2.5-72b) → heavy (умные флагманы)
- distill-версии (deepseek-r1-distill-*) → fast, НЕ heavy
- compound, kimi-k2, gpt-oss-120b → heavy
- Модели типа guard, whisper, tts, embedding, safeguard — пропусти, не включай в ответ
- Если не знаешь модель — ставь fast

Провайдер: {provider}
Модели:
{model_list}

Ответь СТРОГО JSON без markdown:
{{"light": ["model1", ...], "heavy": ["model1", ...], "fast": ["model1", ...]}}
"""


class ModelClassifier:
    """
    Раз в день классифицирует ВСЕ модели от каждого провайдера напрямую через LLM.
    Результат: {"groq": {"light": [...], "heavy": [...], "fast": [...]}, ...}
    Сохраняется в llm_rating.json["model_tiers"].

    Преимущество перед keyword-подходом:
    - Нет ложных срабатываний (mini в gemini, flash в flash-lite)
    - LLM знает что такое kimi-k2, gpt-oss-120b, compound — и правильно их классифицирует
    - Новые модели классифицируются автоматически без правки кода
    """

    def __init__(self, rating: "LLMRating"):
        self.rating = rating

    async def maybe_classify(self, provider_map: dict) -> dict:
        """
        provider_map = {"groq": ["model1", ...], "openrouter": [...], ...}
        Возвращает {"groq": {"light": [], "heavy": [], "fast": []}, ...}
        """
        saved = self.rating.get_model_tiers()
        today = str(date.today())

        if saved.get("updated") == today and saved.get("tiers"):
            logger.debug("[ModelClassifier] Tiers fresh, skipping")
            return saved["tiers"]

        logger.info(f"[ModelClassifier] Classifying {sum(len(v) for v in provider_map.values())} models…")

        # Классифицируем каждого провайдера параллельно
        tasks = {
            provider: self._classify_provider(provider, models)
            for provider, models in provider_map.items()
            if models
        }
        results = {}
        for provider, coro in tasks.items():
            try:
                results[provider] = await coro
            except Exception as e:
                logger.warning(f"[ModelClassifier] {provider} failed: {e} — using fast fallback")
                results[provider] = {"light": [], "heavy": [], "fast": list(provider_map[provider])}

        self.rating.set_model_tiers(results)
        return results

    # Топ N моделей от каждого провайдера — провайдеры сортируют по новизне/популярности
    TOP_MODELS_PER_PROVIDER = {
        "openrouter": 25,   # топ-25: всё актуальное, после 20 уже малоизвестные
        "deepinfra":  30,   # 142 модели, но актуальных ~30
        "groq":       19,   # все (их всего 19)
        "cerebras":   10,   # все (их всего 2)
    }

    async def _classify_provider(self, provider: str, models: list) -> dict:
        # Фильтруем non-LLM
        _NON_LLM = ["whisper","tts","dall-e","embedding","rerank",
                    "transcription","ocr","guard","moderation","orpheus","bark",
                    "imagen","veo","audio","speech"]
        llm_models = [m for m in models if not any(p in m.lower() for p in _NON_LLM)]
        if not llm_models:
            return {"light": [], "heavy": [], "fast": []}

        # Берём только топ N — всё актуальное в начале списка
        limit = self.TOP_MODELS_PER_PROVIDER.get(provider, 50)
        llm_models = llm_models[:limit]
        logger.info(f"[ModelClassifier] {provider}: top {len(llm_models)} models")

        # Один вызов — 50 моделей легко влезают в ответ
        return await self._ask_llm(provider, llm_models)

    async def _ask_llm(self, provider: str, models: list) -> dict:
        import openai as _openai
        model_list = "\n".join(f"- {m}" for m in models)
        prompt = _CLASSIFY_PROMPT.format(provider=provider, model_list=model_list)

        # Приоритет провайдеров для классификации (от дешёвого к дорогому):
        # 1. Groq — llama-3.3-70b-versatile, бесплатно, быстро
        # 2. OpenRouter — qwen-2.5-72b-instruct, платно, запасной
        # Приоритет backends: Gemini (6 бесплатных ключей) → Groq → OpenRouter
        backends = []
        for idx, key in enumerate(GEMINI_KEYS, 1):
            backends.append({
                "client": _openai.AsyncOpenAI(
                    api_key=key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                ),
                "model": "models/gemini-2.5-flash",
                "name": f"gemini[{idx}]",
            })
        if GROQ_KEYS:
            backends.append({
                "client": _openai.AsyncOpenAI(
                    api_key=GROQ_KEYS[0],
                    base_url="https://api.groq.com/openai/v1"
                ),
                "model": "llama-3.3-70b-versatile",
                "name": "groq[1]",
            })
        for idx, key in enumerate(OPENROUTER_KEYS, 1):
            backends.append({
                "client": _openai.AsyncOpenAI(
                    api_key=key,
                    base_url="https://openrouter.ai/api/v1"
                ),
                "model": "qwen/qwen-2.5-72b-instruct",
                "name": f"openrouter[{idx}]",
            })
        if not backends:
            raise ValueError("No LLM backends available for classification")

        last_error = None
        for backend in backends:
            try:
                t0 = time.time()
                resp = await backend["client"].chat.completions.create(
                    model=backend["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=4000,
                )
                ms = (time.time() - t0) * 1000
                content = resp.choices[0].message.content.strip()
                logger.info(f"[ModelClassifier] {provider} via {backend['name']}: {len(models)} models → {ms:.0f}ms")
                break
            except Exception as e:
                # 402 = кончились кредиты на этом ключе → пробуем следующий
                code = getattr(e, "status_code", None)
                logger.warning(f"[ModelClassifier] {backend['name']} failed (code={code}): {e}")
                last_error = e
                continue
        else:
            raise last_error or RuntimeError("All backends exhausted")

        # Парсим JSON — убираем возможные markdown-блоки
        clean = re.sub(r"```[a-z]*", "", content).strip()
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON in response: {content[:200]}")
        result = json.loads(m.group(0))

        # Валидация — каждая модель должна быть из исходного списка
        valid = set(models)
        return {
            "light": [x for x in result.get("light", []) if x in valid],
            "heavy": [x for x in result.get("heavy", []) if x in valid],
            "fast":  [x for x in result.get("fast",  []) if x in valid],
        }


# ═════════════════════════════════════════════════════════════════════════════
#  get_model_tier — хелпер для _discover_stacks (заменяет get_current_keywords)
# ═════════════════════════════════════════════════════════════════════════════

def get_model_tier(model_id: str, provider: str, tiers: dict) -> str:
    """
    Возвращает тир модели из сохранённой классификации.
    tiers = {"groq": {"light": [...], "heavy": [...], "fast": [...]}, ...}
    Fallback: fast.
    """
    provider_tiers = tiers.get(provider, {})
    for tier in ("light", "heavy", "fast"):
        if model_id in provider_tiers.get(tier, []):
            return tier
    return "fast"  # незнакомая модель → fast


# ═════════════════════════════════════════════════════════════════════════════
#  CLI — python core/llm_rating.py [команда]
# ═════════════════════════════════════════════════════════════════════════════

def _cli():
    import sys, asyncio

    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    rating = LLMRating()

    # ── status: активные модели в каждом тире ────────────────────────────────
    if cmd == "status":
        print("\n📡 АКТИВНЫЕ МОДЕЛИ (из llm_selector через selector.active):")
        print("   Запусти: python core/llm_rating.py stacks\n")

        hints = rating.get_keyword_hints()
        print(f"🔑 KEYWORDS (обновлены: {hints.get('updated', 'никогда')})")
        print(f"   light : {hints.get('light', DEFAULT_KEYWORDS['light'])}")
        print(f"   heavy : {hints.get('heavy', DEFAULT_KEYWORDS['heavy'])}")

        models = rating._data.get("models", {})
        if not models:
            print("\n⚠️  llm_rating.json пуст — данных ещё нет.")
            print("   После первых LLM-вызовов появится статистика.")
        else:
            rating.print_leaderboard()

    # ── top: топ моделей по тиру ─────────────────────────────────────────────
    elif cmd == "top":
        tier = sys.argv[2] if len(sys.argv) > 2 else None
        label = f"тир={tier}" if tier else "все тиры"
        rows = rating.get_sorted(tier=tier, min_calls=1)
        if not rows:
            print(f"  Нет данных для {label}")
            return
        print(f"\n🏆 ТОП МОДЕЛЕЙ ({label}):\n")
        print(f"  {'Модель':<42} {'Smart':>6} {'Qual':>6} {'p50мс':>7} {'Calls':>6} {'Провайдер'}")
        print(f"  {'─'*75}")
        for m in rows[:15]:
            print(
                f"  {m['model'][:41]:<42} "
                f"{m.get('smart_score',0):>6.3f} "
                f"{m.get('quality_score',0):>6.1%} "
                f"{m.get('p50_latency_ms',0):>7.0f} "
                f"{m.get('total_calls',0):>6}  "
                f"{m.get('provider','?')}"
            )

    # ── stacks: что сейчас в стеках selector ─────────────────────────────────
    elif cmd == "stacks":
        print("\n📦 СТЕКИ МОДЕЛЕЙ В SELECTOR:\n")
        try:
            import sys as _sys
            import os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from core.llm_selector import LLMSelector

            # Модели которые НЕ являются LLM — фильтруем из отображения
            _NON_LLM = {"whisper", "tts", "dall-e", "stable-diffusion",
                        "embedding", "rerank", "transcription", "ocr"}

            def _is_llm(model_id: str) -> bool:
                ml = model_id.lower()
                return not any(x in ml for x in _NON_LLM)

            async def _show():
                sel = LLMSelector()
                try:
                    await sel.ensure_ready()

                    for tier in ["light", "fast", "heavy"]:
                        active = sel.active.get(tier)
                        stack  = [e for e in sel.stacks.get(tier, []) if _is_llm(e["model"])]
                        total  = len(sel.stacks.get(tier, []))
                        filtered = total - len(stack)
                        suffix = f", отфильтровано non-LLM: {filtered}" if filtered else ""
                        print(f"  ── {tier.upper()} ({len(stack)} LLM-моделей{suffix}) ────────────")
                        if active:
                            print(f"  ✅ ACTIVE : {active.get('active_model','?')} [{active.get('type','?')}] {active.get('latency_ms',0):.0f}мс")
                        else:
                            print(f"  ⚠️  ACTIVE : OFFLINE")

                        for i, e in enumerate(stack[:5]):
                            mid       = e["model"]
                            score     = rating.get_score(mid)
                            score_str = f"smart={score:.3f}" if score else "нет данных"
                            marker    = "▶" if (active and active.get("active_model") == mid) else " "
                            print(f"  {marker} {i+1}. {mid:<45} [{e['type']:<12}] {score_str}")
                        if len(stack) > 5:
                            print(f"       … ещё {len(stack)-5} LLM-моделей")
                        print()
                finally:
                    await sel.close()   # закрываем aiohttp сессию

            asyncio.run(_show())

        except Exception as e:
            print(f"  ❌ Не удалось загрузить LLMSelector: {e}")
            print("     Запусти из директории /root/ukrsell_v4/")

    # ── keywords: текущие keywords и история ─────────────────────────────────
    elif cmd == "keywords":
        hints = rating.get_keyword_hints()
        print(f"\n🔑 KEYWORDS ДЛЯ КЛАССИФИКАЦИИ МОДЕЛЕЙ")
        print(f"   Обновлены : {hints.get('updated', 'никогда')}")
        print(f"   light     : {hints.get('light', DEFAULT_KEYWORDS['light'])}")
        print(f"   heavy     : {hints.get('heavy', DEFAULT_KEYWORDS['heavy'])}")
        print(f"\n   (fast управляется вручную через FAST_KEYWORDS в llm_selector.py)")
        needs = rating.needs_keyword_refresh()
        print(f"\n   Нужно обновление сегодня: {'ДА' if needs else 'нет'}")

    # ── worst: худшие модели (кандидаты на блеклист) ─────────────────────────
    elif cmd == "worst":
        rows = rating.get_sorted(min_calls=5)
        worst = [r for r in rows if r.get("quality_score", 1) < 0.6]
        if not worst:
            print("\n✅ Моделей с quality < 60% нет (или < 5 вызовов)")
            return
        print(f"\n⚠️  СЛАБЫЕ МОДЕЛИ (quality < 60%, вызовов ≥ 5):\n")
        for m in worst:
            print(
                f"  {m['model'][:45]:<46} "
                f"smart={m.get('smart_score',0):.3f} "
                f"q={m.get('quality_score',0):.1%} "
                f"p50={m.get('p50_latency_ms',0):.0f}мс "
                f"({m.get('total_calls',0)} calls) [{m.get('provider','?')}]"
            )

    # ── help ─────────────────────────────────────────────────────────────────
    else:
        print("""
📖 llm_rating.py — CLI для мониторинга рейтинга моделей

КОМАНДЫ (из /root/ukrsell_v4/):
  python core/llm_rating.py                  → лидерборд + keywords
  python core/llm_rating.py status           → то же самое
  python core/llm_rating.py top              → топ всех моделей
  python core/llm_rating.py top light        → топ light-тира
  python core/llm_rating.py top heavy        → топ heavy-тира
  python core/llm_rating.py stacks           → активные модели в selector прямо сейчас
  python core/llm_rating.py keywords         → текущие keywords light/heavy
  python core/llm_rating.py worst            → модели с quality < 60%

ФАЙЛ ДАННЫХ:
  /root/ukrsell_v4/llm_rating.json           → рейтинг всех моделей

В ЛОГАХ (в продакшне):
  grep "LLMRating" logs/ukrsell.log          → все записи рейтинга
  grep "KeywordAdvisor" logs/ukrsell.log     → обновления keywords
""")


if __name__ == "__main__":
    _cli()