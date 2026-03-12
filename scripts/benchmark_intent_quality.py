"""
benchmark_intent_quality.py v2 — Тест КАЧЕСТВА intent extraction по light моделям
================================================================================
Тестирует напрямую dialog_manager.analyze_intent() для каждой light модели.
Проверяет: action, category, brand, price_limit, troll detection, fuzzy mapping.

Запуск:
  python scripts/benchmark_intent_quality.py
  python scripts/benchmark_intent_quality.py --verbose
"""

import asyncio
import argparse
import time
import sys
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel import UkrSellKernel

# ─────────────────────────────────────────────────────────────────────────────
# КАНДИДАТЫ
# ─────────────────────────────────────────────────────────────────────────────

LIGHT_CANDIDATES = [
    {"provider": "groq",     "model": "llama-3.1-8b-instant"},
    {"provider": "groq",     "model": "llama-3.2-3b-preview"},
    {"provider": "groq",     "model": "llama-3.2-1b-preview"},
    {"provider": "groq",     "model": "gemma2-9b-it"},
    {"provider": "cerebras", "model": "llama3.1-8b"},
    {"provider": "gemini",   "model": "gemini-2.0-flash-lite"},
    {"provider": "gemini",   "model": "gemini-2.0-flash"},
    {"provider": "openrouter", "model": "google/gemini-2.5-flash"},
    {"provider": "deepinfra",  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct"},
    {"provider": "deepinfra",  "model": "google/gemini-2.5-flash"},
    {"provider": "deepinfra",  "model": "qwen/qwen3.5-flash-02-23"},
    {"provider": "deepinfra",  "model": "google/gemini-3-flash-preview"},
    {"provider": "deepinfra",  "model": "z-ai/glm-4.7-flash"},
    {"provider": "openrouter", "model": "qwen/qwen3.5-flash-02-23"},
    {"provider": "openrouter", "model": "anthropic/claude-haiku-4.5"},
]

# ─────────────────────────────────────────────────────────────────────────────
# INTENT TEST CASES
# Каждый кейс проверяет конкретную способность модели
# ─────────────────────────────────────────────────────────────────────────────

INTENT_CASES = [
    # ── TROLL DETECTION ──────────────────────────────────────────────────────
    {
        "id": "TR-01", "group": "troll",
        "desc": "Образливий відгук",
        "query": "Ваш магазин — повне лайно, все дороге і некрасиве",
        "expect": {"action": "TROLL"},
    },
    {
        "id": "TR-02", "group": "troll",
        "desc": "Мат та агресія",
        "query": "Йдіть нафіг зі своїм магазином",
        "expect": {"action": "TROLL"},
    },
    {
        "id": "TR-03", "group": "troll",
        "desc": "Мусорний запит",
        "query": "аааааааааа ффффф 123123 !!!???###",
        "expect": {"action": "CHAT"},
    },

    # ── CATEGORY EXTRACTION ──────────────────────────────────────────────────
    {
        "id": "CA-01", "group": "category",
        "desc": "Пряма категорія укр",
        "query": "Шукаю куртку для собаки",
        "expect": {"action": "SEARCH", "category_contains": "куртк"},
    },
    {
        "id": "CA-02", "group": "category",
        "desc": "Пряма категорія рос",
        "query": "Хочу тёплую курточку для собаки на зиму до 800 грн",
        "expect": {"action": "SEARCH", "category_contains": "куртк", "price_limit": 800},
    },
    {
        "id": "CA-03", "group": "category",
        "desc": "Нашийник укр",
        "query": "Нашийник для лабрадора",
        "expect": {"action": "SEARCH", "category_contains": "нашийник"},
    },
    {
        "id": "CA-04", "group": "category",
        "desc": "Питальна форма → SEARCH",
        "query": "Є куртки для собак?",
        "expect": {"action": "SEARCH", "category_contains": "куртк"},
    },

    # ── FUZZY MAPPING ────────────────────────────────────────────────────────
    {
        "id": "FZ-01", "group": "fuzzy",
        "desc": "Мерзне → куртка/одяг",
        "query": "Мій пухнастий друг постійно мерзне на прогулянці",
        "expect": {"action": "SEARCH", "category_not_null": True},
    },
    {
        "id": "FZ-02", "group": "fuzzy",
        "desc": "Рве лапи → взуття",
        "query": "Собака рве лапи взимку на снігу",
        "expect": {"action": "SEARCH", "category_not_null": True},
    },
    {
        "id": "FZ-03", "group": "fuzzy",
        "desc": "Хвостатий малюк → одяг/аксесуари",
        "query": "Щось гарне для мого хвостатого малюка",
        "expect": {"action": "SEARCH", "category_not_null": True},
    },

    # ── BRAND EXTRACTION ─────────────────────────────────────────────────────
    {
        "id": "BR-01", "group": "brand",
        "desc": "Бренд Waudog",
        "query": "Цікавить бренд Waudog",
        "expect": {"action": "SEARCH", "brand_contains": "waudog"},
    },
    {
        "id": "BR-02", "group": "brand",
        "desc": "Порода НЕ бренд",
        "query": "Нашийник для лабрадора",
        "expect": {"brand_null": True},
    },

    # ── PRICE EXTRACTION ─────────────────────────────────────────────────────
    {
        "id": "PR-01", "group": "price",
        "desc": "Ціна до 500 грн",
        "query": "Хочу щось тепле, до 500 грн",
        "expect": {"action": "SEARCH", "price_limit": 500},
    },
    {
        "id": "PR-02", "group": "price",
        "desc": "Ціна до 800 грн рос",
        "query": "Куртка до 800 рублей",
        "expect": {"price_limit": 800},
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# ОЦІНКА
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_intent(intent: dict, expect: dict) -> tuple[bool, str, str]:
    """Возвращает (passed, fail_reason, intent_summary)."""
    entities = intent.get("entities", {})
    action   = intent.get("action", "")
    category = str(entities.get("category") or "").lower()
    brand    = str(entities.get("brand") or "").lower()
    price    = entities.get("price_limit")

    summary = f"action={action} cat={category or 'null'} brand={brand or 'null'} price={price}"
    fails   = []

    if "action" in expect:
        if action != expect["action"]:
            fails.append(f"action={action!r} want={expect['action']!r}")

    if "category_contains" in expect:
        if expect["category_contains"] not in category:
            fails.append(f"category={category!r} want contains {expect['category_contains']!r}")

    if "category_not_null" in expect and expect["category_not_null"]:
        if not category or category == "none":
            fails.append(f"category is null/none")

    if "brand_contains" in expect:
        if expect["brand_contains"] not in brand:
            fails.append(f"brand={brand!r} want contains {expect['brand_contains']!r}")

    if "brand_null" in expect and expect["brand_null"]:
        raw_brand = entities.get("brand")
        if raw_brand and str(raw_brand).lower() not in ("null", "none", ""):
            fails.append(f"brand should be null, got {raw_brand!r}")

    if "price_limit" in expect:
        if price != expect["price_limit"]:
            fails.append(f"price={price} want={expect['price_limit']}")

    passed = len(fails) == 0
    return passed, " | ".join(fails), summary

# ─────────────────────────────────────────────────────────────────────────────
# ЗАПУСК ДЛЯ ОДНОЙ МОДЕЛИ
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntentCaseResult:
    case_id: str
    group: str
    desc: str
    passed: bool
    latency_ms: float
    intent_summary: str
    fail_reason: str = ""

@dataclass
class IntentModelResult:
    model: str
    provider: str
    cases: List[IntentCaseResult] = field(default_factory=list)
    error: str = ""

    @property
    def passed(self): return sum(1 for c in self.cases if c.passed)
    @property
    def total(self): return len(self.cases)
    @property
    def score(self): return f"{self.passed}/{self.total}"
    @property
    def avg_latency(self):
        if not self.cases: return 0
        return sum(c.latency_ms for c in self.cases) / len(self.cases)

    def by_group(self, group: str):
        cases = [c for c in self.cases if c.group == group]
        passed = sum(1 for c in cases if c.passed)
        return f"{passed}/{len(cases)}"


async def run_model_intent(kernel, slug: str, candidate: dict,
                           verbose: bool) -> IntentModelResult:
    model    = candidate["model"]
    provider = candidate["provider"]
    result   = IntentModelResult(model=model, provider=provider)

    selector = kernel.selector
    ctx = kernel.registry.get_context(slug)
    if not ctx:
        result.error = f"store {slug} not found"
        return result

    dm = ctx.dialog_manager

    # Форсируем light модель
    try:
        client, resolved = await _force_light(selector, candidate)
        if client is None:
            result.error = "model not available"
            return result
    except Exception as e:
        result.error = str(e)[:80]
        return result

    original_light = selector.active.get("light")
    selector.active["light"] = {
        "active_client": client, "active_model": resolved,
        "type": provider, "provider": provider,
        "model": resolved
    }

    try:
        for case in INTENT_CASES:
            chat_id = f"iq_{model}_{case['id']}".replace("/", "_")
            t0 = time.time()
            try:
                intent = await dm.analyze_intent(case["query"], chat_id)
                latency_ms = (time.time() - t0) * 1000
                passed, fail_reason, summary = evaluate_intent(intent, case["expect"])
                result.cases.append(IntentCaseResult(
                    case_id=case["id"], group=case["group"], desc=case["desc"],
                    passed=passed, latency_ms=latency_ms,
                    intent_summary=summary, fail_reason=fail_reason,
                ))
                if verbose:
                    icon = "✅" if passed else "❌"
                    print(f"     {icon} [{case['id']}] {case['desc']}")
                    print(f"          → {summary}")
                    if fail_reason:
                        print(f"          ✗ {fail_reason}")
            except Exception as e:
                result.cases.append(IntentCaseResult(
                    case_id=case["id"], group=case["group"], desc=case["desc"],
                    passed=False, latency_ms=(time.time() - t0) * 1000,
                    intent_summary="", fail_reason=f"exception: {e}",
                ))
    finally:
        selector.active["light"] = original_light

    return result


async def _force_light(selector, candidate: dict):
    provider = candidate["provider"]
    model    = candidate["model"]
    clients  = selector._clients

    if provider == "groq":
        return clients.get("groq"), model
    if provider == "cerebras":
        return clients.get("cerebras"), model
    if provider in ("gemini", "gemini_direct"):
        c = clients.get("gemini_direct") or clients.get("gemini")
        return c, model
    if provider == "openrouter":
        return clients.get("openrouter"), model
    if provider == "deepinfra":
        return clients.get("deepinfra"), model
    return None, model

# ─────────────────────────────────────────────────────────────────────────────
# ВЫВОД
# ─────────────────────────────────────────────────────────────────────────────

GROUPS = ["troll", "category", "fuzzy", "brand", "price"]

def print_report(results: List[IntentModelResult]):
    print("\n" + "═" * 95)
    print("  BENCHMARK INTENT QUALITY — UkrSell v4")
    print("═" * 95)

    header = f"{'MODEL':<35} {'PROV':<10} {'SCORE':>6} {'ms/q':>6}"
    for g in GROUPS:
        header += f"  {g[:5].upper():>5}"
    print(header)
    print("─" * 95)

    sorted_r = sorted(
        [r for r in results if not r.error],
        key=lambda r: (-r.passed, r.avg_latency)
    )

    for r in sorted_r:
        pct = r.passed / r.total if r.total else 0
        marker = "✅" if pct == 1.0 else ("⚠️ " if pct >= 0.7 else "❌")
        row = f"{marker} {r.model:<32} {r.provider:<10} {r.score:>6} {r.avg_latency:>5.0f}ms"
        for g in GROUPS:
            row += f"  {r.by_group(g):>5}"
        print(row)

    for r in results:
        if r.error:
            print(f"💀 {r.model:<32} {r.provider:<10}  ERROR: {r.error}")

    # Детали по лучшей
    if sorted_r:
        best = sorted_r[0]
        print(f"\n{'─'*95}")
        print(f"  🏆 ЛУЧШАЯ: {best.model} ({best.provider}) — {best.score} | {best.avg_latency:.0f}ms/q")
        failed = [c for c in best.cases if not c.passed]
        if failed:
            print("  Провалы:")
            for c in failed:
                print(f"    [{c.case_id}] {c.desc}")
                print(f"          got:  {c.intent_summary}")
                print(f"          fail: {c.fail_reason}")

    # Детали провалов по всем моделям
    print(f"\n{'─'*95}")
    print("  ДЕТАЛИ ПРОВАЛОВ:")
    case_ids = [c["id"] for c in INTENT_CASES]
    for case_id in case_ids:
        case_meta = next(c for c in INTENT_CASES if c["id"] == case_id)
        row = f"  [{case_id}] {case_meta['desc'][:35]:<35}"
        for r in sorted_r:
            cr = next((c for c in r.cases if c.case_id == case_id), None)
            if cr:
                row += f"  {'✅' if cr.passed else '❌'}"
            else:
                row += "   ?"
        print(row)

    print("═" * 95)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug",    default="luckydog")
    parser.add_argument("--models",  default="all")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    candidates = LIGHT_CANDIDATES
    if args.models != "all":
        filter_vals = [v.strip().lower() for v in args.models.split(",")]
        candidates = [c for c in candidates
                      if c["provider"].lower() in filter_vals
                      or c["model"].lower() in filter_vals]

    print(f"🔬 Intent Quality Benchmark: {len(candidates)} моделей × {len(INTENT_CASES)} кейсов")
    print(f"   Store: {args.slug}\n")

    kernel = UkrSellKernel()
    await kernel.initialize()

    selector = kernel.selector
    if not hasattr(selector, '_clients'):
        print("❌ selector._clients недоступен")
        await kernel.close()
        return

    results = []
    for i, candidate in enumerate(candidates, 1):
        label = f"{candidate['provider']}/{candidate['model']}"
        print(f"[{i}/{len(candidates)}] {label}", flush=True)
        if args.verbose:
            print()
        t0 = time.time()
        r = await run_model_intent(kernel, args.slug, candidate, args.verbose)
        elapsed = time.time() - t0
        if r.error:
            print(f"  ❌ {r.error}")
        else:
            print(f"  → {r.score} за {elapsed:.1f}s | {r.avg_latency:.0f}ms/q")
        results.append(r)
        if args.verbose:
            print()

    await kernel.close()
    print_report(results)

    # Сохраняем JSON
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "benchmark_intent_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "slug": args.slug,
            "results": [
                {
                    "model": r.model, "provider": r.provider,
                    "score": r.score, "passed": r.passed, "total": r.total,
                    "avg_latency_ms": round(r.avg_latency, 1),
                    "error": r.error,
                    "by_group": {g: r.by_group(g) for g in GROUPS},
                    "cases": [
                        {"id": c.case_id, "group": c.group, "desc": c.desc,
                         "passed": c.passed, "latency_ms": round(c.latency_ms, 1),
                         "intent": c.intent_summary, "fail": c.fail_reason}
                        for c in r.cases
                    ]
                }
                for r in results
            ]
        }, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Сохранено: benchmark_intent_results.json")

if __name__ == "__main__":
    asyncio.run(main())