"""
benchmark_light_models.py — Перебор light-моделей для intent extraction
========================================================================
Прогоняет все 8 стресс-кейсов для каждой light-модели и выводит
сравнительную таблицу: accuracy / latency / troll detection / fuzzy.

Запуск:
  python scripts/benchmark_light_models.py
  python scripts/benchmark_light_models.py --slug luckydog
  python scripts/benchmark_light_models.py --models groq  # только groq
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
# КАНДИДАТЫ — light модели для перебора
# ─────────────────────────────────────────────────────────────────────────────

LIGHT_CANDIDATES = [
    # GROQ — быстрые
    {"provider": "groq",     "model": "llama-3.1-8b-instant"},
    {"provider": "groq",     "model": "llama-3.2-3b-preview"},
    {"provider": "groq",     "model": "llama-3.2-1b-preview"},
    {"provider": "groq",     "model": "gemma2-9b-it"},
    # CEREBRAS — очень быстрые
    {"provider": "cerebras", "model": "llama3.1-8b"},
    # GEMINI — fallback
    {"provider": "gemini",   "model": "gemini-2.0-flash-lite"},
    {"provider": "gemini",   "model": "gemini-2.0-flash"},
]

# ─────────────────────────────────────────────────────────────────────────────
# ТЕСТ-КЕЙСЫ (копия из stress_test, упрощённая)
# ─────────────────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "id": "FU-01", "group": "followup",
        "desc": "Одяг для лабрадора до 500 грн",
        "turns": ["Шукаю одяг для собаки", "У мене лабрадор, великий", "Хочу щось тепле, до 500 грн"],
        "expect": {"has_products": True, "keywords_out": ["немає", "вибачте", "не знайшли"]},
    },
    {
        "id": "FU-02", "group": "followup",
        "desc": "Куртки → Waudog",
        "turns": ["Є куртки для собак?", "Цікавить бренд Waudog"],
        "expect": {"has_products": True, "keywords_in": ["waudog", "Waudog"]},
    },
    {
        "id": "TR-01", "group": "troll",
        "desc": "Образливий відгук",
        "turns": ["Ваш магазин — повне лайно, все дороге і некрасиве"],
        "expect": {"has_products": False, "keywords_out": ["помилка", "error"]},
    },
    {
        "id": "TR-02", "group": "troll",
        "desc": "Мусорний запит",
        "turns": ["аааааааааа ффффф 123123 !!!???###"],
        "expect": {"has_products": False},
    },
    {
        "id": "SE-01", "group": "semantic",
        "desc": "Нашийник для лабрадора",
        "turns": ["Нашийник для лабрадора"],
        "expect": {"has_products": True, "keywords_out": ["немає", "не знайшли"]},
    },
    {
        "id": "SE-02", "group": "semantic",
        "desc": "Тепла курточка до 800 грн (рос.)",
        "turns": ["Хочу тёплую курточку для собаки на зиму до 800 грн"],
        "expect": {"has_products": True, "keywords_out": ["немає", "не знайшли"]},
    },
    {
        "id": "FZ-01", "group": "fuzzy",
        "desc": "Пухнастий друг мерзне",
        "turns": ["Мій пухнастий друг постійно мерзне на прогулянці"],
        "expect": {"has_products": True, "keywords_out": ["немає", "не знайшли"]},
    },
    {
        "id": "FZ-02", "group": "fuzzy",
        "desc": "Собака рве лапи",
        "turns": ["Собака рве лапи взимку на снігу"],
        "expect": {"has_products": True, "keywords_out": ["немає", "не знайшли"]},
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# РЕЗУЛЬТАТ
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    case_id: str
    group: str
    passed: bool
    latency_ms: float
    response_preview: str
    fail_reason: str = ""

@dataclass
class ModelResult:
    model: str
    provider: str
    cases: List[CaseResult] = field(default_factory=list)
    total_ms: float = 0.0
    error: str = ""

    @property
    def passed(self):
        return sum(1 for c in self.cases if c.passed)

    @property
    def total(self):
        return len(self.cases)

    @property
    def score(self):
        return f"{self.passed}/{self.total}"

    @property
    def avg_latency(self):
        if not self.cases:
            return 0
        return self.total_ms / len(self.cases)

    def by_group(self, group: str):
        cases = [c for c in self.cases if c.group == group]
        passed = sum(1 for c in cases if c.passed)
        return f"{passed}/{len(cases)}"

# ─────────────────────────────────────────────────────────────────────────────
# ЛОГИКА ОЦЕНКИ
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(response: str, expect: dict) -> tuple[bool, str]:
    """Возвращает (passed, fail_reason)."""
    r_lower = response.lower()

    has_products = expect.get("has_products")
    if has_products is True:
        # Признаки наличия товаров: ссылки или названия
        product_signals = ["http", "грн", "<b>", "prom.ua", "₴"]
        # Признаки отсутствия товаров
        no_product_signals = ["немає", "не знайшли", "нажаль", "на жаль",
                              "не містить", "відсутні", "немає інформації",
                              "не має", "немає товар"]
        found_product = any(s in r_lower for s in product_signals)
        says_no = any(s in r_lower for s in no_product_signals)
        if says_no or not found_product:
            return False, f"no_products (signals={found_product}, says_no={says_no})"

    if has_products is False:
        # Тролль/мусор — не должен возвращать список товаров как результат поиска
        product_signals = ["prom.ua/ua/p"]
        if any(s in r_lower for s in product_signals):
            return False, "unexpected_products"

    keywords_in = expect.get("keywords_in", [])
    if keywords_in:
        if not any(k.lower() in r_lower for k in keywords_in):
            return False, f"missing_keywords_in={keywords_in}"

    keywords_out = expect.get("keywords_out", [])
    for k in keywords_out:
        if k.lower() in r_lower:
            return False, f"bad_keyword_found='{k}'"

    return True, ""

# ─────────────────────────────────────────────────────────────────────────────
# ЗАПУСК ОДНОЙ МОДЕЛИ
# ─────────────────────────────────────────────────────────────────────────────

async def run_model(kernel, slug: str, candidate: dict) -> ModelResult:
    model_name = candidate["model"]
    provider   = candidate["provider"]
    result     = ModelResult(model=model_name, provider=provider)

    # Форсируем light модель в selector
    selector = kernel.selector
    try:
        # Находим клиент для этой модели
        client, resolved_model = await _force_light(selector, candidate)
        if client is None:
            result.error = f"model not available"
            return result
    except Exception as e:
        result.error = str(e)[:80]
        return result

    # Патчим active["light"]
    original_light = selector.active.get("light")
    selector.active["light"] = {"client": client, "model": resolved_model,
                                  "type": provider, "provider": provider}

    try:
        for case in TEST_CASES:
            chat_id = f"bench_{model_name}_{case['id']}".replace("/", "_")
            t0 = time.time()
            try:
                response = ""
                for turn in case["turns"]:
                    ctx = kernel.registry.get_context(slug)
                    response = await kernel.get_recommendations(
                        ctx=ctx,
                        query=turn,
                        user_id=chat_id,
                    )
                latency_ms = (time.time() - t0) * 1000
                passed, fail_reason = evaluate(str(response), case["expect"])
                result.cases.append(CaseResult(
                    case_id=case["id"],
                    group=case["group"],
                    passed=passed,
                    latency_ms=latency_ms,
                    response_preview=str(response)[:120],
                    fail_reason=fail_reason,
                ))
                result.total_ms += latency_ms
            except Exception as e:
                result.cases.append(CaseResult(
                    case_id=case["id"],
                    group=case["group"],
                    passed=False,
                    latency_ms=(time.time() - t0) * 1000,
                    response_preview="",
                    fail_reason=f"exception: {e}",
                ))
    finally:
        # Восстанавливаем оригинальную light модель
        selector.active["light"] = original_light

    return result


async def _force_light(selector, candidate: dict):
    """Возвращает (client, model) для кандидата."""
    provider = candidate["provider"]
    model    = candidate["model"]

    if provider == "groq":
        client = selector._clients.get("groq")
        return client, model

    if provider == "cerebras":
        client = selector._clients.get("cerebras")
        return client, model

    if provider == "gemini":
        # Gemini через openai-compatible
        client = selector._clients.get("gemini_direct") or selector._clients.get("gemini")
        return client, model

    if provider == "openrouter":
        client = selector._clients.get("openrouter")
        return client, model

    return None, model

# ─────────────────────────────────────────────────────────────────────────────
# ВЫВОД ТАБЛИЦЫ
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: List[ModelResult]):
    GROUPS = ["followup", "troll", "semantic", "fuzzy"]

    print("\n" + "═" * 90)
    print("  BENCHMARK LIGHT MODELS — UkrSell v4")
    print("═" * 90)

    # Заголовок
    header = f"{'MODEL':<35} {'PROV':<10} {'SCORE':>6} {'ms/req':>7}"
    for g in GROUPS:
        header += f"  {g[:6].upper():>6}"
    print(header)
    print("─" * 90)

    # Сортируем: сначала по score, потом по latency
    sorted_results = sorted(
        [r for r in results if not r.error],
        key=lambda r: (-r.passed, r.avg_latency)
    )
    errored = [r for r in results if r.error]

    for r in sorted_results:
        marker = "✅" if r.passed == r.total else ("⚠️ " if r.passed >= r.total // 2 else "❌")
        row = f"{marker} {r.model:<32} {r.provider:<10} {r.score:>6} {r.avg_latency:>6.0f}ms"
        for g in GROUPS:
            row += f"  {r.by_group(g):>6}"
        print(row)

    for r in errored:
        print(f"💀 {r.model:<32} {r.provider:<10}  ERROR: {r.error}")

    # Детали провалов лучшей модели
    if sorted_results:
        best = sorted_results[0]
        print(f"\n{'─'*90}")
        print(f"  🏆 ЛУЧШАЯ: {best.model} ({best.provider}) — {best.score} | {best.avg_latency:.0f}ms/req")
        failed = [c for c in best.cases if not c.passed]
        if failed:
            print(f"  Провалы:")
            for c in failed:
                print(f"    [{c.case_id}] {c.fail_reason}")
                print(f"          Preview: {c.response_preview[:100]}")
        else:
            print(f"  Все тесты пройдены! 🎉")

    # Сравнительная таблица по группам
    print(f"\n{'─'*90}")
    print("  ДЕТАЛИ ПО ГРУППАМ:")
    for g in GROUPS:
        print(f"\n  {g.upper()}:")
        for r in sorted_results:
            cases = [c for c in r.cases if c.group == g]
            icons = "".join("✅" if c.passed else "❌" for c in cases)
            avg = sum(c.latency_ms for c in cases) / len(cases) if cases else 0
            print(f"    {r.model:<35} {icons}  {avg:.0f}ms")

    print("═" * 90)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Benchmark light models for intent extraction")
    parser.add_argument("--slug",    default="luckydog")
    parser.add_argument("--models",  default="all",
                        help="all | groq | cerebras | gemini | comma-separated model names")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Фильтрация кандидатов
    candidates = LIGHT_CANDIDATES
    if args.models != "all":
        filter_vals = [v.strip().lower() for v in args.models.split(",")]
        candidates = [
            c for c in candidates
            if c["provider"].lower() in filter_vals
            or c["model"].lower() in filter_vals
        ]

    print(f"🔬 Benchmark: {len(candidates)} light моделей × {len(TEST_CASES)} кейсов")
    print(f"   Store: {args.slug}")
    print(f"   Кандидаты: {[c['model'] for c in candidates]}\n")

    # Инициализация kernel
    kernel = UkrSellKernel()
    await kernel.initialize()

    # Проверяем что _clients доступен
    selector = kernel.selector
    if not hasattr(selector, '_clients'):
        print("❌ selector._clients недоступен — проверь версию llm_selector.py")
        await kernel.close()
        return

    results: List[ModelResult] = []
    for i, candidate in enumerate(candidates, 1):
        model_display = f"{candidate['provider']}/{candidate['model']}"
        print(f"[{i}/{len(candidates)}] Тестирую: {model_display} ...", flush=True)
        t0 = time.time()
        result = await run_model(kernel, args.slug, candidate)
        elapsed = time.time() - t0
        if result.error:
            print(f"  ❌ Ошибка: {result.error}")
        else:
            print(f"  ✅ {result.score} за {elapsed:.1f}s | avg {result.avg_latency:.0f}ms/req")
            if args.verbose:
                for c in result.cases:
                    icon = "✅" if c.passed else "❌"
                    print(f"     {icon} [{c.case_id}] {c.latency_ms:.0f}ms — {c.fail_reason or 'ok'}")
        results.append(result)

    await kernel.close()

    print_report(results)

    # Сохраняем JSON с результатами
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "benchmark_light_results.json")
    out_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "slug": args.slug,
        "results": [
            {
                "model": r.model,
                "provider": r.provider,
                "score": r.score,
                "passed": r.passed,
                "total": r.total,
                "avg_latency_ms": round(r.avg_latency, 1),
                "error": r.error,
                "cases": [
                    {"id": c.case_id, "group": c.group, "passed": c.passed,
                     "latency_ms": round(c.latency_ms, 1), "fail_reason": c.fail_reason}
                    for c in r.cases
                ]
            }
            for r in results
        ]
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Результаты сохранены: benchmark_light_results.json")


if __name__ == "__main__":
    asyncio.run(main())