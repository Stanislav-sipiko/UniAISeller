# /root/ukrsell_v4/scripts/self_debug.py v2.0.0
"""
UkrSell Self-Debug Pipeline v2.0.0

Полный цикл автоматической диагностики бота:
  1. LOAD        — каталог как ground truth (normalized_products_final.json)
  2. GENERATE    — LLM генерирует тест-кейсы; каталог индексируется через embeddings (>80 SKU)
  3. RUN         — прогон: одиночные запросы + conversation chains (2–3 follow-up)
  4. EVALUATE    — LLM оценивает каждый ответ (pass/fail/hallucination/partial)
  5. METRICS     — hallucination_rate, coverage_rate, latency p50/p95, по типам запросов
  6. ANALYZE     — LLM группирует паттерны ошибок по компонентам
  7. SUGGEST     — LLM предлагает патчи промптов/кода
  8. SELF-CORRECT— применяет патч на тестовом ctx, повторный прогон провалов, сравнение
  9. REPORT      — JSON + HTML dashboard + консольный summary

Запуск:
  python scripts/self_debug.py [slug] [n_cases]
  python scripts/self_debug.py luckydog 20
  python scripts/self_debug.py luckydog 50 --no-selfcorrect

Зависимости: kernel.py, core/, stores/<slug>/normalized_products_final.json
"""

import asyncio
import sys
import os
import json
import time
import logging
import re
import math
import argparse
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)

from kernel import UkrSellKernel
from core.logger import logger

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_SLUG          = "luckydog"
DEFAULT_N_CASES       = 20
CONCURRENCY           = 3
CHAIN_LENGTH          = 3       # follow-up сообщений в цепочке
CHAINS_RATIO          = 0.3     # 30% кейсов — цепочки, 70% — одиночные
MAX_CATALOG_FOR_LLM   = 80      # при превышении — embeddings sampling
SELFCORRECT_MAX_ITER  = 2       # максимум итераций self-correction
SELFCORRECT_MIN_DELTA = 3.0     # минимальный прирост pass_rate % для подтверждения патча
REPORT_DIR            = ROOT_DIR / "logs"


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD CATALOG
# ═════════════════════════════════════════════════════════════════════════════

def load_catalog(slug: str) -> List[Dict]:
    path = ROOT_DIR / "stores" / slug / "normalized_products_final.json"
    if not path.exists():
        raise FileNotFoundError(f"Каталог не найден: {path}")
    with open(path, encoding="utf-8") as f:
        products = json.load(f)
    print(f"✅ Каталог загружен: {len(products)} товаров")
    return products


def sample_catalog_for_llm(
    products: List[Dict],
    kernel: UkrSellKernel,
    n: int = MAX_CATALOG_FOR_LLM,
) -> List[Dict]:
    """
    Если каталог > MAX_CATALOG_FOR_LLM — делаем репрезентативную выборку
    через embeddings: берём по n//categories_count из каждой категории.
    """
    if len(products) <= n:
        return products

    print(f"  📦 Каталог {len(products)} SKU > {n} — embedding-sampling по категориям...")
    from collections import defaultdict
    by_cat: Dict[str, List] = defaultdict(list)
    for p in products:
        by_cat[p.get("category", "unknown")].append(p)

    cats = list(by_cat.keys())
    per_cat = max(1, n // len(cats))
    sampled = []
    for cat in cats:
        items = by_cat[cat]
        # Берём равномерно по индексу — без LLM вызова
        step = max(1, len(items) // per_cat)
        sampled.extend(items[::step][:per_cat])

    result = sampled[:n]
    print(f"  ✅ Выборка: {len(result)} товаров из {len(cats)} категорий")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: GENERATE TEST CASES
# ═════════════════════════════════════════════════════════════════════════════

def _catalog_str(products: List[Dict]) -> str:
    return "\n".join([
        f"ID:{p['product_id']} | {p['title']} | cat:{p.get('category','')} | "
        f"animal:{p.get('animal',[])} | brand:{p.get('brand','')} | price:{p.get('price','')}"
        for p in products
    ])


async def generate_single_cases(
    client,
    model: str,
    catalog_sample: List[Dict],
    n: int,
    language: str,
) -> List[Dict]:
    """Генерирует n одиночных тест-кейсов."""
    lang_hint = "украинском" if language.lower() == "ukrainian" else "русском"
    prompt = f"""You are a QA engineer testing an e-commerce pet store chatbot.
Generate exactly {n} single-turn test cases in JSON from the catalog below.

Types to mix: direct, vague, animal, brand, price, negative.
- direct: clear product name query
- vague: broad category, no specifics
- animal: specify dog/cat
- brand: mention a brand
- price: mention price limit
- negative: ask for product NOT in catalog (set expected_id="NONE")

Write queries in conversational {lang_hint} language.
Return ONLY a JSON array. Schema per item:
{{"query":"...","expected_id":"...","expected_title":"...","query_type":"...","is_chain":false}}

CATALOG:
{_catalog_str(catalog_sample)}
"""
    return await _llm_json_array(client, model, prompt, max_tokens=4000, temperature=0.4)


async def generate_chain_cases(
    client,
    model: str,
    catalog_sample: List[Dict],
    n_chains: int,
    language: str,
    chain_length: int = CHAIN_LENGTH,
) -> List[Dict]:
    """
    Генерирует n_chains разговорных цепочек.
    Каждая цепочка — список из chain_length сообщений:
      turn_0: начальный запрос
      turn_1: follow-up (уточнение)
      turn_2: финальный выбор
    Возвращает плоский список кейсов с полем chain_id и turn_index.
    """
    lang_hint = "украинском" if language.lower() == "ukrainian" else "русском"
    prompt = f"""You are a QA engineer testing a chatbot. Generate {n_chains} conversation chains.
Each chain has exactly {chain_length} turns simulating a real customer.
Turn 0: broad/unclear request. Turn 1: follow-up narrowing. Turn 2: specific choice.

Write in conversational {lang_hint} language.
Return ONLY a JSON array of chains. Schema:
[
  {{
    "chain_id": "c1",
    "turns": [
      {{"turn": 0, "query": "...", "expected_id": "...", "expected_title": "...", "query_type": "vague"}},
      {{"turn": 1, "query": "...", "expected_id": "...", "expected_title": "...", "query_type": "direct"}},
      {{"turn": 2, "query": "...", "expected_id": "...", "expected_title": "...", "query_type": "direct"}}
    ]
  }}
]

CATALOG (use real products):
{_catalog_str(catalog_sample)}
"""
    raw_chains = await _llm_json_array(client, model, prompt, max_tokens=5000, temperature=0.4)

    # Разворачиваем цепочки в плоский список с is_chain=True
    flat = []
    for chain in raw_chains:
        cid = chain.get("chain_id", f"c{len(flat)}")
        turns = chain.get("turns", [])
        for t in turns:
            flat.append({
                "query":          t.get("query", ""),
                "expected_id":    t.get("expected_id", "NONE"),
                "expected_title": t.get("expected_title", "NONE"),
                "query_type":     t.get("query_type", "vague"),
                "is_chain":       True,
                "chain_id":       cid,
                "turn_index":     t.get("turn", 0),
            })
    return flat


async def generate_test_cases(
    client,
    model: str,
    products: List[Dict],
    n: int,
    language: str,
    kernel: UkrSellKernel,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Возвращает (single_cases, chain_cases).
    chain_cases — плоские, группируются по chain_id при прогоне.
    """
    catalog_sample = sample_catalog_for_llm(products, kernel)

    n_chains  = max(1, int(n * CHAINS_RATIO) // CHAIN_LENGTH)
    n_single  = n - n_chains * CHAIN_LENGTH

    print(f"⚙️  Генерация: {n_single} одиночных + {n_chains} цепочек×{CHAIN_LENGTH}...")
    t0 = time.perf_counter()

    single_task = generate_single_cases(client, model, catalog_sample, n_single, language)
    chain_task  = generate_chain_cases(client, model, catalog_sample, n_chains, language)
    singles, chains = await asyncio.gather(single_task, chain_task)

    dur = round((time.perf_counter() - t0) * 1000)
    print(f"✅ Сгенерировано: {len(singles)} одиночных + {len(chains)} chain-шагов за {dur}ms")
    return singles, chains


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: RUN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

async def run_single(kernel, ctx, case: Dict, idx: int, total: int) -> Dict:
    """Прогоняет один кейс без истории."""
    t0 = time.perf_counter()
    try:
        response = await kernel.get_recommendations(
            ctx=ctx, query=case["query"], filters=None,
            top_k=5, user_id=90000 + idx,
        )
        ms = round((time.perf_counter() - t0) * 1000)
        print(f"  [{idx}/{total}] ✓ {case['query'][:55]!r} → {ms}ms")
        return {**case, "bot_response": response, "latency_ms": ms, "error": None,
                "conversation_history": []}
    except Exception as e:
        ms = round((time.perf_counter() - t0) * 1000)
        return {**case, "bot_response": "", "latency_ms": ms, "error": str(e),
                "conversation_history": []}


async def run_chain(
    kernel,
    ctx,
    turns: List[Dict],
    chain_id: str,
    base_idx: int,
    total: int,
) -> List[Dict]:
    """
    Прогоняет цепочку turn_0 → turn_1 → turn_2.
    История диалога накапливается и передаётся в каждый следующий запрос
    через dialog_manager (user_id общий для всей цепочки).
    """
    user_id = 80000 + base_idx
    history: List[str] = []
    results = []

    for turn in sorted(turns, key=lambda t: t.get("turn_index", 0)):
        query = turn["query"]
        t0 = time.perf_counter()
        try:
            # Передаём накопленный контекст через filters (dialog_manager подхватит по user_id)
            response = await kernel.get_recommendations(
                ctx=ctx, query=query, filters=None,
                top_k=5, user_id=user_id,
            )
            ms = round((time.perf_counter() - t0) * 1000)
            history.append(f"U: {query}\nB: {response[:200]}")
            print(f"  [{base_idx}/{total}] ✓ chain={chain_id} turn={turn.get('turn_index')} "
                  f"{query[:45]!r} → {ms}ms")
            results.append({
                **turn,
                "bot_response":          response,
                "latency_ms":            ms,
                "error":                 None,
                "conversation_history":  history.copy(),
            })
        except Exception as e:
            ms = round((time.perf_counter() - t0) * 1000)
            results.append({
                **turn,
                "bot_response":         "",
                "latency_ms":           ms,
                "error":                str(e),
                "conversation_history": history.copy(),
            })
    return results


async def run_all(
    kernel,
    ctx,
    singles: List[Dict],
    chains_flat: List[Dict],
) -> List[Dict]:
    """Запускает одиночные и цепочки, возвращает объединённый список."""
    total = len(singles) + len(chains_flat)
    print(f"\n🔄 Прогон {total} шагов (concurrency={CONCURRENCY})...")
    semaphore = asyncio.Semaphore(CONCURRENCY)

    # Одиночные
    async def bounded_single(case, idx):
        async with semaphore:
            return await run_single(kernel, ctx, case, idx, total)

    single_tasks = [bounded_single(c, i + 1) for i, c in enumerate(singles)]

    # Цепочки — группируем по chain_id
    from collections import defaultdict
    by_chain: Dict[str, List] = defaultdict(list)
    for t in chains_flat:
        by_chain[t["chain_id"]].append(t)

    async def bounded_chain(cid, turns, idx):
        async with semaphore:
            return await run_chain(kernel, ctx, turns, cid, idx, total)

    chain_tasks = [
        bounded_chain(cid, turns, len(singles) + i + 1)
        for i, (cid, turns) in enumerate(by_chain.items())
    ]

    single_results_nested = await asyncio.gather(*single_tasks)
    chain_results_nested  = await asyncio.gather(*chain_tasks)

    results = list(single_results_nested)
    for chain_res in chain_results_nested:
        results.extend(chain_res)

    print(f"✅ Прогон завершён: {len(results)} результатов")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: EVALUATE
# ═════════════════════════════════════════════════════════════════════════════

async def evaluate_single(client, model: str, result: Dict) -> Dict:
    expected_id    = result.get("expected_id", "NONE")
    expected_title = result.get("expected_title", "NONE")
    bot_response   = result.get("bot_response", "")
    query          = result.get("query", "")
    query_type     = result.get("query_type", "")
    is_negative    = expected_id == "NONE"

    prompt = f"""QA evaluator for e-commerce chatbot.

USER QUERY: {query}
QUERY TYPE: {query_type}
EXPECTED PRODUCT ID: {expected_id}
EXPECTED PRODUCT TITLE: {expected_title}
IS NEGATIVE TEST: {is_negative}
BOT RESPONSE:
{bot_response[:1500]}

VERDICTS:
- pass: correct product found (title/ID present), no hallucinations
- fail: expected product absent, bot said not-found when it should exist
- hallucination: bot invented product/brand/category not matching expected
- partial: product found but with invented alternatives or wrong details
- negative_pass: (negative test) bot correctly said unavailable
- negative_fail: (negative test) bot invented the unavailable product

Return ONLY JSON:
{{"verdict":"...","reason":"one sentence","hallucinated_text":"exact invented text or null"}}
"""
    max_retries = 4
    for attempt in range(max_retries):
        try:
            r = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "Precise JSON evaluator."},
                          {"role": "user", "content": prompt}],
                max_tokens=300, temperature=0.0,
            )
            raw = _strip_md(r.choices[0].message.content)
            v = json.loads(raw)
            return {**result, **v}
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate_limit_exceeded" in err_str.lower()
            if is_rate_limit and attempt < max_retries - 1:
                wait_sec = 2 ** (attempt + 1)  # 2s, 4s, 8s
                print(f"  ⏳ evaluate_single: 429 rate limit, retry {attempt+1}/{max_retries-1} in {wait_sec}s...")
                await asyncio.sleep(wait_sec)
                continue
            return {**result, "verdict": "error", "reason": err_str, "hallucinated_text": None}
    return {**result, "verdict": "error", "reason": "max retries exceeded", "hallucinated_text": None}


async def evaluate_all(client, model: str, results: List[Dict]) -> List[Dict]:
    print(f"\n🔍 Оценка {len(results)} результатов...")
    semaphore = asyncio.Semaphore(2)  # снижено с 5 до 2 для предотвращения 429

    async def bounded(r):
        async with semaphore:
            return await evaluate_single(client, model, r)

    evaluated = await asyncio.gather(*[bounded(r) for r in results])
    verdicts = [e.get("verdict") for e in evaluated]
    counts = {v: verdicts.count(v) for v in set(verdicts)}
    print(f"✅ Оценка: {counts}")
    return list(evaluated)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(evaluated: List[Dict]) -> Dict:
    """
    Вычисляет:
      pass_rate, fail_rate, hallucination_rate, coverage_rate
      latency p50, p95
      breakdown по query_type
      breakdown по is_chain
    """
    total = len(evaluated)
    if not total:
        return {}

    pass_v   = {"pass", "negative_pass"}
    hall_v   = {"hallucination"}
    fail_v   = {"fail", "negative_fail"}

    passes   = [e for e in evaluated if e.get("verdict") in pass_v]
    halls    = [e for e in evaluated if e.get("verdict") in hall_v]
    fails    = [e for e in evaluated if e.get("verdict") in fail_v]
    partials = [e for e in evaluated if e.get("verdict") == "partial"]

    latencies = sorted([e.get("latency_ms", 0) for e in evaluated if e.get("latency_ms")])
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0

    # По типу запроса
    from collections import defaultdict
    by_type: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "pass": 0})
    for e in evaluated:
        qt = e.get("query_type", "unknown")
        by_type[qt]["total"] += 1
        if e.get("verdict") in pass_v:
            by_type[qt]["pass"] += 1

    type_rates = {
        qt: round(v["pass"] / v["total"] * 100, 1)
        for qt, v in by_type.items() if v["total"]
    }

    # Цепочки vs одиночные
    chains  = [e for e in evaluated if e.get("is_chain")]
    singles = [e for e in evaluated if not e.get("is_chain")]
    chain_pass_rate  = round(sum(1 for e in chains if e.get("verdict") in pass_v)
                             / len(chains) * 100, 1) if chains else None
    single_pass_rate = round(sum(1 for e in singles if e.get("verdict") in pass_v)
                             / len(singles) * 100, 1) if singles else None

    # coverage = доля каталога, которую покрывают успешные ответы
    found_ids = {e.get("expected_id") for e in passes if e.get("expected_id") != "NONE"}
    all_ids   = {e.get("expected_id") for e in evaluated if e.get("expected_id") != "NONE"}
    coverage_rate = round(len(found_ids) / len(all_ids) * 100, 1) if all_ids else 0.0

    return {
        "total":              total,
        "pass_rate":          round(len(passes)   / total * 100, 1),
        "fail_rate":          round(len(fails)     / total * 100, 1),
        "hallucination_rate": round(len(halls)     / total * 100, 1),
        "partial_rate":       round(len(partials)  / total * 100, 1),
        "coverage_rate":      coverage_rate,
        "latency_p50_ms":     p50,
        "latency_p95_ms":     p95,
        "pass_by_type":       type_rates,
        "chain_pass_rate":    chain_pass_rate,
        "single_pass_rate":   single_pass_rate,
        "counts": {
            "pass": len(passes), "fail": len(fails),
            "hallucination": len(halls), "partial": len(partials),
            "error": sum(1 for e in evaluated if e.get("verdict") == "error"),
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6+7: ANALYZE + SUGGEST
# ═════════════════════════════════════════════════════════════════════════════

async def analyze_and_suggest(
    client,
    model: str,
    evaluated: List[Dict],
    metrics: Dict,
    slug: str,
) -> Dict:
    failures = [e for e in evaluated
                if e.get("verdict") in ("fail", "hallucination", "partial", "negative_fail")]

    failures_summary = json.dumps([{
        "query":       f["query"],
        "type":        f.get("query_type"),
        "is_chain":    f.get("is_chain", False),
        "chain_turn":  f.get("turn_index"),
        "expected":    f.get("expected_title"),
        "verdict":     f.get("verdict"),
        "reason":      f.get("reason"),
        "hallucinated":f.get("hallucinated_text"),
        "response":    (f.get("bot_response") or "")[:300],
    } for f in failures[:30]], ensure_ascii=False, indent=2)

    metrics_str = json.dumps({
        k: v for k, v in metrics.items() if k != "counts"
    }, ensure_ascii=False)

    print(f"\n🧠 Анализ паттернов ({len(failures)} провалов)...")

    prompt = f"""Senior AI engineer debugging UkrSell v4 Telegram bot. Store: {slug}.
Metrics: {metrics_str}

FAILED CASES:
{failures_summary}

Provide structured analysis:
1. Root cause patterns (group similar failures, note if chain turns fail more)
2. Component responsible: confidence.py | analyzer.py | retrieval.py | category_map.json
3. Specific fix per pattern (concrete prompt text or code change, max 3 sentences)
4. Priority: high/medium/low by frequency and severity

Return ONLY JSON:
{{
  "patterns": [
    {{
      "name": "...",
      "count": N,
      "component": "file.py",
      "description": "...",
      "fix": "concrete fix text",
      "fix_type": "prompt|code|config",
      "priority": "high|medium|low",
      "affects_chains": true|false
    }}
  ],
  "top_recommendation": "single most impactful fix",
  "selfcorrect_patch": {{
    "component": "analyzer.py",
    "target": "function or prompt section name",
    "current_text": "text to replace (max 100 chars)",
    "new_text": "replacement text (max 200 chars)"
  }}
}}
"""
    try:
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "Precise JSON analyzer."},
                      {"role": "user", "content": prompt}],
            max_tokens=2500, temperature=0.2,
        )
        raw = _strip_md(r.choices[0].message.content)
        analysis = json.loads(raw)
        print(f"✅ Анализ: {len(analysis.get('patterns', []))} паттернов")
        return analysis
    except Exception as e:
        print(f"❌ Ошибка анализа: {e}")
        return {"error": str(e), "patterns": [], "top_recommendation": "", "selfcorrect_patch": None}


# ═════════════════════════════════════════════════════════════════════════════
# STEP 8: SELF-CORRECTION LOOP
# ═════════════════════════════════════════════════════════════════════════════

async def self_correct_loop(
    client,
    model_fast: str,
    model_heavy: str,
    kernel: UkrSellKernel,
    ctx,
    failures: List[Dict],
    analysis: Dict,
    slug: str,
    max_iter: int = SELFCORRECT_MAX_ITER,
    min_delta: float = SELFCORRECT_MIN_DELTA,
) -> Dict:
    """
    Self-correction loop:
      iter 1: применяет selfcorrect_patch из analysis к ctx.analyzer._build_prompt
      iter N: повторный прогон только провальных кейсов → оценка → сравнение pass_rate
      если delta >= min_delta → патч подтверждён
      иначе → LLM уточняет патч и повторяет (до max_iter)
    """
    if not failures:
        print("\n⏭️  Self-correction: нет провалов для повторного прогона.")
        return {"skipped": True, "reason": "no failures"}

    patch = analysis.get("selfcorrect_patch")
    if not patch:
        print("\n⏭️  Self-correction: анализ не предложил патч.")
        return {"skipped": True, "reason": "no patch suggested"}

    print(f"\n🔧 Self-Correction Loop (max_iter={max_iter}, min_delta={min_delta}%)")

    sc_history = []
    baseline_pass = sum(1 for f in failures if f.get("verdict") in ("pass", "negative_pass"))
    baseline_rate = round(baseline_pass / len(failures) * 100, 1)
    print(f"  Baseline pass rate на провалах: {baseline_rate}%")

    current_patch = patch

    for iteration in range(1, max_iter + 1):
        print(f"\n  🔁 Итерация {iteration}: применяем патч компонент={current_patch.get('component')}")
        print(f"     target: {current_patch.get('target')}")
        print(f"     current: {str(current_patch.get('current_text',''))[:80]}")
        print(f"     new:     {str(current_patch.get('new_text',''))[:80]}")

        # Применяем патч к store_context_prompt (безопасная точка инъекции)
        patch_applied = _apply_patch_to_ctx(ctx, current_patch)
        if not patch_applied:
            print(f"  ⚠️  Патч не применён (компонент не поддерживается в runtime)")
            sc_history.append({"iteration": iteration, "patch": current_patch,
                               "applied": False, "new_pass_rate": None})
            break

        # Повторный прогон только провальных кейсов
        re_results = await run_all(kernel, ctx, failures, [])
        re_evaluated = await evaluate_all(client, model_fast, re_results)

        new_pass = sum(1 for e in re_evaluated if e.get("verdict") in ("pass", "negative_pass"))
        new_rate = round(new_pass / len(re_evaluated) * 100, 1) if re_evaluated else 0
        delta = new_rate - baseline_rate

        print(f"  📊 pass_rate: {baseline_rate}% → {new_rate}% (Δ{delta:+.1f}%)")
        sc_history.append({
            "iteration":    iteration,
            "patch":        current_patch,
            "applied":      True,
            "baseline_rate":baseline_rate,
            "new_pass_rate":new_rate,
            "delta":        delta,
            "confirmed":    delta >= min_delta,
        })

        if delta >= min_delta:
            print(f"  ✅ Патч ПОДТВЕРЖДЁН (Δ{delta:+.1f}% ≥ {min_delta}%)")
            return {
                "confirmed":      True,
                "final_pass_rate":new_rate,
                "delta":          delta,
                "iterations":     sc_history,
                "final_patch":    current_patch,
            }

        # Патч не помог — просим уточнённый
        if iteration < max_iter:
            print(f"  ⚠️  Δ{delta:+.1f}% < {min_delta}% — запрашиваем уточнённый патч...")
            current_patch = await _request_refined_patch(
                client, model_heavy, current_patch, re_evaluated, delta, slug
            )
            if not current_patch:
                print("  ❌ LLM не вернул уточнённый патч. Стоп.")
                break
        else:
            print(f"  ❌ Исчерпаны итерации. Патч не подтверждён.")

        # Откатываем патч перед следующей итерацией
        _revert_patch_to_ctx(ctx, current_patch)

    return {
        "confirmed":   False,
        "delta":       sc_history[-1].get("delta") if sc_history else 0,
        "iterations":  sc_history,
        "final_patch": current_patch,
    }


def _apply_patch_to_ctx(ctx, patch: Dict) -> bool:
    """
    Безопасная runtime-инъекция патча.
    Поддерживаемые компоненты:
      analyzer.py → store_context_prompt (store_profile["store_context_prompt"])
    Остальные компоненты логируются — патч предлагается вручную.
    """
    component = patch.get("component", "")
    new_text  = patch.get("new_text", "")

    if "analyzer" in component:
        # Патчим store_context_prompt — он инжектится в каждый промпт
        if hasattr(ctx, "analyzer") and hasattr(ctx.analyzer, "store_context_prompt"):
            original = ctx.analyzer.store_context_prompt or ""
            current  = patch.get("current_text", "")
            if current and current in original:
                ctx.analyzer._sc_backup = original
                ctx.analyzer.store_context_prompt = original.replace(current, new_text, 1)
            else:
                # Append режим — добавляем в конец
                ctx.analyzer._sc_backup = original
                ctx.analyzer.store_context_prompt = original + "\n" + new_text
            logger.info(f"[SC] Патч применён → analyzer.store_context_prompt")
            return True

    logger.warning(f"[SC] Компонент {component!r} не поддерживается в runtime. Патч только для отчёта.")
    return False


def _revert_patch_to_ctx(ctx, patch: Dict):
    """Откатывает примённый патч."""
    if hasattr(ctx, "analyzer") and hasattr(ctx.analyzer, "_sc_backup"):
        ctx.analyzer.store_context_prompt = ctx.analyzer._sc_backup
        del ctx.analyzer._sc_backup
        logger.info("[SC] Патч откатан")


async def _request_refined_patch(
    client,
    model: str,
    prev_patch: Dict,
    re_evaluated: List[Dict],
    delta: float,
    slug: str,
) -> Optional[Dict]:
    """Запрашивает уточнённый патч после неудачной итерации."""
    remaining_fails = [e for e in re_evaluated
                       if e.get("verdict") not in ("pass", "negative_pass")]
    summary = json.dumps([{
        "query":   f["query"],
        "verdict": f["verdict"],
        "reason":  f.get("reason"),
    } for f in remaining_fails[:15]], ensure_ascii=False)

    prompt = f"""Previous self-correction patch for store {slug} improved pass_rate by only {delta:+.1f}%.
Previous patch: {json.dumps(prev_patch, ensure_ascii=False)}
Remaining failures: {summary}

Suggest a REFINED patch that addresses the remaining failures.
Return ONLY JSON with same schema:
{{"component":"...","target":"...","current_text":"...","new_text":"..."}}
"""
    try:
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":"Precise JSON patcher."},
                      {"role":"user","content":prompt}],
            max_tokens=500, temperature=0.2,
        )
        return json.loads(_strip_md(r.choices[0].message.content))
    except Exception as e:
        logger.error(f"[SC] refined patch error: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# STEP 9: REPORT (JSON + HTML)
# ═════════════════════════════════════════════════════════════════════════════

def save_report(
    slug: str,
    evaluated: List[Dict],
    metrics: Dict,
    analysis: Dict,
    sc_result: Dict,
    duration_sec: float,
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    REPORT_DIR.mkdir(exist_ok=True)
    json_path = REPORT_DIR / f"self_debug_{slug}_{ts}.json"
    html_path = REPORT_DIR / f"self_debug_{slug}_{ts}.html"

    report = {
        "meta":            {"slug": slug, "timestamp": ts,
                            "total_cases": len(evaluated), "duration_sec": round(duration_sec, 1)},
        "metrics":         metrics,
        "analysis":        analysis,
        "self_correction": sc_result,
        "cases":           evaluated,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _write_html_report(html_path, slug, ts, metrics, analysis, sc_result, evaluated)

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print(f"📊  SELF-DEBUG REPORT — {slug.upper()} — {ts}")
    print("═" * 62)
    c = metrics.get("counts", {})
    print(f"  Всего кейсов:       {metrics.get('total')}")
    print(f"  ✅ pass:            {c.get('pass')}  ({metrics.get('pass_rate')}%)")
    print(f"  ❌ fail:            {c.get('fail')}")
    print(f"  👻 hallucination:   {c.get('hallucination')}  ({metrics.get('hallucination_rate')}%)")
    print(f"  🟡 partial:         {c.get('partial')}")
    print(f"  📦 coverage:        {metrics.get('coverage_rate')}%")
    print(f"  ⏱  latency p50/p95: {metrics.get('latency_p50_ms')}ms / {metrics.get('latency_p95_ms')}ms")
    if metrics.get("chain_pass_rate") is not None:
        print(f"  🔗 chain pass:      {metrics.get('chain_pass_rate')}%")
    if metrics.get("single_pass_rate") is not None:
        print(f"  💬 single pass:     {metrics.get('single_pass_rate')}%")
    print(f"  ⏱  Общее время:     {round(duration_sec, 1)}s")
    print("─" * 62)

    for p in analysis.get("patterns", []):
        pri = p.get("priority", "?").upper()
        print(f"  [{pri}] {p.get('name')} ({p.get('count','?')}) → {p.get('component')}")
        print(f"        {p.get('description','')}")
        fix = p.get("fix", "")
        if fix:
            print(f"        FIX: {fix[:110]}{'...' if len(fix)>110 else ''}")

    top = analysis.get("top_recommendation")
    if top:
        print(f"\n🎯  TOP: {top}")

    sc = sc_result or {}
    if not sc.get("skipped"):
        confirmed = sc.get("confirmed", False)
        delta     = sc.get("delta", 0)
        print(f"\n🔧  Self-Correction: {'✅ ПОДТВЕРЖДЁН' if confirmed else '❌ не подтверждён'} "
              f"(Δ{delta:+.1f}%)")

    print("─" * 62)
    print(f"💾  JSON:  {json_path}")
    print(f"🌐  HTML:  {html_path}")
    print("═" * 62)
    return html_path


def _write_html_report(
    path: Path,
    slug: str,
    ts: str,
    metrics: Dict,
    analysis: Dict,
    sc_result: Dict,
    evaluated: List[Dict],
):
    """Генерирует HTML dashboard с Chart.js."""
    c = metrics.get("counts", {})
    patterns = analysis.get("patterns", [])

    # Данные для графиков
    verdict_labels  = json.dumps(["pass", "fail", "hallucination", "partial", "error"])
    verdict_data    = json.dumps([c.get("pass",0), c.get("fail",0),
                                  c.get("hallucination",0), c.get("partial",0), c.get("error",0)])
    type_labels     = json.dumps(list(metrics.get("pass_by_type", {}).keys()))
    type_data       = json.dumps(list(metrics.get("pass_by_type", {}).values()))

    # Latency histogram — из кейсов
    latencies   = [e.get("latency_ms", 0) for e in evaluated if e.get("latency_ms")]
    lat_buckets = [0, 500, 1000, 2000, 3000, 5000, 99999]
    lat_labels  = ["<500ms", "500-1s", "1-2s", "2-3s", "3-5s", ">5s"]
    lat_counts  = [0] * len(lat_labels)
    for ms in latencies:
        for i in range(len(lat_buckets) - 1):
            if lat_buckets[i] <= ms < lat_buckets[i + 1]:
                lat_counts[i] += 1
                break
    lat_labels_js = json.dumps(lat_labels)
    lat_data_js   = json.dumps(lat_counts)

    # Таблица кейсов
    rows_html = ""
    verdict_color = {
        "pass": "#22c55e", "negative_pass": "#22c55e",
        "fail": "#ef4444", "negative_fail": "#ef4444",
        "hallucination": "#f97316",
        "partial": "#eab308",
        "error": "#6b7280",
    }
    for e in evaluated:
        v     = e.get("verdict", "?")
        color = verdict_color.get(v, "#6b7280")
        chain = "🔗" if e.get("is_chain") else ""
        hall  = f'<span style="color:#f97316">{e.get("hallucinated_text","")[:60]}</span>' \
                if e.get("hallucinated_text") else ""
        rows_html += (
            f'<tr>'
            f'<td>{chain}{_esc(e.get("query","")[:60])}</td>'
            f'<td>{e.get("query_type","")}</td>'
            f'<td>{_esc(e.get("expected_title","")[:40])}</td>'
            f'<td style="color:{color};font-weight:bold">{v}</td>'
            f'<td>{e.get("latency_ms","")}ms</td>'
            f'<td>{_esc(e.get("reason","")[:80])}{hall}</td>'
            f'</tr>\n'
        )

    # Паттерны
    patterns_html = ""
    pri_color = {"high": "#ef4444", "medium": "#f97316", "low": "#eab308"}
    for p in patterns:
        pc = pri_color.get(p.get("priority",""), "#6b7280")
        patterns_html += f"""
        <div class="pattern">
          <span class="badge" style="background:{pc}">{p.get('priority','').upper()}</span>
          <strong>{_esc(p.get('name',''))}</strong>
          <span class="comp">→ {p.get('component','')}</span>
          ({p.get('count','?')} кейсов)<br>
          <em>{_esc(p.get('description',''))}</em><br>
          <code>{_esc(p.get('fix','')[:200])}</code>
        </div>"""

    sc  = sc_result or {}
    sc_html = ""
    if not sc.get("skipped"):
        sc_color = "#22c55e" if sc.get("confirmed") else "#ef4444"
        sc_html = f"""
        <div class="sc-block">
          <h3>🔧 Self-Correction</h3>
          <p>Статус: <strong style="color:{sc_color}">
            {"✅ ПОДТВЕРЖДЁН" if sc.get("confirmed") else "❌ не подтверждён"}
          </strong> (Δ{sc.get('delta',0):+.1f}%)</p>
          <p>Итераций: {len(sc.get('iterations',[]))}</p>
          {"<pre>" + _esc(json.dumps(sc.get("final_patch",{}), ensure_ascii=False, indent=2)) + "</pre>"
           if sc.get("final_patch") else ""}
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="utf-8">
<title>UkrSell Self-Debug — {slug} — {ts}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  body  {{ font-family: system-ui, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }}
  h1    {{ background: #1e293b; padding: 1.5rem 2rem; margin: 0;
           border-bottom: 2px solid #6366f1; font-size: 1.4rem; }}
  .meta {{ background: #1e293b; padding: .5rem 2rem; font-size:.85rem; color:#94a3b8; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px,1fr));
           gap: 1rem; padding: 1.5rem 2rem; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 1.2rem; }}
  .card h2 {{ margin: 0 0 .8rem; font-size: 1rem; color: #94a3b8; }}
  .big  {{ font-size: 2.8rem; font-weight: 700; }}
  .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px,1fr));
             gap: 1rem; padding: 0 2rem 1.5rem; }}
  canvas {{ max-height: 220px; }}
  .patterns {{ padding: 0 2rem 1.5rem; }}
  .pattern  {{ background:#1e293b; border-radius:8px; padding:1rem; margin-bottom:.8rem; }}
  .badge    {{ display:inline-block; padding:.2rem .5rem; border-radius:4px;
               font-size:.75rem; font-weight:700; color:#fff; margin-right:.5rem; }}
  .comp     {{ color:#6366f1; margin-left:.3rem; }}
  code      {{ display:block; background:#0f172a; padding:.5rem; border-radius:4px;
               font-size:.8rem; margin-top:.4rem; white-space:pre-wrap; color:#a5f3fc; }}
  .sc-block {{ background:#1e293b; border-radius:8px; padding:1rem;
               margin: 0 2rem 1.5rem; }}
  .sc-block pre {{ background:#0f172a; padding:.5rem; border-radius:4px;
                   font-size:.8rem; color:#a5f3fc; overflow:auto; }}
  .tbl-wrap {{ overflow-x:auto; padding: 0 2rem 2rem; }}
  table {{ width:100%; border-collapse:collapse; font-size:.82rem; }}
  th    {{ background:#1e293b; padding:.5rem .8rem; text-align:left;
           color:#94a3b8; position:sticky; top:0; }}
  td    {{ padding:.4rem .8rem; border-bottom:1px solid #1e293b; }}
  tr:hover td {{ background:#1e293b55; }}
  .kpi {{ font-size:1rem; color:#94a3b8; margin-top:.3rem; }}
</style>
</head>
<body>
<h1>🤖 UkrSell Self-Debug — {slug.upper()}</h1>
<div class="meta">
  {ts} &nbsp;|&nbsp;
  {metrics.get('total')} кейсов &nbsp;|&nbsp;
  pass {metrics.get('pass_rate')}% &nbsp;|&nbsp;
  hallucination {metrics.get('hallucination_rate')}% &nbsp;|&nbsp;
  coverage {metrics.get('coverage_rate')}% &nbsp;|&nbsp;
  p50 {metrics.get('latency_p50_ms')}ms / p95 {metrics.get('latency_p95_ms')}ms
</div>

<div class="grid">
  <div class="card">
    <h2>✅ Pass Rate</h2>
    <div class="big" style="color:#22c55e">{metrics.get('pass_rate')}%</div>
    <div class="kpi">chain: {metrics.get('chain_pass_rate','—')}% &nbsp;|&nbsp;
                     single: {metrics.get('single_pass_rate','—')}%</div>
  </div>
  <div class="card">
    <h2>👻 Hallucination Rate</h2>
    <div class="big" style="color:#f97316">{metrics.get('hallucination_rate')}%</div>
    <div class="kpi">{c.get('hallucination',0)} з {metrics.get('total')} кейсів</div>
  </div>
  <div class="card">
    <h2>📦 Coverage</h2>
    <div class="big" style="color:#6366f1">{metrics.get('coverage_rate')}%</div>
    <div class="kpi">унікальних товарів знайдено</div>
  </div>
  <div class="card">
    <h2>⏱ Latency</h2>
    <div class="big" style="color:#38bdf8">{metrics.get('latency_p50_ms')}ms</div>
    <div class="kpi">p50 &nbsp;|&nbsp; p95: {metrics.get('latency_p95_ms')}ms</div>
  </div>
</div>

<div class="charts">
  <div class="card">
    <h2>Результати</h2>
    <canvas id="verdictChart"></canvas>
  </div>
  <div class="card">
    <h2>Pass Rate за типом запиту</h2>
    <canvas id="typeChart"></canvas>
  </div>
  <div class="card">
    <h2>Latency розподіл</h2>
    <canvas id="latChart"></canvas>
  </div>
</div>

<div class="patterns">
  <div class="card">
    <h2>🔍 Паттерни помилок</h2>
    {patterns_html or '<p style="color:#94a3b8">Паттернів не знайдено</p>'}
    {('<p style="color:#22c55e">🎯 <strong>' + _esc(analysis.get("top_recommendation","")) + '</strong></p>')
     if analysis.get("top_recommendation") else ""}
  </div>
</div>

{sc_html}

<div class="tbl-wrap">
  <div class="card" style="background:none;padding:0">
  <h2 style="padding:.5rem 0;color:#94a3b8">📋 Всі кейси</h2>
  <table>
    <tr><th>Запит</th><th>Тип</th><th>Очікувано</th>
        <th>Вердикт</th><th>Latency</th><th>Причина / галюцинація</th></tr>
    {rows_html}
  </table>
  </div>
</div>

<script>
new Chart(document.getElementById('verdictChart'), {{
  type: 'doughnut',
  data: {{
    labels: {verdict_labels},
    datasets: [{{
      data: {verdict_data},
      backgroundColor: ['#22c55e','#ef4444','#f97316','#eab308','#6b7280']
    }}]
  }},
  options: {{ plugins: {{ legend: {{ labels: {{ color: '#e2e8f0' }} }} }} }}
}});
new Chart(document.getElementById('typeChart'), {{
  type: 'bar',
  data: {{
    labels: {type_labels},
    datasets: [{{
      label: 'Pass %',
      data: {type_data},
      backgroundColor: '#6366f1'
    }}]
  }},
  options: {{
    scales: {{
      x: {{ ticks: {{ color:'#94a3b8' }}, grid: {{ color:'#1e293b' }} }},
      y: {{ min:0, max:100, ticks: {{ color:'#94a3b8' }}, grid: {{ color:'#1e293b' }} }}
    }},
    plugins: {{ legend: {{ labels: {{ color:'#e2e8f0' }} }} }}
  }}
}});
new Chart(document.getElementById('latChart'), {{
  type: 'bar',
  data: {{
    labels: {lat_labels_js},
    datasets: [{{
      label: 'Кейсів',
      data: {lat_data_js},
      backgroundColor: '#38bdf8'
    }}]
  }},
  options: {{
    scales: {{
      x: {{ ticks: {{ color:'#94a3b8' }}, grid: {{ color:'#1e293b' }} }},
      y: {{ ticks: {{ color:'#94a3b8' }}, grid: {{ color:'#1e293b' }} }}
    }},
    plugins: {{ legend: {{ labels: {{ color:'#e2e8f0' }} }} }}
  }}
}});
</script>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"🌐 HTML dashboard: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# UTILS
# ═════════════════════════════════════════════════════════════════════════════

def _strip_md(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    return re.sub(r"\s*```$", "", text)


def _esc(text: str) -> str:
    return (str(text)
            .replace("&","&amp;").replace("<","&lt;")
            .replace(">","&gt;").replace('"',"&quot;"))


async def _llm_json_array(client, model, prompt, max_tokens=4000, temperature=0.4) -> List:
    try:
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system",
                       "content": "You are a precise JSON generator. Output only valid JSON array."},
                      {"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature,
        )
        raw = _strip_md(r.choices[0].message.content)
        return json.loads(raw)
    except Exception as e:
        print(f"  ❌ LLM JSON array error: {e}")
        return []


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

async def main(slug: str, n_cases: int, enable_sc: bool):
    t0 = time.perf_counter()
    print("\n" + "═" * 62)
    print(f"🤖  UkrSell Self-Debug Pipeline v2.0.0")
    print(f"    store={slug}  cases={n_cases}  self-correct={enable_sc}")
    print("═" * 62)

    # 1. Load
    products = load_catalog(slug)

    # 2. Init kernel
    print("\n⚙️  Инициализация ядра...")
    kernel = UkrSellKernel()
    await kernel.initialize()

    active = kernel.get_all_active_slugs()
    if slug not in active:
        print(f"❌ Магазин '{slug}' не найден. Доступные: {active}")
        await kernel.close()
        return

    ctx      = kernel.registry.get_context(slug)
    language = getattr(ctx, "language", "Ukrainian")
    print(f"✅ Ядро готово. Язык: {language}")

    client_fast, model_fast = await kernel.selector.get_fast()
    client_heavy, model_heavy = await kernel.selector.get_heavy()
    print(f"   fast={model_fast} | heavy={model_heavy}")

    # 3. Generate
    singles, chains = await generate_test_cases(
        client_fast, model_fast, products, n_cases, language, kernel
    )
    if not singles and not chains:
        print("❌ Тест-кейсы не сгенерированы. Выход.")
        await kernel.close()
        return

    # 4. Run
    results = await run_all(kernel, ctx, singles, chains)

    # 5. Evaluate
    evaluated = await evaluate_all(client_fast, model_fast, results)

    # 6. Metrics
    metrics = compute_metrics(evaluated)
    print(f"\n📊 Метрики: pass={metrics.get('pass_rate')}% | "
          f"hall={metrics.get('hallucination_rate')}% | "
          f"coverage={metrics.get('coverage_rate')}% | "
          f"p50={metrics.get('latency_p50_ms')}ms")

    # 7. Analyze + Suggest
    analysis = await analyze_and_suggest(
        client_heavy, model_heavy, evaluated, metrics, slug
    )

    # 8. Self-correction
    sc_result = {"skipped": True, "reason": "disabled"}
    if enable_sc:
        failures = [e for e in evaluated
                    if e.get("verdict") in ("fail", "hallucination", "partial", "negative_fail")]
        sc_result = await self_correct_loop(
            client_fast, model_fast, model_heavy,
            kernel, ctx, failures, analysis, slug,
        )
    else:
        print("\n⏭️  Self-correction отключён (--no-selfcorrect)")

    # 9. Report
    duration = time.perf_counter() - t0
    save_report(slug, evaluated, metrics, analysis, sc_result, duration)

    await kernel.close()
    print("\n✅ Self-Debug завершён.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UkrSell Self-Debug Pipeline v2.0.0")
    parser.add_argument("slug",     nargs="?", default=DEFAULT_SLUG)
    parser.add_argument("n_cases",  nargs="?", type=int, default=DEFAULT_N_CASES)
    parser.add_argument("--no-selfcorrect", dest="no_sc", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    try:
        asyncio.run(main(args.slug, args.n_cases, not args.no_sc))
    except KeyboardInterrupt:
        print("\n🛑 Прервано.")