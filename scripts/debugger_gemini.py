#/root/ukrsell_v4/scripts/debugger_gemini.py v2.6.0
import asyncio
import sys
import os
import json
import time
import logging
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, UTC

FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

from kernel import UkrSellKernel
from core.logger import logger

DEFAULT_SLUG = "luckydog"
DEFAULT_N_CASES = 3
CHAIN_LENGTH = 4
MAX_CATALOG_FOR_LLM = 80
ANALYSIS_FILE = ROOT_DIR / "logs" / "analysis_selection.jsonl"

TEST_MODES = [
    "normal_purchase",
    "trolling",
    "humor",
    "absurd",
    "small_talk",
    "stress_dialog",
    "nonsense",
    "long_context"
]

TROLL_CASES = [
    "У вас є лежанка для динозавра?",
    "Когтеріз підходить для бороди?",
    "Мій кіт мільйонер, що порадите?",
    "А якщо собака з'їсть когтеріз?",
    "Ти взагалі розумний бот?"
]

LONG_CHAIN = [
    "Покажи лежанки",
    "А ця підходить для котів?",
    "Який розмір?",
    "А колір?",
    "А гарантія?",
    "А доставка?",
    "А виробник?",
    "А є дешевше?",
    "А для собак?",
    "А водонепроникна?"
]

def print_status(component, status, message=""):
    color = "\033[92m[OK]\033[0m" if status else "\033[91m[FAIL]\033[0m"
    print(f"{color} {component:25} : {message}")

def save_debug_entry(entry: dict):
    os.makedirs(os.path.dirname(ANALYSIS_FILE), exist_ok=True)
    with open(ANALYSIS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def load_catalog(slug: str) -> List[Dict]:
    path = ROOT_DIR / "stores" / slug / "normalized_products_final.json"
    if not path.exists():
        raise FileNotFoundError(f"Файл каталога не найден по пути: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
        print(f"✅ Каталог загружен: {len(data)} товаров")
        return data

def _catalog_str(products: List[Dict]) -> str:
    rows = []
    for p in products[:MAX_CATALOG_FOR_LLM]:
        rows.append(
            f"ID:{p.get('product_id')} | "
            f"{p.get('title')} | "
            f"cat:{p.get('category','')} | "
            f"brand:{p.get('brand','')}"
        )
    return "\n".join(rows)

def inject_long_query(cases):
    long_query = (
        "Мій кіт живе у великій квартирі, дуже вибагливий, "
        "любить спати тільки на м'яких поверхнях, але іноді "
        "дряпає меблі, тому я думаю купити лежанку або "
        "когтеріз, але не знаю що краще, що ви порадите?"
    )
    cases.append({
        "query": long_query,
        "expected_id": None,
        "query_type": "stress"
    })
    return cases

def inject_trolling(cases):
    for t in TROLL_CASES:
        cases.append({
            "query": t,
            "expected_id": None,
            "query_type": "trolling"
        })
    return cases

async def _llm_json_array(kernel: UkrSellKernel, prompt: str, max_tokens=3000):
    max_retries = 3
    raw = ""
    model_id = "not_acquired"
    for attempt in range(max_retries):
        try:
            client, model_id = await kernel.selector.get_fast()
            r = await client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are a strict QA JSON generator. You output ONLY raw JSON arrays. "
                            "STRICT RULES: No markdown blocks (no ```json), no conversational filler, "
                            "no explanations. If you break these rules, the system fails. "
                            "Output must start with '[' and end with ']'. "
                            "Ensure every turn in a chain is a complete object, not a string."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.2 + (attempt * 0.1)
            )
            raw = r.choices[0].message.content.strip()
            if not raw:
                raise ValueError("LLM returned EMPTY RESPONSE")
            clean_raw = re.sub(r"```json\s*", "", raw)
            clean_raw = re.sub(r"```", "", clean_raw)
            start = clean_raw.find('[')
            end = clean_raw.rfind(']') + 1
            if start != -1 and end != 0:
                final_json = clean_raw[start:end]
                return json.loads(final_json)
            else:
                raise ValueError("JSON Array markers '[' or ']' not found in response")
        except Exception as e:
            current_model = model_id
            print(f"\n⚠️ Попытка {attempt+1}/{max_retries} провалена на модели {current_model}: {e}")
            if model_id and model_id != "not_acquired":
                kernel.selector.report_failure(model_id)
            if attempt == max_retries - 1:
                print("\n" + "!"*40)
                print("❌ КРИТИЧЕСКАЯ ОШИБКА ОБМЕНА С LLM")
                print(f"Последняя модель в попытке: {current_model}")
                print("-" * 20 + " RAW RESPONSE " + "-" * 20)
                print(raw if raw else "[EMPTY RESPONSE]")
                print("!"*40 + "\n")
                return []
            await asyncio.sleep(1)
    return []

async def generate_single_cases(kernel, catalog_sample, n, language):
    if n <= 0: return []
    prompt = f"""
Generate {n} diverse user messages for testing an AI shopping assistant for a pet store in {language}.

IMPORTANT:
Include a MIX of the following categories:
1 normal purchase intent
2 trolling (user mocking the bot)
3 absurd requests
4 humor
5 nonsense queries
6 small talk
7 intentionally confusing queries
8 stress tests (very long queries)

For purchase queries include expected_id from catalog.
For non-purchase queries expected_id must be null.

Return ONLY JSON array.

SCHEMA:
[
  {{
    "query": "string",
    "expected_id": "ID or null",
    "query_type": "purchase | trolling | humor | absurd | smalltalk | nonsense | stress"
  }}
]

CATALOG SAMPLE:
{_catalog_str(catalog_sample)}
"""
    cases = await _llm_json_array(kernel, prompt)
    if cases:
        cases = inject_long_query(cases)
        cases = inject_trolling(cases)
    return cases

async def generate_chain_cases(kernel, catalog_sample, n_chains, language):
    if n_chains <= 0: return []
    prompt = f"""
Generate {n_chains} conversation chains in {language} for testing an AI shopping assistant.
Each chain must have exactly {CHAIN_LENGTH} turns.

Include different scenarios:
1 normal purchase flow
2 trolling conversation
3 humor conversation
4 absurd questions
5 long dialogue with many follow-ups
6 stress test where user changes topic
7 nonsense dialogue

Some chains should start serious and then become trolling.

SCHEMA:
[
  {{
    "chain_id": "string",
    "turns": [
      {{
        "query": "string",
        "turn": 0,
        "expected_id": "ID or null",
        "query_type": "purchase | trolling | humor | absurd | nonsense | stress"
      }}
    ]
  }}
]

CATALOG SAMPLE:
{_catalog_str(catalog_sample)}
"""
    chains_data = await _llm_json_array(kernel, prompt)
    flat_cases = []
    if not isinstance(chains_data, list): return []
    for chain in chains_data:
        if not isinstance(chain, dict): continue
        cid = chain.get("chain_id", f"cid_{int(time.time())}")
        turns = chain.get("turns", [])
        if not isinstance(turns, list): continue
        for t in turns:
            if isinstance(t, dict):
                flat_cases.append({
                    "query": t.get("query", ""),
                    "expected_id": t.get("expected_id"),
                    "query_type": t.get("query_type", "direct"),
                    "is_chain": True,
                    "chain_id": cid,
                    "turn_index": t.get("turn", 0)
                })
    return flat_cases

async def generate_test_cases(kernel, products, n, language, chains_enabled=False):
    sample = products[:MAX_CATALOG_FOR_LLM]
    if chains_enabled:
        n_chains_count = max(1, n // CHAIN_LENGTH)
        print(f"⚙️ План генерации: {n_chains_count} цепочек по {CHAIN_LENGTH} шагов.")
        chains = await generate_chain_cases(kernel, sample, n_chains_count, language)
        return [], chains
    else:
        print(f"⚙️ План генерации: {n} проверочных запросов (Single mode)")
        singles = await generate_single_cases(kernel, sample, n, language)
        return singles, []

async def run_single_test(kernel: UkrSellKernel, ctx: Any, case: dict, chat_id: str):
    t0 = time.perf_counter()
    try:
        response_text = await kernel.process_request(
            case["query"],
            chat_id,
            ctx,
            5
        )
        latency_ms = round((time.perf_counter() - t0) * 1000)
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "query": case["query"],
            "query_type": case.get("query_type"),
            "expected_id": case.get("expected_id"),
            "latency_ms": latency_ms,
            "is_chain": case.get("is_chain", False),
            "chain_id": case.get("chain_id"),
            "chat_id": chat_id,
            "status": "SUCCESS"
        }
        save_debug_entry(entry)
        return {**case, "actual_response": response_text, "latency_ms": latency_ms}
    except Exception as e:
        print(f"❌ Ошибка ядра на запросе '{case['query']}': {str(e)}")
        error_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "query": case["query"],
            "query_type": case.get("query_type"),
            "error": str(e),
            "status": "ERROR",
            "chat_id": chat_id
        }
        save_debug_entry(error_entry)
        return {**case, "error": str(e)}

async def run_all_tests(kernel, ctx, singles, chains):
    results = []
    if singles:
        print("\n🚀 Запуск стабилизационных тестов...")
        for i, case in enumerate(singles):
            res = await run_single_test(kernel, ctx, case, chat_id=f"debug_fix_{i}")
            results.append(res)
            status_icon = "✅" if "actual_response" in res else "❌"
            q_type = case.get('query_type', 'N/A')
            print(f"  {status_icon} [{i+1}/{len(singles)}] [{q_type}] {case['query'][:50]}...")
            await asyncio.sleep(0.5)
    if chains:
        print("\n🚀 Запуск цепочек диалогов (Chain Mode)...")
        current_chain_id = None
        current_chat_id = None
        for i, case in enumerate(chains):
            if case["chain_id"] != current_chain_id:
                current_chain_id = case["chain_id"]
                current_chat_id = f"debug_chain_{int(time.time())}_{i}"
                print(f"\n--- Новая цепочка: {current_chain_id} ---")
            res = await run_single_test(kernel, ctx, case, chat_id=current_chat_id)
            results.append(res)
            status_icon = "✅" if "actual_response" in res else "❌"
            q_type = case.get('query_type', 'N/A')
            print(f"  {status_icon} Turn {case.get('turn_index')} [{q_type}]: {case['query'][:50]}...")
            await asyncio.sleep(0.8)
    return results

async def main(slug: str, n_cases: int, chains_enabled: bool, mode: str = None):
    print("="*60)
    print(f"🛒 UKRSELL V4 STABILITY-FIX DEBUGGER v2.6.0")
    print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Target Store: {slug} | Tests: {n_cases} | Chains: {chains_enabled}")
    print("="*60)
    kernel = UkrSellKernel()
    try:
        products = load_catalog(slug)
        print(f"⚙️ Инициализация ядра для {slug}...")
        await kernel.initialize()
        if not hasattr(kernel, 'registry'):
            raise RuntimeError("Kernel registry is not initialized.")
        ctx = kernel.registry.get_context(slug)
        if not ctx:
            print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Магазин '{slug}' не найден в реестре.")
            return
        print("🤖 Генерация тестов...")
        language = getattr(ctx, "language", "Ukrainian")
        singles, chains = await generate_test_cases(kernel, products, n_cases, language, chains_enabled)
        if not singles and not chains:
            print("⚠️ Набор тестов пуст. Проверьте доступность LLM моделей.")
            return
        start_exec = time.perf_counter()
        all_results = await run_all_tests(kernel, ctx, singles, chains)
        end_exec = time.perf_counter()
        total_time = round(end_exec - start_exec, 2)
        success_count = len([r for r in all_results if "actual_response" in r])
        error_count = len(all_results) - success_count
        print("\n" + "="*60)
        print(f"📊 ИТОГИ ТЕСТИРОВАНИЯ (FIX MODE):")
        print(f"⏱ Общее время выполнения: {total_time} сек.")
        print(f"✅ Успешно обработано: {success_count}")
        print(f"❌ Ошибок выполнения: {error_count}")
        print(f"📂 Детальный лог сохранен в: {ANALYSIS_FILE}")
        print("="*60)
    except Exception as e:
        print(f"🔥 ФАТАЛЬНАЯ ОШИБКА В МЕЙНЕ: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await kernel.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UkrSell V4 Kernel Debugger (Stability Mode)")
    parser.add_argument("slug", nargs="?", default=DEFAULT_SLUG, help="Slug магазина для теста")
    parser.add_argument("n_cases", nargs="?", type=int, default=DEFAULT_N_CASES, help="Количество тестов")
    parser.add_argument("-c", "--chain", action="store_true", help="Включить режим цепочки диалога")
    parser.add_argument("-m", "--mode", choices=TEST_MODES, help="Запустить конкретный тип теста")
    args = parser.parse_args()
    try:
        asyncio.run(main(args.slug, args.n_cases, args.chain, args.mode))
    except KeyboardInterrupt:
        print("\n🛑 Процесс прерван пользователем (Ctrl+C).")
        sys.exit(0)