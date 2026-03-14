# /root/ukrsell_v4/scripts/debugger.py v2.4.8

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
CHAINS_RATIO = 0.0
MAX_CATALOG_FOR_LLM = 80
ANALYSIS_FILE = ROOT_DIR / "logs" / "analysis_selection.jsonl"

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
    for p in products:
        rows.append(
            f"ID:{p.get('product_id')} | "
            f"{p.get('title')} | "
            f"cat:{p.get('category','')} | "
            f"brand:{p.get('brand','')}"
        )
    return "\n".join(rows)

async def _llm_json_array(client, model, prompt, max_tokens=3000):
    try:
        r = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a QA assistant. You generate only valid JSON arrays. No markdown, no triple backticks, just the raw array content."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.5
        )
        raw = r.choices[0].message.content.strip()
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```", "", raw)
        start = raw.find('[')
        end = raw.rfind(']') + 1
        if start != -1 and end != 0:
            raw = raw[start:end]
        return json.loads(raw)
    except Exception as e:
        print(f"❌ Ошибка парсинга JSON от LLM: {e}")
        return []

async def generate_single_cases(client, model, catalog_sample, n, language):
    if n <= 0:
        return []
    prompt = f"""
Generate {n} realistic customer queries for a pet shop in {language}. 
Queries should be natural, as if written by a person in a chat.
Mix types: 
1. Direct (exact item names from catalog)
2. Vague (broad categories or needs)
3. Brand focus (using brand names from catalog).

CATALOG SAMPLE:
{_catalog_str(catalog_sample)}

Return ONLY a JSON array of objects:
[
  {{"query": "текст запроса", "expected_id": "ID товара или null", "query_type": "direct/vague"}}
]
"""
    return await _llm_json_array(client, model, prompt)

async def generate_chain_cases(client, model, catalog_sample, n_chains, language):
    if n_chains <= 0:
        return []
    prompt = f"""
Generate {n_chains} conversation chains in {language}.
Each chain must have exactly {CHAIN_LENGTH} turns.
CATALOG SAMPLE:
{_catalog_str(catalog_sample)}
Return ONLY a JSON array of objects.
"""
    chains_data = await _llm_json_array(client, model, prompt)
    flat_cases = []
    for chain in chains_data:
        cid = chain.get("chain_id", f"cid_{int(time.time())}")
        for t in chain.get("turns", []):
            flat_cases.append({
                "query": t["query"],
                "expected_id": t.get("expected_id"),
                "is_chain": True,
                "chain_id": cid,
                "turn_index": t.get("turn", 0)
            })
    return flat_cases

async def generate_test_cases(client, model, products, n, language):
    sample = products[:MAX_CATALOG_FOR_LLM]
    n_chains_total = max(0, int(n * CHAINS_RATIO))
    n_chains_count = n_chains_total // CHAIN_LENGTH
    n_single = n
    print(f"⚙️ План генерации: {n_single} проверочных запросов (Chain mode disabled)")
    singles = await generate_single_cases(client, model, sample, n_single, language)
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
            print(f"  {status_icon} [{i+1}/{len(singles)}] {case['query'][:50]}...")
            await asyncio.sleep(0.5)
    return results

async def main(slug: str, n_cases: int):
    print("="*60)
    print(f"🛒 UKRSELL V4 STABILITY-FIX DEBUGGER v2.4.8")
    print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Target Store: {slug} | Tests: {n_cases}")
    print("="*60)
    kernel = UkrSellKernel()
    try:
        products = load_catalog(slug)
        print(f"⚙️ Инициализация ядра для {slug}...")
        #await kernel.initialize(target_slug=slug)
        await kernel.initialize()
        ctx = kernel.registry.get_context(slug)
        if not ctx:
            print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Магазин '{slug}' не найден в реестре.")
            return
        print("🤖 Генерация точечных тестов...")
        client, model = await kernel.selector.get_fast()
        language = getattr(ctx, "language", "Ukrainian")
        singles, chains = await generate_test_cases(client, model, products, n_cases, language)
        if not singles:
            print("⚠️ Набор тестов пуст. Проверьте логи и доступ к LLM API.")
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
    args = parser.parse_args()
    try:
        asyncio.run(main(args.slug, args.n_cases))
    except KeyboardInterrupt:
        print("\n🛑 Процесс прерван пользователем (Ctrl+C).")
        sys.exit(0)