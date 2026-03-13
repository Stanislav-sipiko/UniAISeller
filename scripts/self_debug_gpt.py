# /root/ukrsell_v4/scripts/self_debug_gpt.py v2.4.7
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

# Настройка путей для работы внутри структуры проекта
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Отключаем лишний логгинг сторонних библиотек для чистоты вывода в консоль
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

from kernel import UkrSellKernel
from core.logger import logger

# Константы конфигурации
DEFAULT_SLUG = "luckydog"
DEFAULT_N_CASES = 20
CHAIN_LENGTH = 4
CHAINS_RATIO = 0.4
MAX_CATALOG_FOR_LLM = 80
ANALYSIS_FILE = ROOT_DIR / "logs" / "analysis_selection.jsonl"

# --- Вспомогательные функции ---

def save_debug_entry(entry: dict):
    """
    Сохраняет запись о прогоне теста в JSONL файл для последующего анализа.
    """
    os.makedirs(os.path.dirname(ANALYSIS_FILE), exist_ok=True)
    with open(ANALYSIS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def load_catalog(slug: str) -> List[Dict]:
    """
    Загружает нормализованный каталог товаров конкретного магазина.
    """
    path = ROOT_DIR / "stores" / slug / "normalized_products_final.json"
    if not path.exists():
        raise FileNotFoundError(f"Файл каталога не найден по пути: {path}")
    
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
        print(f"✅ Каталог загружен: {len(data)} товаров")
        return data

def _catalog_str(products: List[Dict]) -> str:
    """
    Преобразует список товаров в компактную строку для контекста LLM.
    """
    rows = []
    for p in products:
        rows.append(
            f"ID:{p.get('product_id')} | "
            f"{p.get('title')} | "
            f"cat:{p.get('category','')} | "
            f"brand:{p.get('brand','')}"
        )
    return "\n".join(rows)

# --- Работа с LLM (Генерация тестов) ---

async def _llm_json_array(client, model, prompt, max_tokens=3000):
    """
    Вызывает LLM и гарантирует получение чистого JSON-массива.
    """
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
        
        # Очистка от возможных Markdown-тегов
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```", "", raw)
        
        # Извлечение контента внутри квадратных скобок
        start = raw.find('[')
        end = raw.rfind(']') + 1
        if start != -1 and end != 0:
            raw = raw[start:end]
            
        return json.loads(raw)
    except Exception as e:
        print(f"❌ Ошибка парсинга JSON от LLM: {e}")
        # Возвращаем пустой список, чтобы не ломать выполнение
        return []

async def generate_single_cases(client, model, catalog_sample, n, language):
    """
    Генерирует независимые поисковые запросы.
    """
    if n <= 0:
        return []
    
    prompt = f"""
Generate {n} realistic customer queries for a pet shop in {language}. 
Queries should be natural, as if written by a person in a chat.
Mix types: 
1. Direct (exact item names from catalog)
2. Vague (broad categories or needs)
3. Brand focus (using brand names from catalog)
4. Negative (queries for things NOT in a pet shop).

CATALOG SAMPLE:
{_catalog_str(catalog_sample)}

Return ONLY a JSON array of objects:
[
  {{"query": "текст запроса", "expected_id": "ID товара или null", "query_type": "direct/vague/negative"}}
]
"""
    return await _llm_json_array(client, model, prompt)

async def generate_chain_cases(client, model, catalog_sample, n_chains, language):
    """
    Генерирует цепочки диалогов для тестирования контекста и DialogManager.
    """
    if n_chains <= 0:
        return []
    
    prompt = f"""
Generate {n_chains} conversation chains in {language}.
Each chain must have exactly {CHAIN_LENGTH} turns that logically follow each other:
Turn 0: Vague intent (e.g. "I have a puppy")
Turn 1: Category specification (e.g. "Looking for food")
Turn 2: Brand/Feature specification (e.g. "Something with chicken from Royal Canin")
Turn 3: Final purchase or direct specific request.

CATALOG SAMPLE:
{_catalog_str(catalog_sample)}

Return ONLY a JSON array of objects:
[
  {{
    "chain_id": "unique_string_id",
    "turns": [
      {{"turn": 0, "query": "...", "expected_id": "..."}},
      {{"turn": 1, "query": "...", "expected_id": "..."}},
      {{"turn": 2, "query": "...", "expected_id": "..."}},
      {{"turn": 3, "query": "...", "expected_id": "..."}}
    ]
  }}
]
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
    """
    Подготовка общего набора тестов: одиночные запросы + цепочки.
    """
    # Берем срез каталога, чтобы не превысить лимиты контекста LLM
    sample = products[:MAX_CATALOG_FOR_LLM]
    
    # Рассчитываем количество цепочек
    n_chains_total = max(1, int(n * CHAINS_RATIO))
    n_chains_count = n_chains_total // CHAIN_LENGTH
    
    # Оставшееся количество для одиночных запросов
    n_single = max(0, n - (n_chains_count * CHAIN_LENGTH))
    
    print(f"⚙️ План генерации: {n_single} одиночных + {n_chains_count} цепочек (всего {n_chains_count * CHAIN_LENGTH} шагов)")
    
    singles = await generate_single_cases(client, model, sample, n_single, language)
    # Небольшая пауза для избежания Rate Limits
    await asyncio.sleep(1.5) 
    chains = await generate_chain_cases(client, model, sample, n_chains_count, language)
    
    return singles, chains

# --- Выполнение тестов ---

async def run_single_test(kernel: UkrSellKernel, ctx: Any, case: dict, chat_id: str):
    """
    Выполняет один запрос через ядро и замеряет время выполнения.
    Построчная сверка с kernel.py v7.8.1 (строка 117):
    async def process_request(self, user_text: str, chat_id: Any, ctx: StoreContext, top_k: int = 5)
    """
    t0 = time.perf_counter()
    
    try:
        # Используем ПОЗИЦИОННУЮ передачу аргументов.
        # Порядок: 1. Текст, 2. ID чата, 3. Контекст магазина, 4. Top_k
        response_text = await kernel.process_request(
            case["query"],   # user_text
            chat_id,         # chat_id
            ctx,             # ctx
            5                # top_k
        )
        
        latency_ms = round((time.perf_counter() - t0) * 1000)
        
        # Подготовка данных для сохранения
        # Исправлено: замена datetime.utcnow() на datetime.now(UTC) для совместимости с Python 3.12+
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
        
        return {
            **case,
            "actual_response": response_text,
            "latency_ms": latency_ms
        }
        
    except Exception as e:
        # Важное логгирование: если упало на evaluate() - это баг в ядре
        print(f"❌ Ошибка ядра на запросе '{case['query']}': {str(e)}")
        
        # Исправлено: замена datetime.utcnow() на datetime.now(UTC)
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
    """
    Последовательно запускает все сгенерированные тестов.
    """
    results = []
    
    if singles:
        print("\n🚀 Запуск одиночных запросов...")
        for i, case in enumerate(singles):
            res = await run_single_test(kernel, ctx, case, chat_id=f"single_debug_{i}")
            results.append(res)
            # Визуальный индикатор в консоли
            status_icon = "✅" if "actual_response" in res else "❌"
            print(f"  {status_icon} [{i+1}/{len(singles)}] {case['query'][:50]}...")
            await asyncio.sleep(0.8)
        
    if chains:
        print("\n🔗 Запуск цепочек диалогов (тест контекста)...")
        for i, case in enumerate(chains):
            # Важно: для всей цепочки один и тот же chat_id, чтобы работал DialogManager
            chat_id = f"chain_debug_{case['chain_id']}"
            res = await run_single_test(kernel, ctx, case, chat_id=chat_id)
            results.append(res)
            
            status_icon = "✅" if "actual_response" in res else "❌"
            prefix = f"  {status_icon} Chain:{case['chain_id']} Step:{case.get('turn_index')}"
            print(f"{prefix} | {case['query'][:50]}...")
            
            # Небольшая пауза между репликами одного диалога
            await asyncio.sleep(1.2)
        
    return results

# --- Главный цикл управления ---

async def main(slug: str, n_cases: int):
    """
    Основная логика отладочного скрипта.
    """
    print("="*60)
    print(f"🛒 UKRSELL V4 SELF-DEBUGGER v2.4.7")
    print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Target Store: {slug}")
    print("="*60)
    
    # Инициализация ядра
    kernel = UkrSellKernel()
    
    try:
        # 1. Загрузка данных каталога для формирования промптов LLM
        products = load_catalog(slug)
        
        # 2. Инициализация ядра для конкретного магазина
        print(f"⚙️ Инициализация ядра для {slug}...")
        await kernel.initialize(target_slug=slug)
        
        # 3. Извлечение контекста магазина
        ctx = kernel.registry.get_context(slug)
        if not ctx:
            print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Магазин '{slug}' не найден в реестре.")
            return

        # 4. Получение доступа к LLM через селектор ядра
        print("🤖 Подключение к LLM для генерации тестов...")
        client, model = await kernel.selector.get_fast()
        
        # Определяем язык (по умолчанию Украинский, если не задано в контексте)
        language = getattr(ctx, "language", "Ukrainian")
        
        # 5. Генерация набора тестов
        singles, chains = await generate_test_cases(client, model, products, n_cases, language)
        
        if not singles and not chains:
            print("⚠️ Набор тестов пуст. Проверьте логи и доступ к LLM API.")
            return
            
        # 6. Запуск процесса тестирования
        start_exec = time.perf_counter()
        all_results = await run_all_tests(kernel, ctx, singles, chains)
        end_exec = time.perf_counter()
        
        # 7. Итоговая сводка
        total_time = round(end_exec - start_exec, 2)
        success_count = len([r for r in all_results if "actual_response" in r])
        error_count = len(all_results) - success_count
        
        print("\n" + "="*60)
        print(f"📊 ИТОГИ ТЕСТИРОВАНИЯ:")
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
        # Обязательное закрытие ресурсов ядра (сессии, соединения)
        await kernel.close()

if __name__ == "__main__":
    # Обработка аргументов командной строки
    parser = argparse.ArgumentParser(description="UkrSell V4 Kernel Debugger")
    parser.add_argument("slug", nargs="?", default=DEFAULT_SLUG, help="Slug магазина для теста")
    parser.add_argument("n_cases", nargs="?", type=int, default=DEFAULT_N_CASES, help="Количество тестов")
    
    args = parser.parse_args()

    try:
        # Запуск асинхронного цикла
        asyncio.run(main(args.slug, args.n_cases))
    except KeyboardInterrupt:
        print("\n🛑 Процесс прерван пользователем (Ctrl+C).")
        sys.exit(0)