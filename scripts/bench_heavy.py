# /root/ukrsell_v4/scripts/bench_heavy.py
import asyncio
import sys
import os
import json
import time
import re
from pathlib import Path

# --- Настройка путей (как в debugger_gemini.py) ---
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# --- Импорты после настройки путей ---
from core.config import LLM_TIERS
from kernel import UkrSellKernel

# Глобальный путь к рейтингу моделей
BENCHMARK_FILE = ROOT_DIR / "core" / "llm_rating.json"

async def run_benchmark():
    print("="*60)
    print("🚀 Starting Intelligence Benchmark for 'heavy' tier...")
    print(f"🎯 Target File: {BENCHMARK_FILE}")
    print("="*60)
    
    kernel = UkrSellKernel()
    await kernel.initialize()
    
    models = LLM_TIERS.get("heavy", [])
    if not models:
        print("❌ No models found in 'heavy' tier config.")
        await kernel.close()
        return

    results = {}

    test_prompt = """Generate 1 conversation chain in Ukrainian for a pet shop. 
    The chain must have exactly 4 turns. 
    Return ONLY a JSON array of objects with 'chain_id' and 'turns'. 
    Прислать валидный JSON, без дополнительного текста."""

    for model_id in models:
        print(f"\n🧪 Testing model: {model_id}...")
        scores = {"json_valid": 0, "logic_depth": 0, "id_match": 0}
        
        try:
            # Получаем клиент через селектор ядра
            # Используем защищенный метод селектора
            client, label = kernel.selector._build_client_dispatch(model_id)
            if not client:
                print(f"⚠️ Could not build client for {model_id}")
                continue
            
            t0 = time.perf_counter()
            r = await client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": test_prompt}],
                temperature=0.2,
                max_tokens=2000
            )
            latency = time.perf_counter() - t0
            raw_content = r.choices[0].message.content.strip()
            
            # Очистка контента от Markdown
            clean_content = re.sub(r"```json\s*", "", raw_content)
            clean_content = re.sub(r"```", "", clean_content)
            
            # 1. Проверка JSON
            try:
                start = clean_content.find('[')
                end = clean_content.rfind(']') + 1
                if start != -1 and end != 0:
                    data = json.loads(clean_content[start:end])
                    scores["json_valid"] = 10
                    
                    # 2. Проверка глубины (наличие 4 ходов)
                    if isinstance(data, list) and len(data) > 0:
                        # Обработка возможного вложенного списка или объекта
                        target = data[0] if isinstance(data, list) else data
                        turns = target.get("turns", [])
                        if len(turns) == 4:
                            scores["logic_depth"] = 10
                        
                        # 3. Проверка наличия полей в первом ходе
                        if turns and isinstance(turns[0], dict) and "query" in turns[0]:
                            scores["id_match"] = 10
                else:
                    print(f"❌ {model_id} produced no JSON markers.")
            except Exception as parse_err:
                print(f"❌ {model_id} failed JSON parsing: {parse_err}")

            total_score = sum(scores.values())
            results[model_id] = {
                "score": total_score,
                "latency_avg": round(latency, 2),
            }
            print(f"✅ {model_id} | Score: {total_score} | Latency: {round(latency, 2)}s")

        except Exception as e:
            print(f"💥 {model_id} Critical Error: {e}")

    if not results:
        print("❌ No benchmark results collected.")
        await kernel.close()
        return

    # Ранжирование: приоритет score (desc), затем latency (asc)
    sorted_models = sorted(
        results.items(), 
        key=lambda x: (-x[1]['score'], x[1]['latency_avg'])
    )
    
    final_ranks = {}
    for index, (m_id, data) in enumerate(sorted_models):
        final_ranks[m_id] = index + 1 # 1 = Лучшая модель

    # Сохранение результата
    os.makedirs(BENCHMARK_FILE.parent, exist_ok=True)
    with open(BENCHMARK_FILE, "w", encoding="utf-8") as f:
        json.dump(final_ranks, f, indent=4, ensure_ascii=False)
    
    print("\n" + "="*60)
    print(f"🏁 Benchmark complete. Ranks saved to {BENCHMARK_FILE}")
    for m, r in final_ranks.items():
        print(f"Rank {r}: {m}")
    print("="*60)
    
    await kernel.close()

if __name__ == "__main__":
    asyncio.run(run_benchmark())