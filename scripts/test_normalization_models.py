# /root/ukrsell_v4/scripts/test_normalization_models.py v1.0.2
import os
import json
import asyncio
import time
import sys
from pathlib import Path

# Настройка путей
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from core.llm_selector import LLMSelector
from core.logger import logger

STORE_SLUG = "luckydog"
BATCH_SIZE = 5
TOTAL_TEST_ITEMS = 15

# Смешанный список для надежности
TEST_MODELS = [
    {"type": "openrouter", "model": "google/gemini-2.0-flash-001", "label": "gemini_flash"}, # Легкая модель
    {"type": "deepinfra", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "label": "llama_heavy"}, # Через DeepInfra
    {"type": "openrouter", "model": "qwen/qwen-2.5-72b-instruct", "label": "qwen_heavy"}
]

PROMPT_TEMPLATE = """Normalize these 5 pet products into JSON.
Return ONLY valid JSON array.
DATA: {batch_json}"""

async def run_model_test(selector, model_entry, products_batch):
    client, prepared_model = selector.prepare_entry(model_entry, tier_label="TEST")
    if not client: return None

    try:
        # Устанавливаем жесткий лимит на max_tokens, чтобы пройти по балансу OpenRouter
        response = await client.chat.completions.create(
            model=prepared_model,
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(batch_json=json.dumps(products_batch, ensure_ascii=False))}],
            temperature=0.0,
            max_tokens=2000 
        )
        content = response.choices[0].message.content
        clean_json = content.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        logger.error(f"Error {model_entry['label']}: {e}")
        return None

async def main():
    store_path = Path(f"/root/ukrsell_v4/stores/{STORE_SLUG}")
    raw_path = store_path / "products.json"

    with open(raw_path, "r", encoding="utf-8") as f:
        test_items = json.load(f)[:TOTAL_TEST_ITEMS]
    
    selector = LLMSelector()
    await selector.ensure_ready()

    for m_meta in TEST_MODELS:
        results = []
        print(f"--- Testing {m_meta['label']} ---")
        for i in range(0, len(test_items), BATCH_SIZE):
            batch = test_items[i:i+BATCH_SIZE]
            lite_batch = [{"id": p.get("product_id"), "title": p.get("title")} for p in batch]
            
            res = await run_model_test(selector, m_meta, lite_batch)
            if res: 
                results.extend(res)
                print(f"Batch {i//5 + 1} OK")
            else:
                print(f"Batch {i//5 + 1} FAILED")

        output_file = store_path / f"test_{m_meta['label']}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())