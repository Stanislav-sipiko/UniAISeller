import sys
from pathlib import Path

# Добавляем корень проекта в пути поиска модулей
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import asyncio
from openai import AsyncOpenAI
from core.config import GEMINI_KEYS, OPENROUTER_KEYS, DEEPINFRA_KEY, COHERE_KEY

async def test():
    # Настройка клиентов
    g_client = AsyncOpenAI(api_key=GEMINI_KEYS[0], base_url="https://generativelanguage.googleapis.com/v1beta/openai")
    o_client = AsyncOpenAI(api_key=OPENROUTER_KEYS[0], base_url="https://openrouter.ai/api/v1")
    d_client = AsyncOpenAI(api_key=DEEPINFRA_KEY, base_url="https://api.deepinfra.com/v1/openai")
    c_client = AsyncOpenAI(api_key=COHERE_KEY, base_url="https://api.cohere.ai/v1")
    
    targets = [
        (g_client, "gemini-2.0-flash", "Direct Gemini"),
        (o_client, "google/gemini-2.0-flash-001", "OpenRouter (Gemini)"),
        (d_client, "meta-llama/Llama-3.3-70B-Instruct-Turbo", "DeepInfra (Llama 3.3)"),
        (c_client, "c4ai-aya-expanse-8b", "Cohere (Aya 8B)")
    ]
    
    print("--- Starting API Connectivity Test ---")
    for client, model, name in targets:
        try:
            resp = await client.chat.completions.create(
                model=model, 
                messages=[{"role": "user", "content": "say ok"}], 
                timeout=10
            )
            print(f"✅ {name}: {resp.choices[0].message.content.strip()}")
        except Exception as e:
            # Выводим короткую ошибку для наглядности
            err_msg = str(e)
            if "429" in err_msg:
                print(f"❌ {name}: Quota Exceeded (429)")
            else:
                print(f"❌ {name} Error: {err_msg[:80]}")

if __name__ == "__main__":
    asyncio.run(test())