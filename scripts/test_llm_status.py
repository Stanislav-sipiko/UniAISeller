import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.llm_selector import LLMSelector

async def test_tier(selector, name, getter):
    print(f"🧪 Тестирую {name}...")
    try:
        client, model = getter()
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hi, 1 word response."}],
            max_tokens=5,
            timeout=10.0
        )
        result = resp.choices[0].message.content.strip()
        print(f"✅ {name} OK [{model}]: {result}")
    except Exception as e:
        print(f"❌ {name} FAIL: {e}")

async def diagnostic_test():
    print("\n🛰️ ЗАПУСК ДИАГНОСТИКИ LLM СТЕКА v6.0.0 (Key Pooling)")
    print("="*60)
    
    selector = LLMSelector()
    
    # 1. Подготовка стеков
    print("🔍 Обновляю стеки моделей...")
    await selector.ensure_ready()
    
    status = selector.get_status()
    print(f"💎 ТЕКУЩИЙ СТАТУС:")
    print(f"   - LIGHT: {status['light']}")
    print(f"   - FAST:  {status['fast']}")
    print(f"   - HEAVY: {status['heavy']}")
    print("-" * 60)

    # 2. Тесты всех тиров
    await test_tier(selector, "LIGHT (8B)", selector.get_light)
    await test_tier(selector, "FAST (70B)", selector.get_fast)
    await test_tier(selector, "HEAVY (PRO)", selector.get_heavy)

    # 3. Проверка Cohere
    print("\n🎯 Тестирую Cohere (Reranker)...")
    try:
        co = selector.get_reranker()
        # Простая проверка через список моделей (синхронный вызов в Cohere SDK)
        res = co.models.list()
        print(f"✅ Cohere OK (Connected)")
    except Exception as e:
        print(f"❌ Cohere FAIL: {e}")

    await selector.close()
    print("="*60)
    print("🏁 Диагностика завершена.")

if __name__ == "__main__":
    asyncio.run(diagnostic_test())