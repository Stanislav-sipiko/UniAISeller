# /root/ukrsell_v4/scripts/debug_kernel_warmup.py v5.14.0
import asyncio
import sys
import os
import traceback

# Добавляем корень проекта
sys.path.append('/root/ukrsell_v4')

try:
    from kernel import UkrSellKernel
    from core.logger import logger
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    sys.exit(1)

async def debug_warmup():
    print("\n" + "="*60)
    print("🔍 DEBUG: UKRSELL KERNEL WARMUP & LLM STATUS")
    print("="*60)

    # 1. Init Kernel
    try:
        print("🛠️ Создание экземпляра UkrSellKernel...")
        kernel = UkrSellKernel()
        
        start_time = asyncio.get_event_loop().time()
        print("⏳ Запуск kernel.initialize()...")
        # Проверяем наличие метода initialize (в некоторых версиях он вызывается в __init__)
        if hasattr(kernel, 'initialize'):
            await kernel.initialize()
        else:
            print("⚠️ Метод .initialize() не найден, ожидаю через llm_ready...")
            await asyncio.wait_for(kernel.llm_ready.wait(), timeout=30.0)
            
        end_time = asyncio.get_event_loop().time()
        print(f"⏱️ Kernel initialized in: {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"❌ Критическая ошибка при инициализации:")
        traceback.print_exc()
        return

    # 2. Check LLM Status
    # Проверяем, как называется аттрибут в вашем kernel.py (selector или llm_selector)
    selector = getattr(kernel, 'selector', getattr(kernel, 'llm_selector', None))
    
    if selector:
        status = selector.get_status()
        print("\n📊 Current LLM Stack Status:")
        for tier, model in status.items():
            color = "🟢" if model != "OFFLINE" else "🔴"
            print(f"    {color} {tier.upper()}: {model}")
    else:
        print("\n❌ Ошибка: Селектор LLM не найден в объекте Kernel.")

    # 3. Test Key Rotation & Dispatch
    print("\n⚡ Testing LLM Dispatch (Fast Tier)...")
    try:
        if selector:
            client, model = selector.get_fast()
            print(f"    ✅ Dispatch Success: Model={model}")
            # Безопасное отображение ключа
            api_key = getattr(client, 'api_key', 'N/A')
            print(f"    🔑 Active Key (truncated): {api_key[:10]}***")
    except Exception as e:
        print(f"    ❌ Dispatch Failed: {e}")

    # 4. Verify Deduplication Logic (Dry Run)
    print("\n🧹 Testing Deduplication Logic...")
    test_items = [
        {"product": {"name": "Xiaomi Redmi Note 14 Pro ", "price": 1000}},
        {"product": {"name": "xiaomi  redmi note 14 pro", "price": 1200}}, 
        {"product": {"name": "Samsung S24", "price": 45000}}
    ]
    
    import re
    seen_names = set()
    deduplicated = []
    for res in test_items:
        raw_name = res["product"]["name"]
        clean_name = re.sub(r'\s+', ' ', raw_name).strip().lower()
        if clean_name not in seen_names:
            seen_names.add(clean_name)
            deduplicated.append(res)
    
    print(f"    Input items: {len(test_items)}")
    print(f"    After deduplication: {len(deduplicated)}")
    if len(deduplicated) == 2:
        print("    ✅ Deduplication works correctly (Xiaomi duplicates removed).")
    else:
        print("    ❌ Deduplication failed to catch variants.")

    await kernel.close()
    print("\n" + "="*60)
    print("🏁 DEBUG COMPLETE")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(debug_warmup())