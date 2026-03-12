# /root/ukrsell_v4/scripts/init_persona.py
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.store_profiler import StoreProfiler

async def main():
    store_slug = sys.argv[1] if len(sys.argv) > 1 else "phonestore"
    store_path = f"/root/ukrsell_v4/stores/{store_slug}"
    
    print(f"🚀 Запуск асинхронной инициализации для {store_slug}...")
    profiler = StoreProfiler(store_path)
    
    try:
        # Теперь это честный await без вложенных asyncio.run
        profile = await profiler.load_or_build()
        
        welcome = profile.get("ai_welcome_message")
        if welcome and "Вітаємо" not in welcome: # Если это не дефолтная фраза из заглушки
            print("✅ Приветствие сгенерировано нейросетью:")
            print("-" * 40)
            print(welcome)
            print("-" * 40)
        else:
            print("⚠️ Получено дефолтное приветствие или ошибка. Проверь ключи API.")
            
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")

if __name__ == "__main__":
    asyncio.run(main())