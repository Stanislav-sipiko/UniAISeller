import asyncio
import aiohttp
import json
import os
import shutil
from pathlib import Path
from kernel import PlatformKernel

# --- Константы теста ---
TEST_STORE_SLUG = "test_shop"
BASE_URL = "http://127.0.0.1:8090"

async def setup_mock_store():
    """Создает окружение для теста с нужной структурой."""
    store_path = Path(f"stores/{TEST_STORE_SLUG}")
    store_path.mkdir(parents=True, exist_ok=True)
    (store_path / "data").mkdir(exist_ok=True)

    # 1. Config БЕЗ корма
    config = {
        "bot_token": "123:ABC",
        "store_name": "Тестовый Магазин",
        "indexing_fields": ["name", "description"],
        "filters": {
            "category": ["коврики", "миски"] # Корма тут нет!
        }
    }
    
    # 2. Товары
    products = [
        {
            "name": "Лизательный коврик",
            "type": "коврики",
            "price": 500,
            "url": "http://link.com/mat",
            "description": "Идеален для поедания корма"
        }
    ]

    # 3. Промпты (заглушка)
    prompts = {"welcome": "Привет!"}

    with open(store_path / "config.json", "w") as f: json.dump(config, f)
    with open(store_path / "products.json", "w") as f: json.dump(products, f)
    with open(store_path / "prompts.json", "w") as f: json.dump(prompts, f)

async def run_test_requests():
    """Эмулирует входящие вебхуки от Telegram."""
    async with aiohttp.ClientSession() as session:
        # Тест 1: Запрос того, что ЕСТЬ (Коврик)
        print("\n--- TEST 1: Requesting 'коврик' ---")
        payload_ok = {
            "message": {
                "chat": {"id": 12345},
                "text": "Покажи мне коврик"
            }
        }
        async with session.post(f"{BASE_URL}/webhook/{TEST_STORE_SLUG}/", json=payload_ok) as resp:
            print(f"Status: {resp.status}")
            print(f"Response: {await resp.text()}")

        # Тест 2: Запрос того, чего НЕТ (Корм) - проверка фильтра
        print("\n--- TEST 2: Requesting 'корм' (Should be filtered) ---")
        payload_fail = {
            "message": {
                "chat": {"id": 12345},
                "text": "Я хочу купить корм для кошек"
            }
        }
        async with session.post(f"{BASE_URL}/webhook/{TEST_STORE_SLUG}/", json=payload_fail) as resp:
            print(f"Status: {resp.status}")
            print(f"Response: {await resp.text()}")

async def main():
    # Подготовка
    await setup_mock_store()
    
    # Запуск ядра (в реальном сценарии это отдельный процесс)
    kernel = PlatformKernel()
    
    # Мы запускаем сервер асинхронно, чтобы выполнить тесты в этой же петле
    runner = aiohttp.web.AppRunner(kernel.app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, '127.0.0.1', 8090)
    await site.start()
    
    print("Kernel is up and running for tests.")
    
    try:
        await run_test_requests()
    finally:
        await runner.cleanup()
        # Удаляем тестовую папку
        # shutil.rmtree(f"stores/{TEST_STORE_SLUG}")

if __name__ == "__main__":
    asyncio.run(main())