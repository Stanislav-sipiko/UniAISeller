# /root/ukrsell_v4/scripts/test_router_flow.py
import asyncio
import os
import sqlite3
import json
import datetime
from core.dialog_manager import DialogManager
from core.llm_selector import LLMSelector
from core.store_context import StoreContext

async def test_flow_up_logic():
    print("🚀 Запуск теста Flow Up & Router v5.1...")
    
    # 1. Имитация контекста магазина
    class MockCtx:
        def __init__(self):
            self.slug = "phonestore"
            self.base_path = "/root/ukrsell_v4/stores/phonestore"
            self.prompts = {
                "product_consultant": "Ты эксперт по гаджетам в магазине Phonestore.",
                "troll_stop": "Давайте общаться вежливо."
            }

    ctx = MockCtx()
    selector = LLMSelector()
    dm = DialogManager(ctx, selector)
    
    chat_id = "test_user_999"
    db_path = os.path.join(ctx.base_path, "sessions.db")

    # 2. Очистка и подготовка тестовой истории в БД
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM chat_history WHERE chat_id = ?", (chat_id,))
        # Имитируем запрос 5 минут назад
        past_time = (datetime.datetime.now() - datetime.timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')
        conn.execute(
            "INSERT INTO chat_history (chat_id, message_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
            (chat_id, "100", "user", "Ищу айфон для работы", past_time)
        )
        conn.execute(
            "INSERT INTO chat_history (chat_id, message_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
            (chat_id, "101", "assistant", "У нас есть модели 13 и 14 серии. Какой бюджет?", past_time)
        )

    # 3. Тестовые сценарии
    test_queries = [
        "А есть в черном цвете?", # Должен понять из контекста, что речь об iPhone
        "Сколько стоит доставка?", # GENERAL_FAQ / CHAT
        "Ты любишь жареных гвоздей?", # TROLL / OTHER
        "Покажи 13-й до 20000" # PRODUCT_QUERY с новыми фильтрами
    ]

    for query in test_queries:
        print(f"\nЗапрос: {query}")
        # Анализ интента (включает Flow Up внутри DialogManager)
        result = await dm.analyze_intent(query, chat_id)
        
        print(f"ACTION: {result.get('action')}")
        print(f"ENTITIES: {json.dumps(result.get('entities'), ensure_ascii=False)}")
        if result.get('action') == "SEARCH":
            print("✅ Router определил потребность в поиске.")
        elif result.get('action') == "TROLL":
            print("🚫 Сработал фильтр троллинга.")

if __name__ == "__main__":
    asyncio.run(test_flow_up_logic())