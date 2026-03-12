# /root/ukrsell_v4/scripts/final_chain_test.py v7.5.0
import asyncio
import os
import json
import sqlite3
import datetime
import sys

# Настройка путей
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.store_context import StoreContext
from core.llm_selector import LLMSelector
from core.retrieval import RetrievalEngine
from core.analyzer import Analyzer
from core.dialog_manager import DialogManager
from kernel import UkrSellKernel

async def run_final_test():
    # Используем large-модель для соответствия индексу (1024d)
    MODEL_NAME = 'intfloat/multilingual-e5-large-instruct'
    STORE_SLUG = "luckydog"
    STORE_PATH = f"/root/ukrsell_v4/stores/{STORE_SLUG}"
    
    print(f"🛠️ Запуск финального теста цепочки v7.5.0 (Store: {STORE_SLUG})...")
    
    # Инициализируем ядро и селектор
    kernel = UkrSellKernel(model_name=MODEL_NAME)
    await kernel.selector.ensure_ready()
    
    # Инициализируем реальный контекст вместо Mock
    ctx = StoreContext(
        base_path=STORE_PATH,
        db_engine=None, # Для теста сессий используем прямой путь к sqlite ниже
        llm_selector=kernel.selector,
        kernel=kernel
    )
    
    print(f"--- Шаг 0: Инициализация контекста и профиля ---")
    if not await ctx.initialize():
        print("❌ Ошибка инициализации контекста. Проверьте наличие normalized_products_final.json")
        return

    # RetrievalEngine получает реальный проинициализированный контекст
    retrieval = RetrievalEngine(
        ctx=ctx, 
        shared_model=kernel.model, 
        shared_translator=kernel.translator
    )
    
    dm = DialogManager(ctx, kernel.selector)
    analyzer = Analyzer(ctx)
    
    chat_id = "test_user_v7"
    db_path = os.path.join(ctx.base_path, "sessions.db")

    # Подготовка базы данных сессий для теста Flow Up
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS chat_history")
        conn.execute("""
            CREATE TABLE chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT,
                message_id TEXT,
                role TEXT,
                content TEXT,
                timestamp DATETIME
            )
        """)
        past_time = (datetime.datetime.now() - datetime.timedelta(minutes=2)).strftime('%Y-%m-%d %H:%M:%S')
        
        # Эмулируем контекст: пользователь ранее спрашивал про корм
        conn.execute(
            "INSERT INTO chat_history (chat_id, message_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
            (chat_id, "201", "user", "Я шукаю сухий корм для дорослих собак", past_time)
        )
        conn.execute(
            "INSERT INTO chat_history (chat_id, message_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
            (chat_id, "202", "assistant", "У нас є багато варіантів. Якої ваги упаковка вас цікавить?", past_time)
        )

    # Тестовый запрос, требующий извлечения сущностей и фильтрации
    user_query = "А є з ягням до 2000 грн?"
    print(f"\n👉 Запрос пользователя: {user_query}")

    # ШАГ 1: Router (DialogManager)
    print("\n--- Шаг 1: Router (Context Analysis) ---")
    decision = await dm.analyze_intent(user_query, chat_id)
    print(f"Action: {decision.get('action')}")
    print(f"Detected Entities: {json.dumps(decision.get('entities'), ensure_ascii=False, indent=2)}")

    # ШАГ 2: Retrieval
    print("\n--- Шаг 2: Retrieval (Hybrid Search & Filters) ---")
    search_results = await retrieval.search(user_query, entities=decision.get('entities'))
    print(f"Статус: {search_results.get('status')}")
    print(f"Найдено товаров: {len(search_results.get('products', []))}")
    print(f"Примененные фильтры: {search_results.get('applied_filters')}")

    # ШАГ 3: Analyzer
    print("\n--- Шаг 3: Analyzer (Synthesis) ---")
    # Эмулируем структуру данных, которую ждет Analyzer
    final_output = analyzer.synthesize_response(
        search_results=search_results.get("products", []),
        entities=decision.get('entities'),
        user_query=user_query
    )

    print("\n✅ ФИНАЛЬНЫЙ РЕЗУЛЬТАТ СИНТЕЗА:")
    print("=" * 60)
    print(final_output.get("text", "No text generated"))
    print("=" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(run_final_test())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ Тест провален: {e}")
        import traceback
        traceback.print_exc()