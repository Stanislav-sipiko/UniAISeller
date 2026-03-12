# /root/ukrsell_v4/scripts/test_kernel_flow.py v5.13.1
import asyncio
import sys
import os
import logging
from pathlib import Path

# Добавляем корень проекта в sys.path для корректного импорта kernel и core
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from kernel import UkrSellKernel
from core.logger import logger

async def test_integration():
    print("\n" + "="*60)
    print("🧪 ЗАПУСК ИНТЕГРАЦИОННОГО ТЕСТА (KERNEL + FAISS + RERANKER)")
    print("="*60)
    
    # 1. Инициализируем ядро
    # Оно автоматически загрузит модели, транслятор и реестр магазинов
    try:
        kernel = UkrSellKernel()
        
        # Ожидаем готовности LLM Selector (важно для генерации интро)
        print("⏳ Инициализация ядра и моделей (может занять до 30 сек)...")
        # В v5.13.0 kernel.llm_ready — это asyncio.Event()
        await asyncio.wait_for(kernel.llm_ready.wait(), timeout=30.0)
        print("✅ Ядро готово к работе.")
        
    except Exception as e:
        print(f"❌ Ошибка при инициализации ядра: {e}")
        return

    # Целевой магазин из структуры /root/ukrsell_v4/stores/
    slug = "phonestore"
    
    # 2. Извлекаем уже готовый engine (и контекст внутри него)
    engine = kernel.registry.get_engine(slug)
    if not engine:
        print(f"❌ Магазин '{slug}' не найден в реестре!")
        print(f"Доступные магазины: {kernel.registry.get_all_slugs()}")
        await kernel.close()
        return
        
    ctx = engine.ctx
    
    # 3. Параметры теста
    # Тестируем запрос, который требует фильтрации и векторного поиска
    query = "Xiaomi з NFC до 12000 грн"
    filters = {"brand": "xiaomi", "price_limit": 12000} 
    
    print(f"\n🔍 Входящий запрос: '{query}'")
    print(f"⚙️ Активные фильтры: {filters}")
    print("-" * 30)
    
    # 4. Вызов пайплайна ядра
    # Метод get_recommendations возвращает готовую строку для отправки пользователю
    try:
        print("📡 Запуск пайплайна (Vector Search -> Rerank -> Synthesis)...")
        formatted_response = await kernel.get_recommendations(
            ctx=ctx, 
            query=query, 
            filters=filters, 
            top_k=3
        )
        
        if not formatted_response:
            print("⚠️ Получен пустой ответ от ядра. Возможно, товары не прошли фильтры.")
        else:
            print("\n✅ СФОРМИРОВАННЫЙ ОТВЕТ ДЛЯ КЛИЕНТА:")
            print("-" * 40)
            print(formatted_response)
            print("-" * 40)
            
        # 5. Дополнительная проверка сырых данных (через внутренний метод)
        # Это поможет понять, работает ли FAISS отдельно от LLM-генерации
        print("\n🛠️ Техническая проверка FAISS (Raw Search Debug):")
        # Кодируем запрос той же моделью, что в ядре
        query_vector = kernel.model.encode(f"query: {query}", normalize_embeddings=True).astype('float32')
        
        # Применяем фильтры вручную для проверки
        allowed_ids = ctx.apply_filters(filters) if hasattr(ctx, 'apply_filters') else None
        print(f"📊 Найдено {len(allowed_ids) if allowed_ids else 0} ID после фильтрации по цене/бренду.")
        
        # Используем внутренний метод поиска ядра
        raw_results = kernel._perform_vector_search(ctx, query_vector, allowed_ids, top_k=3)
        
        if not raw_results:
            print("❌ FAISS не вернул результатов. Проверьте faiss.index и наличие товаров в id_map.json")
        else:
            print(f"🎯 Топ-{len(raw_results)} кандидатов из FAISS:")
            for i, res in enumerate(raw_results, 1):
                p = res['data']
                print(f"   [{i}] ID: {res['id']} | {p.get('name')} | Цена: {p.get('price')} | Score: {res['score']:.4f}")

    except Exception as e:
        print(f"💥 Критическая ошибка во время теста: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Корректно закрываем ресурсы ядра (соединения с LLM API и сессии aiohttp)
        await kernel.close()
        print("\n" + "="*60)
        print("🏁 ТЕСТ ЗАВЕРШЕН")

if __name__ == "__main__":
    # Подавляем лишние логи от sentence_transformers для чистоты вывода
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    
    try:
        asyncio.run(test_integration())
    except KeyboardInterrupt:
        print("\n🛑 Тест прерван пользователем.")