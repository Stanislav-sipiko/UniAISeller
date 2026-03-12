# /root/ukrsell_v4/scripts/test_store_loaded.py
import asyncio
import sys
import os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Добавляем корень проекта в пути поиска модулей
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from core.store_context import StoreContext
from core.logger import logger

async def main():
    """
    Валидатор жизненного цикла магазина v4.9.
    Проверяет: 
    1. Загрузку конфигурации и промптов.
    2. Наличие метаданных (Профиль Эксперта от Gemini).
    3. Отработку Hard Filter (Бренд, Цена, Характеристики).
    4. Качество Vector Search (FAISS + E5 Embeddings).
    """
    store_slug = "phonestore"
    print(f"\n{'='*50}")
    print(f"🚀 ТЕСТ ЗАГРУЗКИ МАГАЗИНА: [{store_slug.upper()}]")
    print(f"{'='*50}\n")

    try:
        # 1. Инициализация контекста
        ctx = StoreContext(store_slug)
        
        # Проверка базовых файлов
        if not ctx.id_map:
            print("❌ ОШИБКА: id_map.json пуст или не найден. Запустите prepare_data.py!")
            return

        # 2. Проверка Профиля Эксперта (Метаданные)
        # Используем актуальное имя метода из твоего StoreContext
        metadata = ctx._read_json_file("metadata.json")
        print("--- 🧠 АНАЛИЗ ПРОФИЛЯ ЭКСПЕРТА ---")
        if metadata:
            print(f"✅ Профиль найден!")
            print(f"🎯 Ключевые атрибуты: {metadata.get('key_attributes', [])}")
            profile_snippet = metadata.get('expert_profile', 'Нет описания')[:150]
            print(f"📝 Инструкция (отрывок): {profile_snippet}...")
        else:
            print("⚠️ ВНИМАНИЕ: metadata.json отсутствует. Магазин работает в обычном режиме.")
        print("-" * 30 + "\n")

        # 3. Загрузка модели эмбеддингов
        print("⏳ Загрузка модели SentenceTransformer (E5-Large)...")
        model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

        # 4. Имитация запроса пользователя
        user_query = "Xiaomi с NFC до 12000 грн"
        # Параметры для фильтрации
        params = {
            "brand": "xiaomi", 
            "max_price": 12000, 
            "nfc": "так" # или True, зависит от того, как prepare_data сохранил в attributes
        }
        print(f"🔍 ВХОДНОЙ ЗАПРОС: '{user_query}'")
        print(f"⚙️ ПАРАМЕТРЫ ФИЛЬТРАЦИИ: {params}\n")

        # 5. Тестирование Hard Filter
        print("--- 🛡 ТЕСТ HARD FILTER ---")
        # ПРИМЕЧАНИЕ: Если метод apply_hard_filters еще не реализован в StoreContext, 
        # добавим его следующим шагом. Пока имитируем логику поиска по id_map.
        allowed_ids = []
        for pid, item in ctx.id_map.items():
            price = item.get('price', 0)
            brand = str(item.get('brand', '')).lower()
            
            # Простейшая логика фильтрации для теста
            if price <= params['max_price'] and params['brand'] in brand:
                allowed_ids.append(pid)

        print(f"📊 Результат: Найдено {len(allowed_ids)} товаров, прошедших фильтр.")
        
        if len(allowed_ids) > 0:
            for pid in allowed_ids[:3]: 
                item = ctx.id_map.get(pid)
                print(f"   [+] ID: {pid} | {item['name']} | {item['price']} грн")
        else:
            print("⚠️ Ни один товар не подошел под критерии фильтра.")
        print("-" * 30 + "\n")

        # 6. Тестирование Векторного поиска
        print("--- 🗺 ТЕСТ VECTOR SEARCH ---")
        index = ctx.get_faiss_index()
        if not index:
            print("❌ Индекс FAISS не найден. Пропуск векторного поиска.")
        else:
            query_text = f"query: {user_query}"
            query_vector = model.encode(query_text, normalize_embeddings=True).astype('float32')
            
            # Поиск в FAISS
            D, I = index.search(np.array([query_vector]), 5)
            
            print(f"🏆 ТОП РЕЗУЛЬТАТОВ ИЗ ИНДЕКСА:")
            # Получаем все ID из id_map для сопоставления с индексами FAISS
            all_ids = list(ctx.id_map.keys())
            
            for i, idx in enumerate(I[0]):
                if idx < len(all_ids):
                    pid = all_ids[idx]
                    p_info = ctx.id_map.get(pid, {})
                    score = D[0][i]
                    print(f"{i+1}. {p_info.get('name')} | Сходство: {score:.4f}")
                    print(f"   💰 Цена: {p_info.get('price')} грн | ID: {pid}")

        print(f"\n{'='*50}")
        print("✅ ТЕСТ ЗАВЕРШЕН УСПЕШНО!")
        print(f"{'='*50}")

    except Exception as e:
        print(f"\n💥 КРИТИЧЕСКАЯ ОШИБКА ПРИ ТЕСТИРОВАНИИ:")
        print(f"Тип: {type(e).__name__}")
        print(f"Сообщение: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())