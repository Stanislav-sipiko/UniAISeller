# /root/ukrsell_v4/scripts/create_faiss_index.py v2.0.0
import sqlite3
import faiss
import json
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

def build_index(slug):
    base_path = f"/root/ukrsell_v4/stores/{slug}"
    db_path = os.path.join(base_path, "products.db")
    index_path = os.path.join(base_path, "faiss.index")
    map_path = os.path.join(base_path, "id_map.json")

    if not os.path.exists(db_path):
        print(f"[-] DB not found: {db_path}")
        return

    # 1. Загрузка данных из SQL
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT product_id, title, category_ukr, brand, search_blob, attributes FROM products")
    rows = cur.fetchall()
    conn.close()

    p_ids = []
    ordered_passages = []
    products_dict = {}

    print(f"[*] Обработка {len(rows)} записей для E5...")

    for row in rows:
        pid, title, cat, brand, blob, attrs = row
        p_ids.append(pid)
        
        # Восстанавливаем dict для id_map
        products_dict[pid] = {
            "product_id": pid,
            "title": title,
            "category_ukr": cat,
            "brand": brand,
            "attributes": json.loads(attrs)
        }

        # Формируем passage для E5 (как в v7.1)
        # Используем очищенный blob и метаданные для усиления
        content = f"passage: [id: {pid}] [cat: {cat}] [brand: {brand}] {blob}"
        ordered_passages.append(content)

    # 2. Генерация эмбеддингов
    print(f"[*] Кодирование через intfloat/multilingual-e5-small...")
    model = SentenceTransformer('intfloat/multilingual-e5-small')
    embeddings = model.encode(ordered_passages, normalize_embeddings=True, batch_size=64)
    embeddings_np = np.array(embeddings).astype('float32')

    # 3. Сборка FAISS (IndexFlatIP для нормализованных E5)
    print("[*] Сборка IndexFlatIP...")
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)

    # 4. Сохранение
    faiss.write_index(index, index_path)
    
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump({
            "ids": p_ids,
            "products": products_dict,
            "total_items": len(p_ids),
            "model": "multilingual-e5-small"
        }, f, ensure_ascii=False, indent=2)

    print(f"[+] Готово. Индекс: {index_path}, Маппинг: {map_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 create_faiss_index.py <slug>")
    else:
        build_index(sys.argv[1])