# /root/ukrsell_v4/scripts/upload_to_sql.py v1.3.0
import json
import sqlite3
import os

# Полный маппинг на основе твоего списка
CATEGORY_MAP = {
    "Grooming": "Грумінг",
    "Feeding": "Годування",
    "Apparel": "Одяг",
    "Care": "Догляд",
    "Walking": "Прогулянки",
    "Toys": "Іграшки",
    "Meds": "Ветаптека",
    "Beds & Furniture": "Лежаки та меблі",
    "Other": "Різне"
}

def migrate_to_sql(slug):
    store_dir = f"/root/ukrsell_v4/stores/{slug}"
    json_path = os.path.join(store_dir, "deduplicated_products.json")
    db_path = os.path.join(store_dir, "products.db")

    if not os.path.exists(json_path):
        print(f"[-] Файл {json_path} не найден.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        products = json.load(f)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Создаем таблицу с правильной структурой
    cur.execute('''
        CREATE TABLE IF NOT EXISTS products (
            product_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            category TEXT,
            category_ukr TEXT,
            subtype TEXT,
            brand TEXT,
            animal TEXT,
            price_min REAL,
            price_max REAL,
            image_url TEXT,
            search_blob TEXT,
            attributes TEXT,
            variants TEXT
        )
    ''')
    
    cur.execute("DELETE FROM products")

    rows = []
    for p in products:
        eng_cat = p.get('category', 'Other')
        # Если категории нет в маппинге, оставляем как есть (защита от пропусков)
        ukr_cat = CATEGORY_MAP.get(eng_cat, eng_cat)
        
        rows.append((
            p.get('product_id'),
            p.get('title'),
            p.get('category'),
            ukr_cat,
            p.get('subtype'),
            p.get('brand'),
            p.get('animal'),
            p.get('price_min'),
            p.get('price_max'),
            p.get('image_url'),
            p.get('search_blob'),
            json.dumps(p.get('attributes', {}), ensure_ascii=False),
            json.dumps(p.get('variants', []), ensure_ascii=False)
        ))

    cur.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    print(f"[+] База {db_path} готова. Загружено {len(rows)} товаров.")

if __name__ == "__main__":
    migrate_to_sql("luckydog")