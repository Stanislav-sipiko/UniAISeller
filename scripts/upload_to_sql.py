# /root/ukrsell_v4/scripts/upload_to_sql.py v2.1.0
"""
Загрузка товаров в products.db и построение faiss_map.

Changelog v2.1.0:
  - Источник изменён: deduplicated_products.json вместо normalized_products_final.json.
    deduplicated_products.json — финальный артефакт пайплайна, уже содержит
    все нужные поля (search_blob, attributes как dict, price_min/max, variants с URL).
    normalized_products_final.json не содержит полей которых нет в deduplicated.

Changelog v2.0.0:
  - Схема "Flat Power": availability, search_metadata, last_updated.
  - faiss_map из id_map.json — заменяет JSON в runtime retrieval.
  - INSERT OR REPLACE (upsert) — без downtime.
  - Батчевая загрузка по BATCH_SIZE строк.
  - PRAGMA WAL + synchronous=OFF на время bulk insert.
  - attributes нормализуется в dict при загрузке.
  - SAVEPOINT вместо BEGIN/COMMIT — нет вложенных транзакций.
"""

import json
import sqlite3
import os
import sys
import argparse
import datetime
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────────────────────

BASE_DIR   = "/root/ukrsell_v4"
BATCH_SIZE = 500  # строк за одну транзакцию

CATEGORY_MAP: Dict[str, str] = {
    "Grooming":       "Грумінг",
    "Feeding":        "Годування",
    "Apparel":        "Одяг",
    "Care":           "Догляд",
    "Walking":        "Прогулянки",
    "Toys":           "Іграшки",
    "Meds":           "Ветаптека",
    "Beds & Furniture": "Лежаки та меблі",
    "Other":          "Різне",
}


# ─────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────

def _normalize_attributes(raw: Any) -> str:
    """
    Нормализует поле attributes в JSON-строку dict.
    Если пришла строка — парсим. Если dict — сериализуем. Иначе — пустой dict.
    Вызывается один раз при загрузке, чтобы в БД всегда лежал валидный JSON.
    """
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return json.dumps(parsed, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            pass
    return "{}"


def _normalize_variants(raw: Any) -> str:
    """Нормализует поле variants в JSON-строку list."""
    if isinstance(raw, list):
        return json.dumps(raw, ensure_ascii=False)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return json.dumps(parsed, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            pass
    return "[]"


def _normalize_search_metadata(raw: Any) -> Optional[str]:
    """
    Нормализует search_metadata в JSON-строку dict или None.
    Используется для хранения синонимов, тегов, версии эмбеддинга.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return json.dumps(parsed, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _build_product_row(p: Dict[str, Any]) -> Optional[Tuple]:
    """
    Строит кортеж для INSERT из одного продукта.
    Возвращает None если product_id отсутствует или невалиден.
    """
    raw_id = p.get("product_id") or p.get("id")
    if raw_id is None or str(raw_id).strip().lower() == "none":
        return None

    product_id = str(raw_id).strip()
    eng_cat    = p.get("category", "Other") or "Other"
    ukr_cat    = CATEGORY_MAP.get(eng_cat, eng_cat)
    now        = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return (
        product_id,
        p.get("title") or "",
        eng_cat,
        ukr_cat,
        p.get("subtype"),
        p.get("brand"),
        p.get("animal"),
        p.get("price_min"),
        p.get("price_max"),
        p.get("image_url"),
        p.get("search_blob"),
        _normalize_attributes(p.get("attributes")),
        _normalize_variants(p.get("variants")),
        p.get("availability", "instock") or "instock",
        _normalize_search_metadata(p.get("search_metadata")),
        now,
    )


def _insert_products_batch(
    conn: sqlite3.Connection,
    rows: List[Tuple],
) -> None:
    """
    Вставляет батч строк через INSERT OR REPLACE (upsert).
    Существующие продукты обновляются, новые вставляются.
    Не требует DELETE перед загрузкой — нет downtime.
    """
    conn.executemany(
        """
        INSERT OR REPLACE INTO products (
            product_id, title, category, category_ukr, subtype,
            brand, animal, price_min, price_max, image_url,
            search_blob, attributes, variants,
            availability, search_metadata, last_updated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _rebuild_faiss_map(
    conn: sqlite3.Connection,
    id_map_path: str,
    slug: str,
) -> int:
    """
    Перестраивает таблицу faiss_map из id_map.json.

    faiss_map хранит соответствие position → product_id,
    где position — порядковый номер вектора в faiss.index.
    Это критично для корректного поиска: self.id_list[idx] в retrieval.py
    должен соответствовать position в faiss_map.

    Возвращает количество загруженных маппингов.
    """
    if not os.path.exists(id_map_path):
        print(f"[!] [{slug}] id_map.json not found at {id_map_path}, skipping faiss_map rebuild.")
        return 0

    with open(id_map_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # id_map.json может быть {"ids": [...]} или просто списком
    if isinstance(data, dict):
        ids = data.get("ids", [])
    elif isinstance(data, list):
        ids = data
    else:
        print(f"[!] [{slug}] id_map.json has unexpected format, skipping faiss_map rebuild.")
        return 0

    # Фильтруем невалидные ID
    valid_ids = [
        str(x).strip() for x in ids
        if x is not None and str(x).strip().lower() != "none"
    ]

    if not valid_ids:
        print(f"[!] [{slug}] id_map.json contains no valid IDs.")
        return 0

    # Перестраиваем faiss_map полностью: DELETE + INSERT батчами
    conn.execute("DELETE FROM faiss_map")

    faiss_rows = [(pos, pid) for pos, pid in enumerate(valid_ids)]

    for i in range(0, len(faiss_rows), BATCH_SIZE):
        batch = faiss_rows[i : i + BATCH_SIZE]
        conn.executemany(
            "INSERT INTO faiss_map (position, product_id) VALUES (?, ?)",
            batch,
        )

    return len(valid_ids)


# ─────────────────────────────────────────────────────────────
# Основная функция
# ─────────────────────────────────────────────────────────────

def upload_to_sql(slug: str, db_path: Optional[str] = None) -> bool:
    """
    Загружает normalized_products_final.json в products.db и перестраивает faiss_map.

    Аргументы:
      slug    — идентификатор магазина (папка в stores/).
      db_path — опциональный путь к БД (по умолчанию stores/{slug}/products.db).

    Возвращает True при успехе, False при ошибке.
    """
    store_dir   = os.path.join(BASE_DIR, "stores", slug)
    json_path   = os.path.join(store_dir, "deduplicated_products.json")
    id_map_path = os.path.join(store_dir, "id_map.json")

    if db_path is None:
        db_path = os.path.join(store_dir, "products.db")

    # ── Валидация входных данных ──────────────────────────────

    if not os.path.exists(store_dir):
        print(f"[-] Store directory not found: {store_dir}")
        return False

    if not os.path.exists(json_path):
        print(f"[-] Source file not found: {json_path}")
        print(f"    Run deduplicate.py first.")
        return False

    # ── Загрузка источника ────────────────────────────────────

    print(f"[*] [{slug}] Loading {json_path} ...")
    with open(json_path, "r", encoding="utf-8") as f:
        products_raw = json.load(f)

    if not isinstance(products_raw, list):
        print(f"[-] [{slug}] deduplicated_products.json must be a JSON array.")
        return False

    print(f"[*] [{slug}] Loaded {len(products_raw)} products from source.")

    # ── Построение строк для вставки ──────────────────────────

    rows: List[Tuple] = []
    skipped = 0
    for p in products_raw:
        row = _build_product_row(p)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    if skipped:
        print(f"[!] [{slug}] Skipped {skipped} products with invalid/missing product_id.")

    if not rows:
        print(f"[-] [{slug}] No valid products to insert.")
        return False

    # ── Запись в БД ───────────────────────────────────────────

    print(f"[*] [{slug}] Connecting to {db_path} ...")
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        # PRAGMA для ускорения bulk insert.
        # synchronous=OFF безопасен здесь: если процесс упадёт во время записи,
        # мы просто перезапустим скрипт. Данные не критичны для durability.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA cache_size = -32000")  # 32 MB page cache

        # Убеждаемся что таблицы существуют (на случай если миграции не запускались)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id      TEXT    PRIMARY KEY,
                title           TEXT    NOT NULL,
                category        TEXT,
                category_ukr    TEXT,
                subtype         TEXT,
                brand           TEXT,
                animal          TEXT,
                price_min       REAL,
                price_max       REAL,
                image_url       TEXT,
                search_blob     TEXT,
                attributes      TEXT,
                variants        TEXT,
                availability    TEXT    DEFAULT 'instock',
                search_metadata TEXT,
                last_updated    DATETIME DEFAULT (DATETIME('now', 'utc'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS faiss_map (
                position   INTEGER PRIMARY KEY,
                product_id TEXT    NOT NULL,
                FOREIGN KEY (product_id) REFERENCES products (product_id)
                    ON DELETE CASCADE
            )
        """)

        # Батчевая загрузка продуктов.
        # SAVEPOINT вместо BEGIN/COMMIT: поддерживает вложенность, не конфликтует
        # с autocommit-режимом sqlite3 модуля Python. Каждый батч — отдельная точка
        # отката. При ошибке откатывается только упавший батч, остальные сохраняются.
        total_inserted = 0
        total_batches  = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE

        print(
            f"[*] [{slug}] Inserting {len(rows)} products "
            f"in {total_batches} batch(es) of {BATCH_SIZE} ..."
        )

        for batch_num, i in enumerate(range(0, len(rows), BATCH_SIZE), start=1):
            batch     = rows[i : i + BATCH_SIZE]
            savepoint = f"sp_batch_{batch_num}"
            try:
                conn.execute(f"SAVEPOINT {savepoint}")
                _insert_products_batch(conn, batch)
                conn.execute(f"RELEASE {savepoint}")
                total_inserted += len(batch)
                print(
                    f"    Batch {batch_num}/{total_batches}: "
                    f"+{len(batch)} rows (total: {total_inserted})"
                )
            except Exception as e:
                conn.execute(f"ROLLBACK TO {savepoint}")
                conn.execute(f"RELEASE {savepoint}")
                print(f"[-] [{slug}] Batch {batch_num} failed, rolled back: {e}")
                raise

        # Перестройка faiss_map — отдельный SAVEPOINT, независим от батчей продуктов
        print(f"[*] [{slug}] Rebuilding faiss_map from id_map.json ...")
        try:
            conn.execute("SAVEPOINT sp_faiss")
            mapped = _rebuild_faiss_map(conn, id_map_path, slug)
            conn.execute("RELEASE sp_faiss")
            if mapped:
                print(f"[+] [{slug}] faiss_map: {mapped} entries loaded.")
        except Exception as e:
            conn.execute("ROLLBACK TO sp_faiss")
            conn.execute("RELEASE sp_faiss")
            print(f"[-] [{slug}] faiss_map rebuild failed, rolled back: {e}")
            raise

        # Индексы
        print(f"[*] [{slug}] Rebuilding indexes ...")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_products_category "
            "ON products (category)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_products_brand "
            "ON products (brand)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_products_price "
            "ON products (price_min, price_max)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_products_availability "
            "ON products (availability)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_faiss_product "
            "ON faiss_map (product_id)"
        )
        conn.commit()

        # Сброс PRAGMA обратно для безопасной работы в runtime
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.commit()

        print(
            f"[+] [{slug}] Upload complete: "
            f"{total_inserted} products, {mapped if mapped else 0} FAISS mappings."
        )
        return True

    except Exception as e:
        print(f"[-] [{slug}] Upload failed: {e}")
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        return False
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload normalized_products_final.json to products.db (Flat Power schema)."
    )
    parser.add_argument(
        "slug",
        nargs="?",
        default="luckydog",
        help="Store slug (default: luckydog)",
    )
    parser.add_argument(
        "--db",
        dest="db_path",
        default=None,
        help="Optional custom path to products.db",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run upload for all stores found in stores/ directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.all:
        stores_dir = os.path.join(BASE_DIR, "stores")
        slugs = [
            d for d in os.listdir(stores_dir)
            if os.path.isdir(os.path.join(stores_dir, d))
        ]
        print(f"[*] Running upload for all stores: {slugs}")
        failed = []
        for s in slugs:
            ok = upload_to_sql(s)
            if not ok:
                failed.append(s)
        if failed:
            print(f"[-] Failed stores: {failed}")
            sys.exit(1)
        else:
            print("[+] All stores uploaded successfully.")
    else:
        ok = upload_to_sql(args.slug, db_path=args.db_path)
        sys.exit(0 if ok else 1)