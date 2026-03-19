# /root/ukrsell_v4/db_migrations.py v2.1.1
"""
Централизованный модуль миграций БД v2.1.1.

Единственное место где определяются схемы таблиц для всех БД магазина.
Вызывается из main.py после kernel.initialize().

Changelog v2.1.1:
  - Удалены дублирующиеся определения MIGRATIONS, run_migrations,
    run_all_migrations и вспомогательных функций.
  - Добавлен if __name__ == "__main__" блок для прямого запуска из CLI.
  - DB_VERSION увеличен до 4.

Changelog v2.1.0:
  - Добавлена таблица store_meta в products.db.
  - DB_VERSION увеличен до 3.

Changelog v2.0.0:
  - Добавлена products.db в периметр миграций.
  - Добавлена таблица faiss_map в products.db.
  - Добавлены колонки: availability, search_metadata, last_updated в products.
  - Версия схемы сохраняется через user_version PRAGMA.

Добавление новой таблицы / колонки:
  1. Добавить новую запись в MIGRATIONS с нужным db_file и sql.
  2. Увеличить DB_VERSION.
  3. Перезапустить — модуль применит только новые миграции ко всем магазинам.
"""

import argparse
import os
import sqlite3
import sys
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────
# Импорт BASE_DIR — с fallback если запускается вне kernel
# ─────────────────────────────────────────────────────────────
try:
    from core.logger import logger
    from core.config import BASE_DIR
except ImportError:
    import logging
    logger = logging.getLogger("db_migrations")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
# Версия схемы.
# Сохраняется в каждой БД через PRAGMA user_version.
# Увеличивать при каждом изменении структуры таблиц.
# ─────────────────────────────────────────────────────────────
DB_VERSION = 4

# ─────────────────────────────────────────────────────────────
# СПИСОК МИГРАЦИЙ
# Каждая запись: (db_file, sql_statement)
# Порядок важен — миграции применяются последовательно.
# CREATE TABLE IF NOT EXISTS идемпотентен.
# ALTER TABLE ADD COLUMN бросает "duplicate column name" если колонка уже есть —
# это перехватывается и игнорируется в run_migrations().
# ─────────────────────────────────────────────────────────────
MIGRATIONS: List[Tuple[str, str]] = [

    # ══════════════════════════════════════════════════════════
    # products.db — основная БД товаров, FAISS-маппинга и мета-профиля
    # ══════════════════════════════════════════════════════════

    # Базовая схема products (создаётся если не существует)
    ("products.db", """
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
    """),

    # Добавляем новые колонки в уже существующие таблицы.
    # SQLite не поддерживает ADD COLUMN IF NOT EXISTS напрямую —
    # ошибка "duplicate column name" перехватывается и игнорируется в run_migrations().
    ("products.db",
        "ALTER TABLE products ADD COLUMN availability TEXT DEFAULT 'instock'"
    ),
    ("products.db",
        "ALTER TABLE products ADD COLUMN search_metadata TEXT"
    ),
    ("products.db",
        "ALTER TABLE products ADD COLUMN last_updated DATETIME"
    ),

    # Индексы для products — ускоряют фильтрацию
    ("products.db",
        "CREATE INDEX IF NOT EXISTS idx_products_category ON products (category)"
    ),
    ("products.db",
        "CREATE INDEX IF NOT EXISTS idx_products_brand ON products (brand)"
    ),
    ("products.db",
        "CREATE INDEX IF NOT EXISTS idx_products_price ON products (price_min, price_max)"
    ),
    ("products.db",
        "CREATE INDEX IF NOT EXISTS idx_products_availability ON products (availability)"
    ),

    # faiss_map — маппинг позиций FAISS-индекса → product_id.
    # Заменяет id_map.json в runtime retrieval.
    # position соответствует порядку строк при построении FAISS-индекса.
    ("products.db", """
        CREATE TABLE IF NOT EXISTS faiss_map (
            position   INTEGER PRIMARY KEY,
            product_id TEXT    NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products (product_id)
                ON DELETE CASCADE
        )
    """),

    ("products.db",
        "CREATE INDEX IF NOT EXISTS idx_faiss_product ON faiss_map (product_id)"
    ),

    # store_meta — агрегированный профиль магазина.
    # Заменяет store_profile.json в runtime.
    # Ключи (поле key):
    #   schema_keys        — JSON-массив строк, значимые ключи атрибутов
    #   total_sku          — INTEGER, общее количество товаров
    #   main_category      — TEXT, самая популярная категория
    #   store_type         — TEXT, "mono_vertical" | "multi_category"
    #   top_brands         — JSON-массив строк, топ-10 брендов
    #   price_min          — REAL
    #   price_max          — REAL
    #   price_avg          — REAL
    #   ai_welcome_message — TEXT, LLM-сгенерированное приветствие
    #   language           — TEXT, "Ukrainian" | "Russian"
    #   last_data_update   — DATETIME ISO-8601
    ("products.db", """
        CREATE TABLE IF NOT EXISTS store_meta (
            key        TEXT     PRIMARY KEY,
            value      TEXT     NOT NULL,
            updated_at DATETIME DEFAULT (DATETIME('now', 'utc'))
        )
    """),

    ("products.db",
        "CREATE INDEX IF NOT EXISTS idx_store_meta_key ON store_meta (key)"
    ),

    # ══════════════════════════════════════════════════════════
    # sessions.db — история чата
    # ══════════════════════════════════════════════════════════

    ("sessions.db", """
        CREATE TABLE IF NOT EXISTS chat_history (
            id          INTEGER  PRIMARY KEY AUTOINCREMENT,
            chat_id     TEXT     NOT NULL,
            message_id  TEXT,
            role        TEXT     NOT NULL,
            content     TEXT     NOT NULL,
            timestamp   DATETIME DEFAULT (DATETIME('now', 'utc'))
        )
    """),

    ("sessions.db",
        "CREATE INDEX IF NOT EXISTS idx_chat_time ON chat_history (chat_id, timestamp)"
    ),

    # ══════════════════════════════════════════════════════════
    # users.db — статистика пользователей
    # ══════════════════════════════════════════════════════════

    ("users.db", """
        CREATE TABLE IF NOT EXISTS users_stats (
            id            INTEGER  PRIMARY KEY AUTOINCREMENT,
            user_id       TEXT,
            username      TEXT,
            query         TEXT,
            response_type TEXT,
            category      TEXT,
            timestamp     DATETIME DEFAULT (DATETIME('now', 'utc'))
        )
    """),
]


# ─────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────

def _get_user_version(conn: sqlite3.Connection) -> int:
    """Читает текущую версию схемы из БД."""
    return conn.execute("PRAGMA user_version").fetchone()[0]


def _set_user_version(conn: sqlite3.Connection, version: int) -> None:
    """Сохраняет версию схемы в БД. PRAGMA не поддерживает параметры — форматируем вручную."""
    conn.execute(f"PRAGMA user_version = {int(version)}")


def _apply_pragma_settings(conn: sqlite3.Connection) -> None:
    """Базовые настройки производительности и надёжности."""
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")


# ─────────────────────────────────────────────────────────────
# Основные функции миграций
# ─────────────────────────────────────────────────────────────

def run_migrations(base_path: str, slug: str) -> bool:
    """
    Применяет все миграции к БД одного магазина.

    Логика версионирования:
      - Читает PRAGMA user_version из каждой БД.
      - Если current_version < DB_VERSION — применяет все миграции для этой БД.
      - После успеха обновляет user_version до DB_VERSION.
      - Ошибка "duplicate column name" от ALTER TABLE игнорируется —
        это нормальная идемпотентность при повторном запуске.

    Возвращает True если все БД обновлены успешно, False при критической ошибке.
    """
    if not os.path.exists(base_path):
        logger.warning(f"[{slug}] Migration skipped: store path not found: {base_path}")
        return False

    # Группируем миграции по файлу БД, сохраняя порядок
    by_db: Dict[str, List[str]] = {}
    for db_file, sql in MIGRATIONS:
        by_db.setdefault(db_file, []).append(sql)

    overall_success = True

    for db_file, statements in by_db.items():
        db_path = os.path.join(base_path, db_file)
        try:
            with sqlite3.connect(db_path, timeout=15.0) as conn:
                _apply_pragma_settings(conn)

                current_version = _get_user_version(conn)
                if current_version >= DB_VERSION:
                    logger.debug(
                        f"[{slug}] {db_file}: schema v{current_version} is current, skipping."
                    )
                    continue

                logger.info(
                    f"[{slug}] {db_file}: upgrading schema "
                    f"v{current_version} → v{DB_VERSION} ..."
                )

                for sql in statements:
                    try:
                        conn.execute(sql)
                    except sqlite3.OperationalError as e:
                        # ALTER TABLE ADD COLUMN бросает OperationalError если колонка есть.
                        # Это штатная идемпотентность — игнорируем, продолжаем.
                        if "duplicate column name" in str(e).lower():
                            logger.debug(
                                f"[{slug}] {db_file}: column already exists, skipping: {e}"
                            )
                            continue
                        # Любая другая OperationalError — реальная проблема
                        raise

                _set_user_version(conn, DB_VERSION)
                conn.commit()
                logger.info(
                    f"[{slug}] {db_file}: schema upgraded to v{DB_VERSION} successfully."
                )

        except Exception as e:
            logger.error(f"[{slug}] Migration failed for {db_file}: {e}", exc_info=True)
            overall_success = False

    return overall_success


def run_all_migrations(kernel_instance) -> None:
    """
    Применяет миграции ко всем активным магазинам.
    Вызывается из main.py после kernel.initialize().
    """
    active_slugs = kernel_instance.get_all_active_slugs()
    logger.info(
        f"Running DB migrations v{DB_VERSION} for {len(active_slugs)} store(s)..."
    )

    failed = []
    for slug in active_slugs:
        base_path = os.path.join(BASE_DIR, "stores", slug)
        if not run_migrations(base_path, slug):
            failed.append(slug)

    if failed:
        logger.error(f"Migration failed for stores: {failed}")
    else:
        logger.info("All DB migrations completed successfully.")


# ─────────────────────────────────────────────────────────────
# CLI — прямой запуск: python3 db_migrations.py [slug|--all]
# ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run DB migrations v{DB_VERSION} for UkrSell stores."
    )
    parser.add_argument(
        "slug",
        nargs="?",
        default=None,
        help="Store slug to migrate (e.g. luckydog). Omit with --all to migrate all stores.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run migrations for all stores found in stores/ directory.",
    )
    parser.add_argument(
        "--base-dir",
        dest="base_dir",
        default=None,
        help=f"Override base directory (default: {BASE_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    base_dir = args.base_dir or BASE_DIR
    stores_dir = os.path.join(base_dir, "stores")

    if args.all:
        if not os.path.isdir(stores_dir):
            print(f"[-] stores/ directory not found: {stores_dir}")
            sys.exit(1)
        slugs = [
            d for d in os.listdir(stores_dir)
            if os.path.isdir(os.path.join(stores_dir, d))
        ]
        if not slugs:
            print(f"[-] No stores found in {stores_dir}")
            sys.exit(1)
        print(f"[*] Running migrations for all stores: {slugs}")
        failed = []
        for s in slugs:
            bp = os.path.join(base_dir, "stores", s)
            ok = run_migrations(bp, s)
            if not ok:
                failed.append(s)
        if failed:
            print(f"[-] Failed: {failed}")
            sys.exit(1)
        else:
            print("[+] All stores migrated successfully.")
            sys.exit(0)

    elif args.slug:
        bp = os.path.join(base_dir, "stores", args.slug)
        ok = run_migrations(bp, args.slug)
        sys.exit(0 if ok else 1)

    else:
        # Без аргументов — мигрируем все найденные магазины
        if not os.path.isdir(stores_dir):
            print(f"[-] stores/ directory not found: {stores_dir}")
            sys.exit(1)
        slugs = [
            d for d in os.listdir(stores_dir)
            if os.path.isdir(os.path.join(stores_dir, d))
        ]
        if not slugs:
            print(f"[-] No stores found in {stores_dir}")
            sys.exit(1)
        print(f"[*] No slug specified. Running migrations for all stores: {slugs}")
        failed = []
        for s in slugs:
            bp = os.path.join(base_dir, "stores", s)
            ok = run_migrations(bp, s)
            if not ok:
                failed.append(s)
        if failed:
            print(f"[-] Failed: {failed}")
            sys.exit(1)
        else:
            print("[+] Done.")
            sys.exit(0)