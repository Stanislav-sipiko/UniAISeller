# /root/ukrsell_v4/db_migrations.py
"""
Централизованный модуль миграций БД v1.0.0.

Единственное место где определяются схемы таблиц.
Вызывается из main.py после kernel.initialize().

Добавление новой таблицы/колонки:
  1. Добавить CREATE TABLE или ALTER TABLE в MIGRATIONS.
  2. Увеличить DB_VERSION.
  3. Больше ничего менять не нужно — модуль применит изменения
     ко всем активным магазинам при следующем старте.
"""
import os
import sqlite3
from core.logger import logger

# Версия схемы. Увеличивать при каждом изменении структуры таблиц.
DB_VERSION = 1

# ─────────────────────────────────────────────────────────────
# СХЕМЫ ТАБЛИЦ
# Каждая запись: (db_file, sql_statement)
# db_file: "sessions.db" или "users.db"
# ─────────────────────────────────────────────────────────────
MIGRATIONS = [

    # ── sessions.db ──────────────────────────────────────────

    ("sessions.db", """
        CREATE TABLE IF NOT EXISTS chat_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id     TEXT    NOT NULL,
            message_id  TEXT,
            role        TEXT    NOT NULL,
            content     TEXT    NOT NULL,
            timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """),

    ("sessions.db",
        "CREATE INDEX IF NOT EXISTS idx_chat_time ON chat_history (chat_id, timestamp)"
    ),

    # ── users.db ─────────────────────────────────────────────

    ("users.db", """
        CREATE TABLE IF NOT EXISTS users_stats (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       TEXT,
            username      TEXT,
            query         TEXT,
            response_type TEXT,
            category      TEXT,
            timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """),
]


def run_migrations(base_path: str, slug: str) -> bool:
    """
    Применяет все миграции к БД одного магазина.
    Возвращает True если всё успешно, False при ошибке.
    """
    if not os.path.exists(base_path):
        logger.warning(f"[{slug}] Migration skipped: store path not found: {base_path}")
        return False

    # Группируем миграции по файлу БД
    by_db: dict[str, list[str]] = {}
    for db_file, sql in MIGRATIONS:
        by_db.setdefault(db_file, []).append(sql)

    success = True
    for db_file, statements in by_db.items():
        db_path = os.path.join(base_path, db_file)
        try:
            with sqlite3.connect(db_path) as conn:
                for sql in statements:
                    conn.execute(sql)
                conn.commit()
        except Exception as e:
            logger.error(f"[{slug}] Migration failed for {db_file}: {e}")
            success = False

    if success:
        logger.info(f"[{slug}] DB migrations v{DB_VERSION} applied successfully.")
    return success


def run_all_migrations(kernel_instance) -> None:
    """
    Применяет миграции ко всем активным магазинам.
    Вызывается из main.py после kernel.initialize().
    """
    from core.config import BASE_DIR

    active_slugs = kernel_instance.get_all_active_slugs()
    logger.info(f"Running DB migrations v{DB_VERSION} for {len(active_slugs)} store(s)...")

    failed = []
    for slug in active_slugs:
        base_path = os.path.join(BASE_DIR, "stores", slug)
        if not run_migrations(base_path, slug):
            failed.append(slug)

    if failed:
        logger.error(f"Migration failed for stores: {failed}")
    else:
        logger.info(f"All DB migrations completed successfully.")
