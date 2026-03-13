# /root/ukrsell_v4/main.py v7.8.3
import uvicorn
import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from kernel import UkrSellKernel
from core.logger import logger, log_event
from core.config import BASE_DIR
from db_migrations import run_all_migrations

# Пути к ресурсам
LOG_FILE_PATH = os.path.join(BASE_DIR, "logs", "ukrsell_v4.log")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Системный жизненный цикл приложения.
    v7.8.3: Внедрена инициализация ядра с общей моделью эмбеддингов.
    """
    global kernel
    logger.info("🚀 Starting UkrSell Gateway lifespan sequence (v7.8.3)...")
    try:
        # Инициализация ядра
        kernel = UkrSellKernel()
        logger.info("⚙️ Running async kernel initialization (Models, Registry, LLM Stacks)...")
        
        # Полная асинхронная подготовка всех компонентов
        await kernel.initialize()
        
        # Применяем миграции БД ко всем загруженным магазинам
        run_all_migrations(kernel)
        
        if hasattr(kernel, 'selector'):
            status = kernel.selector.get_status()
            logger.info(f"✅ LLM Pool Ready. Status: {status}")
        
        active_slugs = kernel.get_all_active_slugs()
        logger.info(f"✅ Platform ready. Stores loaded: {len(active_slugs)}")
        
        yield 
    except Exception as e:
        logger.critical(f"FATAL: Failed to launch system kernel: {e}", exc_info=True)
        raise RuntimeError(f"Kernel initialization failed: {e}")
    finally:
        if kernel:
            logger.info("🛑 Shutdown signal received: starting cleanup...")
            await kernel.close()
            logger.info("✅ UkrSell v4 system completely stopped.")


# Инициализация FastAPI с поддержкой lifespan
app = FastAPI(title="UkrSell v4 Single Gateway", version="7.8.3", lifespan=lifespan)
kernel: UkrSellKernel = None


# --- Секция управления логами (Admin UI) ---

@app.get("/logs/view", response_class=HTMLResponse)
async def view_logs_page():
    content = "Log file is empty or not found."
    file_size = "0 KB"
    if os.path.exists(LOG_FILE_PATH):
        try:
            size_bytes = os.path.getsize(LOG_FILE_PATH)
            file_size = f"{size_bytes / 1024:.2f} KB"
            with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
                content = "".join(lines[-1000:]) # Показываем только хвост лога
        except Exception as e:
            content = f"Error reading log: {e}"

    html_content = f"""
    <html>
        <head>
            <title>UkrSell V4 Monitor</title>
            <style>
                body {{ background: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', sans-serif; padding: 20px; }}
                pre {{ background: #000; padding: 15px; border: 1px solid #30363d; border-radius: 6px; 
                       height: 75vh; overflow-y: auto; white-space: pre-wrap; font-size: 13px; color: #8b949e; }}
                .toolbar {{ margin-bottom: 20px; background: #161b22; padding: 15px; border-radius: 6px; 
                            display: flex; gap: 15px; align-items: center; border: 1px solid #30363d; }}
                .btn {{ padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-weight: 600; 
                       text-decoration: none; color: white; transition: 0.2s; font-size: 14px; }}
                .btn-clear {{ background: #da3633; }}
                .btn-refresh {{ background: #238636; }}
                .info {{ flex-grow: 1; }}
                .info span {{ color: #58a6ff; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="toolbar">
                <div class="info">
                    <h2 style="margin:0; font-size: 18px;">System Logs</h2>
                    <small>Size: <span>{file_size}</span> | Showing last 1000 lines</small>
                </div>
                <a href="/logs/view" class="btn btn-refresh">🔄 Refresh</a>
                <form action="/admin/logs/clear" method="post" style="margin:0">
                    <button type="submit" class="btn btn-clear" onclick="return confirm('Очистить лог?')">🗑 Clear Log</button>
                </form>
            </div>
            <pre id="log-box">{content}</pre>
            <script>
                var bbox = document.getElementById('log-box');
                bbox.scrollTop = bbox.scrollHeight;
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/admin/logs/clear")
async def clear_log_file():
    """Безопасная очистка файла логов."""
    try:
        if os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
                f.write("") 
            logger.info("🧹 Log file was cleared via admin interface.")
        return RedirectResponse(url="/logs/view", status_code=303)
    except Exception as e:
        logger.error(f"Failed to clear log: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Webhook Routing ---

@app.post("/webhook/{slug}")
async def telegram_webhook(slug: str, request: Request, background_tasks: BackgroundTasks):
    """
    Основная точка входа для Telegram Webhooks.
    Маршрутизирует запросы на соответствующие StoreEngine через ядро.
    """
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")

    slug = slug.lower()
    # Проверка активности магазина в реестре
    if slug not in kernel.get_all_active_slugs():
        logger.warning(f"⚠️ Received webhook for unknown/inactive store: {slug}")
        raise HTTPException(status_code=404, detail="Store not found")

    try:
        update = await request.json()
        if not update: 
            return {"status": "empty"}
        
        # Передаем обработку в фоновую задачу, чтобы Telegram не ждал завершения генерации LLM
        background_tasks.add_task(kernel.handle_webhook, slug, update)
        
        log_event("GATEWAY_ROUTED", {
            "slug": slug, 
            "update_id": update.get("update_id"),
            "user_id": update.get("message", {}).get("from", {}).get("id")
        })
        
        return {"status": "accepted", "store": slug}
    except Exception as e:
        logger.error(f"Gateway Routing error for {slug}: {e}")
        return {"status": "error"}


@app.get("/health")
async def health():
    """Endpoint для мониторинга работоспособности сервиса."""
    if kernel is None: 
        return {"status": "initializing"}
    
    active_stores = kernel.get_all_active_slugs()
    llm_status = "READY"
    if hasattr(kernel, 'selector'):
        status_map = kernel.selector.get_status()
        if all(v == "OFFLINE" for v in status_map.values()):
            llm_status = "ALL_LLM_OFFLINE"

    return {
        "status": "online", 
        "version": "7.8.3",
        "llm_status": llm_status,
        "active_stores_count": len(active_stores),
        "active_stores": active_stores
    }


if __name__ == "__main__":
    # Запуск через uvicorn (рекомендуется для продакшена)
    # Настройки хоста и порта могут быть переопределены через ENV переменные
    host = os.environ.get("GATEWAY_HOST", "0.0.0.0")
    port = int(os.environ.get("GATEWAY_PORT", 8000))
    
    uvicorn.run(
        "main:app", 
        host=host, 
        port=port, 
        access_log=False, 
        workers=1 # Ядро UkrSellKernel должно работать в одном процессе для управления кэшем
    )