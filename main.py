# /root/ukrsell_v4/main.py v6.5.1
import uvicorn
import asyncio
import os
import json
from functools import lru_cache
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from kernel import UkrSellKernel
from core.logger import logger, log_event
from core.config import BASE_DIR
from db_migrations import run_all_migrations

# Путь к логам (синхронизировано с core/logger.py)
LOG_FILE_PATH = os.path.join(BASE_DIR, "logs", "ukrsell_v4.log")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global kernel
    logger.info("🚀 Starting UkrSell Gateway lifespan sequence (v6.5.1)...")
    try:
        kernel = UkrSellKernel()
        logger.info("⚙️ Running async kernel initialization...")
        await kernel.initialize()
        
        # Применяем миграции БД ко всем магазинам
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

app = FastAPI(title="UkrSell v4 Single Gateway", version="6.5.1", lifespan=lifespan)
kernel: UkrSellKernel = None

# --- Вспомогательные методы ---

@lru_cache(maxsize=128)
def get_display_name(slug: str, tech_key: str):
    """Универсальный маппинг категорий из директории конкретного магазина"""
    try:
        mapping_path = os.path.join(BASE_DIR, "stores", slug, "category_map.json")
        if not os.path.exists(mapping_path):
            return tech_key.capitalize()
            
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            
        for ua_word, en_list in mapping.items():
            if isinstance(en_list, list) and tech_key in en_list:
                return ua_word.capitalize()
            elif en_list == tech_key:
                return ua_word.capitalize()
        return tech_key.capitalize()
    except Exception:
        return tech_key.capitalize()

# --- WebView UI (Универсальный HTML-шаблон) ---

def get_universal_catalog_html():
    return """
<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Catalog</title>
    <script src="https://telegram.org/js/telegram-web-app.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: -apple-system, system-ui, sans-serif; background-color: var(--tg-theme-bg-color, #ffffff); color: var(--tg-theme-text-color, #222222); -webkit-font-smoothing: antialiased; }
        .product-card { background: var(--tg-theme-secondary-bg-color, #f3f4f6); border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; height: 100%; border: 1px solid rgba(0,0,0,0.05); }
        .btn-primary { background: var(--tg-theme-button-color, #3498db); color: var(--tg-theme-button-text-color, #ffffff); }
        .line-clamp-2 { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
    </style>
</head>
<body class="p-4">
    <header class="mb-5 text-center">
        <h1 id="category-title" class="text-lg font-black uppercase tracking-tight">Завантаження...</h1>
        <p id="store-label" class="text-[10px] opacity-50 font-mono tracking-widest"></p>
    </header>
    <div id="product-container" class="grid grid-cols-2 gap-3 mb-10">
        <div class="col-span-2 text-center py-20 opacity-40 animate-pulse">З'єднуємось з сервером...</div>
    </div>
    <script>
        const tg = window.Telegram.WebApp;
        tg.ready();
        tg.expand();

        async function loadProducts() {
            const params = new URLSearchParams(window.location.search);
            const store = params.get('store');
            const cat = params.get('cat');
            
            if (!store || !cat) {
                document.getElementById('product-container').innerHTML = '<div class="col-span-2 text-center py-10 font-bold text-red-500">Помилка: Невірні параметри URL</div>';
                return;
            }

            try {
                // Прямой запрос к API через текущий origin для избежания проблем с CORS
                const response = await fetch(`/api/catalog?store=${store}&cat=${cat}`);
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                
                const data = await response.json();
                
                document.getElementById('category-title').innerText = data.title || 'Каталог';
                document.getElementById('store-label').innerText = data.store_name || '';
                
                renderProducts(data.products, store);
            } catch (e) {
                console.error('API Fetch Failed:', e);
                document.getElementById('product-container').innerHTML = `
                    <div class="col-span-2 text-center py-10">
                        <div class="text-red-500 font-bold mb-2">Помилка завантаження</div>
                        <div class="text-[10px] opacity-50">${e.message}</div>
                    </div>`;
            }
        }

        function renderProducts(products, store) {
            const container = document.getElementById('product-container');
            container.innerHTML = '';

            if (!Array.isArray(products) || products.length === 0) {
                container.innerHTML = '<div class="col-span-2 text-center py-10 opacity-50">Наразі немає доступних товарів.</div>';
                return;
            }

            products.forEach(p => {
                const name = p.product_name || p.subtype || "Товар";
                const info = p.price_current ? `${p.price_current} грн` : (p.count ? `Залишок: ${p.count}` : "");
                const img = p.image_main || "";
                const url = p.url || `https://ukrsellbot.com/search?store=${store}&q=${encodeURIComponent(name)}`;

                const card = `
                    <div class="product-card shadow-sm">
                        <div class="relative pt-[100%] bg-white flex items-center justify-center overflow-hidden">
                            ${img ? `<img src="${img}" class="absolute inset-0 w-full h-full object-contain p-2" onerror="this.parentElement.innerHTML='📦';">` : '<span class="text-4xl opacity-10">📦</span>'}
                        </div>
                        <div class="p-3 flex flex-col flex-grow">
                            <h3 class="text-[11px] font-bold mb-2 leading-tight h-8 line-clamp-2">${name}</h3>
                            <div class="mt-auto">
                                <div class="text-xs font-black text-[#e67e22] mb-2">${info}</div>
                                <button onclick="tg.openLink('${url}')" class="btn-primary w-full py-2 rounded-lg text-[10px] font-bold uppercase tracking-wider">Купити</button>
                            </div>
                        </div>
                    </div>`;
                container.innerHTML += card;
            });
        }
        loadProducts();
        tg.MainButton.setText("ЗАКРИТИ").show().onClick(() => tg.close());
    </script>
</body>
</html>
"""

# --- Routes ---

@app.get("/catalog", response_class=HTMLResponse)
async def get_catalog_page():
    """Отдает универсальный HTML-шаблон WebView"""
    return get_universal_catalog_html()

@app.get("/api/catalog")
async def get_catalog_api(store: str = Query(...), cat: str = Query(...)):
    """API эндпоинт для получения данных товаров конкретного магазина"""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel initializing")
    
    # Нормализация слага для поиска
    slug = store.lower()
    active_slugs = kernel.get_all_active_slugs()
    
    if slug not in active_slugs:
        log_event("CATALOG_ERROR", {"reason": "store_not_found", "slug": slug, "active": active_slugs}, level="warning")
        raise HTTPException(status_code=404, detail="Store not found")

    try:
        cluster_path = os.path.join(BASE_DIR, "stores", slug, "cluster_products.json")
        
        if not os.path.exists(cluster_path):
            log_event("CATALOG_ERROR", {"reason": "missing_cluster_file", "path": cluster_path}, level="error")
            return {"title": cat.capitalize(), "store_name": slug.upper(), "products": []}

        with open(cluster_path, 'r', encoding='utf-8') as f:
            clusters = json.load(f)

        # Безопасное извлечение списка продуктов
        raw_products = clusters.get(cat)
        if not isinstance(raw_products, list):
            raw_products = []
            
        display_name = get_display_name(slug, cat)

        return {
            "title": display_name,
            "store_name": slug.upper(),
            "products": raw_products[:30]  # Увеличен лимит для лучшего UX
        }
    except Exception as e:
        logger.error(f"Catalog API Critical Error for {slug}/{cat}: {e}", exc_info=True)
        return {"title": cat, "products": [], "error": str(e)}

@app.get("/logs/view", response_class=HTMLResponse)
async def view_logs_page():
    """Админ-панель для просмотра системных логов"""
    content = "Log file is empty or not found."
    file_size = "0 KB"
    if os.path.exists(LOG_FILE_PATH):
        try:
            size_bytes = os.path.getsize(LOG_FILE_PATH)
            file_size = f"{size_bytes / 1024:.2f} KB"
            with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
                content = "".join(lines[-1000:])
        except Exception as e:
            content = f"Error reading log: {e}"
    
    html_content = f"""
    <html>
        <head><title>UkrSell Monitor</title><style>body {{ background: #0d1117; color: #c9d1d9; font-family: monospace; padding: 20px; }} pre {{ background: #000; padding: 15px; border: 1px solid #30363d; border-radius: 6px; height: 75vh; overflow-y: auto; white-space: pre-wrap; font-size: 11px; }} .toolbar {{ display: flex; gap: 10px; margin-bottom: 20px; align-items: center; }} .btn {{ padding: 8px 16px; border-radius: 4px; color: white; text-decoration: none; font-family: sans-serif; font-weight: bold; }} .btn-refresh {{ background: #238636; }} .btn-clear {{ background: #da3633; border: none; cursor: pointer; }}</style></head>
        <body>
            <div class="toolbar">
                <div style="flex-grow:1"><h2>System Logs</h2><small>{file_size} | {LOG_FILE_PATH}</small></div>
                <a href="/logs/view" class="btn btn-refresh">Refresh</a>
                <form action="/admin/logs/clear" method="post" style="margin:0"><button class="btn btn-clear">Clear</button></form>
            </div>
            <pre id="log-box">{content}</pre>
            <script>document.getElementById('log-box').scrollTop = document.getElementById('log-box').scrollHeight;</script>
        </body>
    </html>"""
    return HTMLResponse(content=html_content)

@app.post("/admin/logs/clear")
async def clear_log_file():
    """Полная очистка файла логов"""
    try:
        if os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, "w", encoding="utf-8") as f: f.write("")
            log_event("LOGS_CLEARED", "Log file was cleared via admin interface")
        return RedirectResponse(url="/logs/view", status_code=303)
    except Exception as e:
        logger.error(f"Failed to clear logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/{slug}")
async def telegram_webhook(slug: str, request: Request, background_tasks: BackgroundTasks):
    """Шлюз для входящих обновлений Telegram"""
    if kernel is None: 
        raise HTTPException(status_code=503, detail="Kernel initializing")
    
    slug = slug.lower()
    if slug not in kernel.get_all_active_slugs(): 
        raise HTTPException(status_code=404, detail="Store not found")
        
    update = await request.json()
    background_tasks.add_task(kernel.handle_webhook, slug, update)
    return {"status": "accepted"}

@app.get("/health")
async def health():
    """Проверка жизнеспособности системы"""
    if kernel is None: return {"status": "initializing"}
    return {"status": "online", "active_stores": kernel.get_all_active_slugs()}

if __name__ == "__main__":
    # Запуск на 8080 порту для синхронизации с Nginx шлюзом
    uvicorn.run("main:app", host="0.0.0.0", port=8080, access_log=False)