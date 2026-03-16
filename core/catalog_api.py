# /root/ukrsell_v4/core/catalog_api.py v1.0.2
import json
import os
from functools import lru_cache
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

STORES_PATH = "/root/ukrsell_v4/stores"

def find_store_folder(slug: str):
    try:
        folders = os.listdir(STORES_PATH)
        for f in folders:
            if f.lower() == slug.lower():
                return f
    except:
        return None
    return None

@lru_cache(maxsize=128)
def get_display_name(store_slug: str, tech_key: str):
    try:
        path = os.path.join(STORES_PATH, store_slug, "category_map.json")
        with open(path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        for ua_word, en_list in mapping.items():
            if (isinstance(en_list, list) and tech_key in en_list) or en_list == tech_key:
                return ua_word.capitalize()
    except:
        pass
    return tech_key.capitalize()

@app.get("/catalog", response_class=HTMLResponse)
async def get_catalog_page():
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
        body { font-family: -apple-system, system-ui, sans-serif; background-color: var(--tg-theme-bg-color, #ffffff); color: var(--tg-theme-text-color, #222222); }
        .product-card { background: var(--tg-theme-secondary-bg-color, #f3f4f6); border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; height: 100%; border: 1px solid rgba(0,0,0,0.05); }
    </style>
</head>
<body class="p-4">
    <header class="mb-5 text-center">
        <h1 id="category-title" class="text-lg font-black uppercase">Завантаження...</h1>
        <p id="store-label" class="text-[10px] opacity-50 font-mono"></p>
    </header>
    <div id="product-container" class="grid grid-cols-2 gap-3 mb-10">
        <div class="col-span-2 text-center py-20 opacity-40">З'єднуємось...</div>
    </div>
    <script>
        const tg = window.Telegram.WebApp;
        tg.ready();
        tg.expand();
        async function loadProducts() {
            const params = new URLSearchParams(window.location.search);
            try {
                const response = await fetch(`/api/catalog?${params.toString()}`);
                const data = await response.json();
                if (data.error) throw new Error(data.error);
                document.getElementById('category-title').innerText = data.title;
                document.getElementById('store-label').innerText = data.store_name;
                const container = document.getElementById('product-container');
                container.innerHTML = '';
                data.products.forEach(p => {
                    const name = p.product_name || p.subtype || "Товар";
                    const info = p.price_current ? `${p.price_current} грн` : (p.count ? `Залишок: ${p.count}` : "");
                    const card = `
                        <div class="product-card">
                            <div class="relative pt-[100%] bg-white flex items-center justify-center">
                                <img src="${p.image_main || ''}" class="absolute inset-0 w-full h-full object-contain p-2" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 1 1%22><rect width=%221%22 height=%221%22 fill=%22%23eee%22/></svg>';">
                            </div>
                            <div class="p-3 flex flex-col flex-grow">
                                <h3 class="text-[11px] font-bold mb-2 line-clamp-2 h-8 leading-tight">${name}</h3>
                                <div class="mt-auto">
                                    <div class="text-xs font-black text-orange-500 mb-2">${info}</div>
                                    <button onclick="tg.openLink('${p.url || '#'}')" class="w-full py-2 rounded-lg text-[10px] font-bold uppercase" style="background: var(--tg-theme-button-color, #3498db); color: var(--tg-theme-button-text-color, #fff);">Купити</button>
                                </div>
                            </div>
                        </div>`;
                    container.innerHTML += card;
                });
            } catch (e) {
                document.getElementById('product-container').innerHTML = `<div class="col-span-2 text-center py-10 text-red-500 font-bold">${e.message}</div>`;
            }
        }
        loadProducts();
        tg.MainButton.setText("ЗАКРИТИ").show().onClick(() => tg.close());
    </script>
</body>
</html>
"""

@app.get("/api/catalog")
async def get_catalog_data(
    store: str = Query(...),
    cat: str = Query(...)
):
    try:
        real_name = find_store_folder(store)
        if not real_name:
            return {"error": "Store not found", "products": []}
        cluster_path = os.path.join(STORES_PATH, real_name, "cluster_products.json")
        if not os.path.exists(cluster_path):
            return {"error": "Catalog missing", "products": []}
        with open(cluster_path, 'r', encoding='utf-8') as f:
            clusters = json.load(f)
        products = clusters.get(cat, [])[:10]
        return {
            "title": get_display_name(real_name, cat),
            "store_name": real_name.replace("_", " ").upper(),
            "products": products
        }
    except Exception as e:
        return {"error": str(e), "products": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)