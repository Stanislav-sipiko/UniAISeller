# /root/ukrsell_v4/engine/keyboards.py v1.5.0
import json
import os
from urllib.parse import quote
from typing import Dict, Any

def get_main_menu(ctx) -> Dict[str, Any]:
    """
    Универсальная клавиатура:
    1. Читает категории из store_profile.json.
    2. Сопоставляет их с реальными ключами в cluster_products.json (case-insensitive).
    3. Формирует безопасные URL для Telegram WebApp.
    """
    slug = getattr(ctx, 'slug', 'luckydog').strip()
    base_path = getattr(ctx, 'base_path', f"/root/ukrsell_v4/stores/{slug}")
    
    profile_path = os.path.join(base_path, "store_profile.json")
    clusters_path = os.path.join(base_path, "cluster_products.json")
    
    menu_items = []

    if os.path.exists(profile_path) and os.path.exists(clusters_path):
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)
                dist = profile_data.get("profile", {}).get("category_distribution", {})
            
            with open(clusters_path, 'r', encoding='utf-8') as f:
                real_data = json.load(f)
                # Создаем мапу { 'lower_key': 'OriginalKey' } 
                # чтобы 'grooming' нашел 'Grooming', а 'одяг' нашел 'Одяг'
                actual_keys = {k.lower(): k for k in real_data.keys()}

            # Сортируем категории по весу (доле в магазине)
            sorted_cats = sorted(dist.items(), key=lambda x: x[1], reverse=True)

            for cat_name, _ in sorted_cats:
                if len(menu_items) >= 4:
                    break
                
                # Ищем точное совпадение ключа в cluster_products
                target_key = actual_keys.get(cat_name.lower())
                
                if target_key:
                    # Словарь красивых названий для LuckyDog
                    # Для других магазинов будет просто Capitalize
                    labels = {
                        "одяг": "👕 Одяг",
                        "feeding": "🥣 Годування",
                        "walking": "🦮 Прогулянка",
                        "grooming": "🧼 Гігієна",
                        "toys": "🎾 Іграшки"
                    }
                    display_text = labels.get(target_key.lower(), target_key.capitalize())
                    menu_items.append((display_text, target_key))

        except Exception as e:
            # Если что-то пошло не так, в логах будет ошибка, но бот не умрет
            pass

    # Если файлы не найдены или пусты - базовый набор
    if not menu_items:
        menu_items = [("📦 Каталог", "Одяг"), ("❓ Допомога", "Grooming")]

    keyboard = []
    current_row = []
    base_url = "https://ukrsellbot.com/catalog"

    for text, cat_id in menu_items:
        # quote() обязателен для ключа "Одяг" и прочих
        safe_url = f"{base_url}?store={slug}&cat={quote(cat_id)}"
        
        current_row.append({
            "text": text,
            "web_app": {"url": safe_url}
        })
        
        if len(current_row) == 2:
            keyboard.append(current_row)
            current_row = []
            
    if current_row:
        keyboard.append(current_row)

    return {
        "keyboard": keyboard,
        "resize_keyboard": True,
        "persistent": True
    }

def get_empty_results_keyboard() -> Dict[str, Any]:
    return {"keyboard": [[{"text": "Скинути фільтри"}]], "resize_keyboard": True}