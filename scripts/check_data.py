# /root/ukrsell_v4/check_data.py
import json

PATH = "/root/ukrsell_v4/stores/phonestore/id_map.json"

def check():
    try:
        with open(PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Если это Legacy (словарь), берем значения. Если новый формат - data['products']
        products = data.get("products", data)
        
        iphones = [p for p in products.values() if "iphone" in str(p.get("name", "")).lower()]
        
        print(f"--- Результаты проверки {PATH} ---")
        print(f"Всего товаров в базе: {len(products)}")
        print(f"Найдено iPhone: {len(iphones)}")
        
        if iphones:
            print("\nПримеры iPhone в базе:")
            for i, p in enumerate(iphones[:3]):
                name = p.get("name")
                price = p.get("price")
                color_attr = p.get("attributes", {}).get("колір", "не указан")
                print(f"{i+1}. {name} | Цена: {price} | Цвет в аттр: {color_attr}")
                
            # Собираем все уникальные цвета для iPhone
            colors = set()
            for p in iphones:
                c = p.get("attributes", {}).get("колір")
                if c: colors.add(c)
            print(f"\nУникальные значения 'колір' у iPhone: {colors}")
        
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    check()