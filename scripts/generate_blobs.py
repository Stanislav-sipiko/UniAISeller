# /root/ukrsell_v4/scripts/generate_blobs.py v1.6.1
import json
import re
import os

def generate_smart_blob(item):
    """
    Генерирует чистый поисковый индекс, исключая мусорные токены 
    и исправляя ошибки сегментации украинских слов.
    """
    # Список шума: предлоги, частицы, общие термины и технические размеры
    STOP_WORDS = {
        'для', 'та', 'від', 'під', 'час', 'яка', 'який', 'що', 'без', 'через', 'при', 'із', 'за',
        'собак', 'собака', 'собаки', 'собачий', 'кіт', 'коти', 'кішка', 'кішки', 'кішок', 'котiв',
        'cat', 'cats', 'dog', 'dogs', 'animal', 'animals',
        'xl', 'xxl', 'xxxl', '3xl', '4xl', '5xl', 'xs', 'мм', 'см', 'шт', 'грн'
    }

    # Поля-доноры для поискового облака
    parts = [
        item.get('title', ''),
        item.get('subtype', ''),
        item.get('brand', ''),
        item.get('attributes', {}).get('purpose', '')
    ]
    full_text = " ".join(parts).lower()

    # Регулярка для украинского и английского языков, сохраняющая апостроф и дефис
    # Теперь "в'язаний" не развалится на части
    words = re.findall(r"[a-zа-яіїєґ']+(?:-[a-zа-яіїєґ']+)*", full_text)

    final_tokens = set()
    for word in words:
        # Обработка сложных слов через дефис (напр. "кігтеріз-секатор")
        if '-' in word:
            sub_words = word.split('-')
            final_tokens.update([sw for sw in sub_words if len(sw) > 2])
        
        # Фильтрация мусора и слишком коротких слов
        if word not in STOP_WORDS and len(word) > 2:
            final_tokens.add(word)

    return " ".join(sorted(list(final_tokens)))

def process_store_data(slug):
    """
    Загружает JSON по указанному пути, обновляет блобы и сохраняет обратно.
    """
    base_path = f"/root/ukrsell_v4/stores/{slug}"
    file_path = os.path.join(base_path, "deduplicated_products.json")

    if not os.path.exists(file_path):
        print(f"[-] Файл не найден: {file_path}")
        return

    print(f"[*] Обработка стора: {slug}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for product in data:
        product['search_blob'] = generate_smart_blob(product)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[+] Успех! Обновлено {len(data)} товаров.")

if __name__ == "__main__":
    # Здесь можно передать любой slug твоего стора
    STORE_SLUG = "luckydog" 
    process_store_data(STORE_SLUG)