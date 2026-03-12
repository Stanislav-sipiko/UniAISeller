# /root/ukrsell_v4/scripts/prepare_data.py v7.1
import os
import json
import sys
import faiss
import logging
import asyncio
import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Настройка путей
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from core.llm_selector import LLMSelector
from core.logger import logger

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def clean_text(text: str) -> str:
    if not text: return ""
    # Очистка для эмбеддингов: только буквы, цифры и базовые символы
    text = re.sub(r'[^a-zA-Z0-9а-яА-ЯіІєЄїЇґҐ\s]', ' ', str(text))
    return re.sub(r'\s+', ' ', text).strip().lower()

def extract_json_array(text: str) -> str:
    try:
        match = re.search(r'(\[[\s\S]*?\])', text)
        if match:
            content = match.group(0)
            content = "".join(ch for ch in content if ord(ch) >= 32)
            return content
        return text
    except Exception: return text

def extract_mapped_attributes(product, schema_keys):
    """Вытягивает значения из attributes товара на основе schema.json"""
    found_attrs = {}
    if not schema_keys: return found_attrs
    
    # Приводим ключи к нижнему регистру для гибкого сравнения
    product_attrs = {str(a['key']).lower(): a['value'] for a in product.get('attributes', [])}
    
    for key in schema_keys:
        val = product_attrs.get(key.lower())
        if val:
            found_attrs[key] = val
    return found_attrs

async def get_normalized_metadata(selector: LLMSelector, products_chunk: list, retries: int = 3) -> list:
    """Извлечение данных через LLM (FAST/HEAVY) для нормализации."""
    prompt = f"Return ONLY valid JSON array: [{{\"id\": \"...\", \"parent_sku\": \"...\", \"attributes\": \"...\", \"anchor\": \"...\"}}] for: {json.dumps([{'id': p.get('product_id'), 'title': p.get('title')} for p in products_chunk], ensure_ascii=False)}"

    for attempt in range(retries):
        try:
            client, model_name = selector.get_fast() if attempt == 0 else selector.get_heavy()
            
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a data tool. Output ONLY a valid JSON array. PROSE IS FORBIDDEN."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                timeout=15
            )
            raw_content = response.choices[0].message.content
            json_str = extract_json_array(raw_content)
            data = json.loads(json_str)
            if isinstance(data, list): return data
        except Exception as e:
            logger.warning(f"⚠️ Metadata extraction attempt {attempt+1} failed: {str(e)[:100]}")
            if attempt < retries - 1:
                await asyncio.sleep(1.5)
    return []

async def prepare_store_data(slug: str):
    selector = LLMSelector()
    await selector.ensure_ready()
    
    base_path = Path(f"/root/ukrsell_v4/stores/{slug}")
    products_path = base_path / "products.json"
    schema_path = base_path / "schema.json"
    
    if not products_path.exists():
        logger.error(f"File not found: {products_path}")
        return

    # Загрузка схемы атрибутов
    schema_keys = []
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_data = json.load(f)
            schema_keys = schema_data.get("keys", [])
            logger.info(f"📜 SCHEMA LOADED: Found {len(schema_keys)} keys for filtering.")

    with open(products_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    ordered_texts = []
    p_ids = []
    products_dict = {}
    
    logger.info(f"🚀 Старт V7.1 (Universal Indexer) для магазина: {slug}")

    batch_size = 10
    for i in range(0, len(products), batch_size):
        chunk = products[i:i + batch_size]
        llm_data = await get_normalized_metadata(selector, chunk)
        llm_map = {str(item.get('id')): item for item in llm_data if isinstance(item, dict) and item.get('id')}

        for p in chunk:
            pid = str(p.get('product_id'))
            p_ids.append(pid)
            products_dict[pid] = p
            meta = llm_map.get(pid)
            
            # 1. Извлекаем атрибуты по схеме (из базы и из LLM)
            mapped_attrs = extract_mapped_attributes(p, schema_keys)
            if meta and isinstance(meta.get('attributes'), dict):
                 mapped_attrs.update(meta['attributes'])

            # 2. Формируем тех. данные
            status_tag = "" if meta else "[LLM_MISS]"
            parent = meta.get('parent_sku', p.get('title')) if meta else p.get('title')
            anchor = meta.get('anchor', "gen") if meta else "miss"
            
            # 3. Текстовый блок
            title_clean = clean_text(p.get('title', ''))
            desc_raw = p.get('description_clean', '') or p.get('description', '')
            desc_clean = clean_text(desc_raw)[:300]
            
            # 4. Формируем строку атрибутов для усиления эмбеддинга
            attr_str = " ".join([f"[{k}: {v}]" for k, v in mapped_attrs.items()])
            
            # Формируем passage для мультиязычного E5
            content = f"passage: {status_tag} [id: {pid}] [anchor: {anchor}] [parent: {parent}] {attr_str} {title_clean} {desc_clean}"
            ordered_texts.append(content)
        
        if (i // batch_size) % 10 == 0:
            logger.info(f"📊 Прогресс: {min(i + batch_size, len(products))}/{len(products)}")

    # Генерация эмбеддингов
    logger.info("📡 Генерация эмбеддингов (multilingual-e5-small)...")
    model_st = SentenceTransformer('intfloat/multilingual-e5-small')
    embeddings = model_st.encode(ordered_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    
    # Создание FAISS индекса
    logger.info("📦 Сборка FAISS индекса...")
    embeddings_np = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)
    
    faiss.write_index(index, str(base_path / "faiss.index"))
    
    with open(base_path / "id_map.json", "w", encoding="utf-8") as f:
        json.dump({
            "ids": p_ids, 
            "products": products_dict, 
            "total_items": len(p_ids), 
            "version": "7.1",
            "schema_keys": schema_keys
        }, f, ensure_ascii=False, indent=2)

    # --- MIRROR TEST V2 ---
    logger.info("🔍 Запуск Mirror Test...")
    test_points = [0, len(p_ids)//2, len(p_ids)-1] if len(p_ids) > 0 else []
    for idx in test_points:
        target_id = p_ids[idx]
        item = products_dict[target_id]
        
        # Для Mirror Test используем только ID и Title для чистоты эксперимента
        test_q = f"query: [id: {target_id}] {clean_text(item.get('title', ''))}"
        
        q_v = model_st.encode([test_q], normalize_embeddings=True).astype('float32')
        D, I = index.search(q_v, 1)
        
        if I[0][0] == idx:
            logger.info(f"  ✅ Point {idx} OK (Score: {D[0][0]:.4f})")
        else:
            logger.error(f"  ❌ Point {idx} FAIL! Found {p_ids[I[0][0]]}")

    logger.info(f"🏁 Индексация '{slug}' завершена.")
    await selector.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 prepare_data.py <store_slug>")
        sys.exit(1)
    asyncio.run(prepare_store_data(sys.argv[1]))