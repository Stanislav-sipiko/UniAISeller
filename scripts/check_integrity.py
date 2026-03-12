# /root/ukrsell_v4/scripts/check_integrity.py v6.8.1
import faiss
import json
import sys
import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def clean_text(text: str) -> str:
    if not text: return ""
    # Очистка должна быть идентична той, что в prepare_data.py
    text = re.sub(r'[^a-zA-Z0-9а-яА-ЯіІєЄїЇґҐ\s]', ' ', str(text))
    return re.sub(r'\s+', ' ', text).strip().lower()

def check_store_integrity(slug: str):
    base_path = Path(f"/root/ukrsell_v4/stores/{slug}")
    map_path = base_path / "id_map.json"
    index_path = base_path / "faiss.index"
    
    if not map_path.exists() or not index_path.exists():
        print(f"  [ERROR] Path not found: {base_path}")
        return

    print(f"--- Checking Integrity for: {slug} ---")
    
    # 1. Load Data
    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    index = faiss.read_index(str(index_path))
    model = SentenceTransformer('intfloat/multilingual-e5-small')
    
    # 2. Extract IDs
    if isinstance(data, dict) and "ids" in data:
        id_list = data["ids"]
        products_dict = data.get("products", {})
        print("  [INFO] Modern format detected.")
    else:
        # Legacy support
        id_list = list(data.keys())
        products_dict = data
        print("  [WARN] Legacy format detected.")

    # 3. Check Counts
    ntotal = index.ntotal
    n_ids = len(id_list)
    
    print(f"  [STEP 1] FAISS total: {ntotal} | Metadata total: {n_ids}")
    if ntotal != n_ids:
        print("  [CRITICAL] Count mismatch! Vectors and IDs are out of sync.")
        return

    # 4. Check Vector Dimension
    model_dim = model.get_sentence_embedding_dimension()
    print(f"  [STEP 2] Index Dim: {index.d} | Model Dim: {model_dim}")
    if index.d != model_dim:
        print("  [CRITICAL] Dimension mismatch!")
        return

    # 5. Semantic Mirror Test (Identical to prepare_data v6.8.1)
    print("  [STEP 3] Running Mirror Test (Top-1 consistency)...")
    
    # Проверяем тот самый индекс 366 (середина), на котором был FAIL
    test_idx = n_ids // 2 
    test_pid = id_list[test_idx]
    item = products_dict.get(test_pid, {})
    
    # --- ВАЖНО: Синхронизированная сборка query ---
    title_clean = clean_text(item.get('title', ''))
    desc_raw = item.get('description_clean', '') or item.get('description', '')
    desc_clean = clean_text(desc_raw)[:300]
    
    # Формируем запрос в точности как в Mirror Test при индексации
    test_q = f"query: [id: {test_pid}] {title_clean} {desc_clean}"
    # ----------------------------------------------
    
    v = model.encode([test_q], normalize_embeddings=True).astype('float32')
    dist, idxs = index.search(v, 1)
    
    found_idx = idxs[0][0]
    if found_idx == test_idx:
        print(f"  [OK] Point {test_idx} matches product ID: {test_pid}")
        print(f"       Score: {dist[0][0]:.4f}")
    else:
        actual_pid = id_list[found_idx] if found_idx < len(id_list) else "OUT_OF_BOUNDS"
        print(f"  [FAIL] Sync Error!")
        print(f"          Expected index {test_idx} ({test_pid})")
        print(f"          FAISS returned {found_idx} ({actual_pid})")
        print(f"          Query used: {test_q[:80]}...")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 check_integrity.py <store_slug>")
        sys.exit(1)
    
    target_slug = sys.argv[1]
    check_store_integrity(target_slug)