#/root/ukrsell_v4/scripts/health_check.py
import os
import sys
import time
import traceback
import json
from pathlib import Path

def print_status(component, status, message="", warn=False):
    if warn:
        color = "\033[93m[WARN]\033[0m"
    else:
        color = "\033[92m[OK]\033[0m" if status else "\033[91m[FAIL]\033[0m"
    print(f"{color} {component:25} : {message}")

def run_health_check(store_slug="luckydog", fast=False):
    print(f"--- 🏥 UkrSell_V4 Pre-Flight Health Check [{store_slug}] ---")
    all_passed = True
    base_path = Path(f"/root/ukrsell_v4/stores/{store_slug}")

    # 1. Проверка основных зависимостей (Numpy, FAISS)
    try:
        import numpy as np
        import faiss
        # Базовая проверка работоспособности библиотек
        test_arr = np.array([1.0, 2.0, 3.0], dtype="float32")
        test_index = faiss.IndexFlatL2(3)
        test_index.add(test_arr.reshape(1, -1))
        print_status("Dependencies", True, f"Numpy {np.__version__}, FAISS {faiss.__version__}")
    except Exception as e:
        print_status("Dependencies", False, f"Error: {str(e)}\n{traceback.format_exc()}")
        all_passed = False
        return False # Критическая ошибка, дальнейшие тесты невозможны

    # 2. Проверка SentenceTransformer (ML-модель)
    if not fast:
        try:
            from sentence_transformers import SentenceTransformer
            model_name = "intfloat/multilingual-e5-small"
            # Проверка загрузки или наличия в локальном кэше
            model = SentenceTransformer(model_name)
            print_status("ML Model", True, f"{model_name} ready.")
        except Exception as e:
            print_status("ML Model", False, f"Model load error: {str(e)}\n{traceback.format_exc()}")
            all_passed = False
    else:
        print_status("ML Model", True, "Skipped (fast mode)", warn=True)

    # 3. Проверка FAISS Индекса (на диске)
    index_file = base_path / "faiss.index"
    if index_file.exists():
        try:
            import numpy as np
            import faiss
            idx = faiss.read_index(str(index_file))
            if idx.ntotal > 0:
                print_status("FAISS Index", True, f"Loaded {idx.ntotal} vectors from {index_file.name}")
                
                # Мини-тест поиска с динамической размерностью вектора
                try:
                    dim = idx.d
                    test_vec = np.random.random((1, dim)).astype("float32")
                    D, I = idx.search(test_vec, 1)
                    print_status("FAISS Query Test", True, f"Search OK (dim: {dim}, top_dist: {D[0][0]:.4f})")
                except Exception as e_search:
                    print_status("FAISS Query Test", False, f"Search failed: {str(e_search)}")
                    all_passed = False
            else:
                print_status("FAISS Index", True, f"Loaded empty index ({idx.ntotal} vectors)", warn=True)
        except Exception as e:
            print_status("FAISS Index", False, f"Index corruption or format error: {str(e)}\n{traceback.format_exc()}")
            all_passed = False
    else:
        print_status("FAISS Index", False, f"Missing {index_file}")
        all_passed = False

    # 4. Проверка Semantic Cache (JSON + права доступа на запись)
    cache_file = base_path / "semantic_cache.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Определяем количество записей в зависимости от структуры
                if isinstance(data, dict) and "cache" in data:
                    count = len(data["cache"])
                else:
                    count = len(data) if isinstance(data, (dict, list)) else 0
                print_status("Semantic Cache", True, f"File exists, contains {count} entries.")

            # Тест физической записи на диск (права доступа)
            try:
                temp_path = cache_file.with_suffix(".tmp_hc")
                test_payload = {"_hc_test_": time.time(), "status": "ok"}
                with open(temp_path, "w", encoding="utf-8") as tf:
                    json.dump(test_payload, tf)
                temp_path.unlink() # Удаляем временный файл
                print_status("Cache Disk Write", True, "Disk permissions OK")
            except Exception as e_write:
                print_status("Cache Disk Write", False, f"Permission denied: {str(e_write)}")
                all_passed = False
        except Exception as e:
            print_status("Semantic Cache", False, f"JSON Error in {cache_file.name}: {str(e)}")
            all_passed = False
    else:
        print_status("Semantic Cache", True, "Missing (will be created on first run)", warn=True)

    # 5. Проверка путей и структуры данных магазина
    critical_files = ["id_map.json", "normalized_products_final.json", "category_map.json"]
    missing = [f for f in critical_files if not (base_path / f).exists()]
    if not missing:
        print_status("Store Assets", True, "All critical JSON assets found.")
    else:
        print_status("Store Assets", False, f"Missing: {', '.join(missing)}")
        all_passed = False

    # Итоговый вердикт
    print("-" * 55)
    if all_passed:
        print("\033[92m✅ SYSTEM READY FOR PRODUCTION\033[0m")
        return True
    else:
        print("\033[91m❌ SYSTEM NOT READY - Check errors above\033[0m")
        return False

if __name__ == "__main__":
    # Аргументы: [slug] [--fast]
    target_slug = "luckydog"
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        target_slug = sys.argv[1]
    
    fast_mode = "--fast" in sys.argv
    
    success = run_health_check(target_slug, fast=fast_mode)
    sys.exit(0 if success else 1)