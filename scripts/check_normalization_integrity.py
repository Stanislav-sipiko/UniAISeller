# /root/ukrsell_v4/scripts/check_normalization_integrity.py v1.0.1
import json
import sys
from pathlib import Path
from core.logger import logger

def check_integrity(slug: str):
    base_path = Path(f"/root/ukrsell_v4/stores/{slug}")
    raw_path = base_path / "products.json"
    norm_path = base_path / "products_normalized.json"
    profile_path = base_path / "store_profile.json"

    if not norm_path.exists():
        logger.error(f"❌ Файл нормализации не найден: {norm_path}")
        return

    # 1. Загрузка данных
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    with open(norm_path, "r", encoding="utf-8") as f:
        norm_data = json.load(f)
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)
        schema_keys = profile.get("profile", {}).get("schema_keys", [])

    raw_ids = {str(p.get("product_id")) for p in raw_data}
    norm_ids = {str(p.get("id")) for p in norm_data}

    logger.info(f"--- 🔍 Integrity Check: {slug} ---")

    # 2. Проверка полноты (Coverage)
    missing_ids = raw_ids - norm_ids
    coverage = (len(norm_ids) / len(raw_ids)) * 100

    print(f"[STEP 1] Coverage: {len(norm_ids)}/{len(raw_ids)} ({coverage:.2f}%)")
    if missing_ids:
        logger.warning(f"⚠️ Пропущено {len(missing_ids)} товаров! (Например: {list(missing_ids)[:5]})")
    else:
        logger.info("✅ Все товары на месте.")

    # 3. Проверка структуры (Schema Validation)
    print(f"[STEP 2] Schema & Content Validation...")
    errors = {
        "empty_name": 0,
        "missing_entities": 0,
        "empty_animal": 0,
        "malformed_json_fields": 0
    }

    for item in norm_data:
        # Проверка имени
        if not item.get("name") or len(item["name"]) < 3:
            errors["empty_name"] += 1
        
        # Проверка ключей нормализации
        entities = item.get("norm_entities", {})
        if not entities or not any(k in entities for k in schema_keys):
            errors["missing_entities"] += 1
            
        # Проверка животных
        animal = item.get("animal", [])
        if not animal or not isinstance(animal, list):
            errors["empty_animal"] += 1

    # 4. Вывод отчета по качеству LLM
    for err, count in errors.items():
        status = "❌" if count > 0 else "✅"
        print(f"  {status} {err}: {count}")

    # 5. Резюме
    if coverage < 100 or sum(errors.values()) > 0:
        logger.error(f"❌ Критичность: ТРЕБУЕТСЯ ДОРАБОТКА (Fix needed)")
        print(f"\n💡 Совет: Запустите build_normalized_layer.py еще раз. ")
        print(f"   Скрипт v1.0.4 автоматически подхватит пропущенные {len(missing_ids)} ID.")
    else:
        logger.info("💎 Идеально! Данные готовы к индексации (FAISS).")

if __name__ == "__main__":
    slug = sys.argv[1] if len(sys.argv) > 1 else "luckydog"
    check_integrity(slug)