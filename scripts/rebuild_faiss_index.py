"""
rebuild_faiss_index.py v2 — Точная синхронизация search_blob + FAISS
======================================================================
Ключевое отличие от v1:
  - blob строится из КОНКРЕТНОГО subtype товара, а не из всего списка
    синонимов категории. "Комбінезон" получает blob с "комбінезон",
    "Куртка" — с "куртка", "Дощовик" — с "дощовик".
  - Категорийные синонимы добавляются только для category-level поиска
    ("одяг", "apparel"), но не для subtype-level.
  - Животное из animal[] всегда присутствует: "dog", "cat" и т.д.

Структура passage:
  "passage: {animal} {subtype_ua} {subtype_synonyms} {category_ua} {brand} {title}"

Запуск:
  python scripts/rebuild_faiss_index.py --slug luckydog
  python scripts/rebuild_faiss_index.py --slug luckydog --dry-run
  python scripts/rebuild_faiss_index.py --slug luckydog --backup
"""

import argparse, json, re, shutil, sys, time, html
from pathlib import Path
import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-small"
BATCH_SIZE = 64
BASE_DIR   = Path("/root/ukrsell_v4/stores")

# ── Точные синонимы по subtype (реальные значения из каталога) ──────────────
SUBTYPE_SYNONYMS = {
    # Apparel subtypes
    "Комбінезон":  ["комбінезон", "kombinezon", "зимовий одяг", "тепло", "winter suit"],
    "Куртка":      ["куртка", "jacket", "пальто", "верхній одяг", "тепла куртка"],
    "Дощовик":     ["дощовик", "raincoat", "захист від дощу", "непромокальний", "rain"],
    "Светр":       ["светр", "светер", "sweater", "knit", "в'язаний"],
    "Толстовка":   ["толстовка", "худі", "hoodie", "кофта", "sweatshirt"],
    "Худі":        ["худі", "hoodie", "толстовка", "кофта з капюшоном"],
    "Кофта":       ["кофта", "толстовка", "светр", "jumper"],
    "Жилет":       ["жилет", "vest", "безрукавка", "жилетка"],
    "Попона":      ["попона", "жилет", "захисний жилет", "blanket coat"],
    "Взуття":      ["взуття", "черевики", "чоботи", "boots", "shoes", "захист лап", "лапи"],
    "Костюм":      ["костюм", "suit", "комплект", "новорічний одяг", "святковий"],

    # Walking subtypes
    "Шлея":              ["шлея", "шлейка", "harness", "нагрудник"],
    "Нашийник":          ["нашийник", "ошийник", "collar", "обшийник"],
    "Поводок-рулетка":   ["поводок", "повідець", "рулетка", "leash", "retractable"],
    "Намордник":         ["намордник", "muzzle", "museum muzzle"],

    # Feeding subtypes
    "Миска":  ["миска", "bowl", "годівниця", "поїлка", "slow feeder", "лабіринт"],

    # Carriers subtypes
    "Контейнер-переноска": ["переноска", "контейнер", "carrier", "crate", "клітка"],

    # Toys
    "Для собак і кішок": ["іграшка", "toy", "гра", "м'яч", "розвага"],
}

# ── Категорийные синонимы (только UA-рівень, без subtypes) ──────────────────
CATEGORY_UA = {
    "Apparel":            "одяг",
    "Walking":            "прогулянка амуніція",
    "Accessories":        "аксесуари",
    "Toys":               "іграшки",
    "Grooming":           "груминг догляд",
    "Feeding":            "годування миска",
    "Feeding & Watering": "годування поїлка миска",
    "Food":               "їжа ласощі",
    "Hygiene":            "гігієна",
    "Beds & Furniture":   "лежак лежанка",
    "Carriers":           "переноска",
    "Carriers & Travel":  "переноска подорож",
    "Travel":             "подорож",
    "Care":               "догляд",
    "Meds":               "ліки здоров'я",
    "Health":             "здоров'я",
}


def build_search_blob(title: str, brand: str, category: str,
                      subtype: str, animal_list: list) -> str:
    """
    Точный blob с subtype-level синонимами.

    Порядок токенов (влияет на вес при dot-product):
      1. animal          — "dog cat"
      2. subtype_ua      — "комбінезон"  (сам subtype — самый важный)
      3. subtype synonyms — "зимовий одяг тепло winter suit"
      4. category_ua     — "одяг"
      5. brand           — "dogs bomba"
      6. title tokens    — "зимовий комбінезон бульдогів 3xl"
    """
    # 1. Animal
    animal_str = " ".join(str(a).lower() for a in (animal_list or ["dog"]))

    # 2-3. Subtype + synonyms
    sub_lower  = (subtype or "").strip()
    sub_synonyms = SUBTYPE_SYNONYMS.get(sub_lower, [])
    sub_str = f"{sub_lower.lower()} {' '.join(sub_synonyms)}" if sub_lower else ""

    # 4. Category UA
    cat_ua  = CATEGORY_UA.get(category, category.lower() if category else "")

    # 5. Brand
    brand_str = (brand or "").lower()

    # 6. Title
    title_clean = re.sub(r"[^\w\s]", " ",
                         html.unescape(title or "").lower())

    # Собираем, дедуплицируем
    raw = f"{animal_str} {sub_str} {cat_ua} {brand_str} {title_clean}"
    tokens, seen = [], set()
    for seg in raw.split():
        if len(seg) > 1 and seg not in seen:
            tokens.append(seg)
            seen.add(seg)
    return " ".join(tokens)


def build_passage(product: dict) -> str:
    """passage: {blob} {title} — формат для multilingual-e5-small"""
    blob  = product.get("search_blob", "")
    title = html.unescape(product.get("title", ""))
    return f"passage: {blob} {title}".strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug",    required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backup",  action="store_true")
    args = parser.parse_args()

    store_path = BASE_DIR / args.slug
    data_file  = store_path / "normalized_products_final.json"
    index_file = store_path / "faiss.index"
    idmap_file = store_path / "id_map.json"

    print("=" * 62)
    print(f"  Rebuild FAISS v2  |  {args.slug.upper()}")
    print("=" * 62)

    if not data_file.exists():
        print(f"❌ Не найден: {data_file}")
        sys.exit(1)

    with open(data_file, encoding="utf-8") as f:
        products = json.load(f)
    print(f"\n📦 Товаров: {len(products)}")

    # ── Строим search_blob ──────────────────────────────────────
    print(f"\n🔧 Строим search_blob (subtype-aware)...")
    for p in products:
        attrs   = p.get("attributes", {})
        subtype = attrs.get("subtype", "")
        brand   = p.get("brand") or attrs.get("brand", "")
        p["search_blob"] = build_search_blob(
            title       = p.get("title", ""),
            brand       = brand,
            category    = p.get("category", ""),
            subtype     = subtype,
            animal_list = p.get("animal", ["dog"]),
        )

    # Показываем примеры по одному на subtype
    print("\n   Примеры (по одному на subtype):")
    shown = set()
    for p in products:
        sub = p.get("attributes", {}).get("subtype", "(none)")
        if sub not in shown:
            shown.add(sub)
            print(f"   [{p.get('category','?'):20} / {(sub or '(none)'):20}]")
            print(f"     title: {p.get('title','')[:50]}")
            print(f"     blob:  {p['search_blob'][:80]}")

    if args.dry_run:
        print("\n⚠️  --dry-run: файлы НЕ изменены")

        # Показываем потенциальные коллизии — товары с похожим blob
        print("\n   Проверка различимости blob:")
        blob_map = {}
        for p in products:
            sub = p.get("attributes", {}).get("subtype", "")
            blob_tokens = set(p["search_blob"].split())
            for other_sub, other_tokens in blob_map.items():
                if sub != other_sub:
                    overlap = len(blob_tokens & other_tokens) / max(len(blob_tokens), 1)
                    if overlap > 0.7:
                        print(f"   ⚠️  Высокий overlap {overlap:.0%}: '{sub}' vs '{other_sub}'")
            blob_map[sub] = blob_tokens
        print("   ✅ Проверка завершена")
        sys.exit(0)

    # ── Бэкап ──────────────────────────────────────────────────
    if args.backup:
        ts = int(time.time())
        for fp in [data_file, index_file, idmap_file]:
            if fp.exists():
                bak = fp.parent / f"{fp.stem}.bak_{ts}{fp.suffix}"
                shutil.copy2(fp, bak)
                print(f"💾 Бэкап: {bak.name}")

    # ── Сохраняем JSON с blob ───────────────────────────────────
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)
    print(f"\n✅ {data_file.name} обновлён (search_blob)")

    # ── Passages ───────────────────────────────────────────────
    p_ids, passages = [], []
    for p in products:
        pid = str(p.get("product_id", "")).strip()
        if not pid:
            continue
        p_ids.append(pid)
        passages.append(build_passage(p))
    print(f"\n📝 Passages: {len(passages)}")
    for i in range(min(3, len(passages))):
        print(f"   [{i}] {passages[i][:110]}")

    # ── Кодирование ────────────────────────────────────────────
    print(f"\n🤖 Загружаем {MODEL_NAME}...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ pip install sentence-transformers")
        sys.exit(1)

    model = SentenceTransformer(MODEL_NAME)
    print(f"⚡ Кодируем {len(passages)} passages (batch={BATCH_SIZE})...")
    t0 = time.time()
    embeddings = model.encode(
        passages,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
    )
    enc_time = time.time() - t0
    print(f"   Готово: {enc_time:.1f}с  shape={embeddings.shape}")

    # ── FAISS ──────────────────────────────────────────────────
    print(f"\n📦 Строим IndexFlatIP...")
    try:
        import faiss
    except ImportError:
        print("❌ pip install faiss-cpu")
        sys.exit(1)

    emb_np = np.array(embeddings).astype("float32")
    index  = faiss.IndexFlatIP(emb_np.shape[1])
    index.add(emb_np)
    print(f"   Векторов: {index.ntotal}")

    # ── Тест-поиск ─────────────────────────────────────────────
    print(f"\n🔍 Тест-поиск:")
    prod_by_id = {str(p.get("product_id")): p for p in products}
    tests = [
        ("зимовий комбінезон для собаки",      "Комбінезон"),
        ("тепла куртка для собаки",             "Куртка"),
        ("дощовик щоб не промокла",             "Дощовик"),
        ("взуття захист лап взимку на снігу",   "Взуття"),
        ("нашийник для лабрадора",              "Нашийник"),
        ("шлея для середньої собаки",           "Шлея"),
        ("миска повільне годування",            "Миска"),
    ]
    all_pass = True
    for q_text, expected_sub in tests:
        q_emb = model.encode(
            f"query: {q_text}",
            normalize_embeddings=True
        ).astype("float32")
        scores, idxs = index.search(np.expand_dims(q_emb, axis=0), 3)
        hits = []
        for sc, ix in zip(scores[0], idxs[0]):
            if ix < 0 or ix >= len(p_ids):
                continue
            pr  = prod_by_id.get(p_ids[ix], {})
            sub = pr.get("attributes", {}).get("subtype", "?")
            hits.append((pr.get("title", "?")[:40], sub, sc))
        ok = any(expected_sub in h[1] for h in hits)
        if not ok:
            all_pass = False
        print(f"   {'✅' if ok else '❌'} '{q_text}'  (ожидаем: {expected_sub})")
        for title, sub, sc in hits:
            marker = " ◀" if expected_sub in sub else ""
            print(f"        [{(sub or '?'):20}] {title} s={sc:.3f}{marker}")

    if all_pass:
        print("\n✅ Все тест-запросы нашли правильный subtype!")
    else:
        print("\n⚠️  Некоторые запросы не попали в ожидаемый subtype — проверь SUBTYPE_SYNONYMS")

    # ── Запись индекса ─────────────────────────────────────────
    faiss.write_index(index, str(index_file))
    print(f"\n✅ faiss.index  ({index.ntotal} vectors)  →  {index_file}")

    with open(idmap_file, "w", encoding="utf-8") as f:
        json.dump({
            "ids":         p_ids,
            "products":    prod_by_id,
            "total_items": len(p_ids),
            "version":     "7.2-subtype-blob",
            "built_at":    int(time.time()),
            "model":       MODEL_NAME,
        }, f, ensure_ascii=False)
    print(f"✅ id_map.json  →  {idmap_file}")

    print(f"\n{'='*62}")
    print(f"  ГОТОВО: {len(p_ids)} векторов  |  {enc_time:.0f}с кодирования")
    print(f"  Перезапусти сервис: systemctl restart ukrsell")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()