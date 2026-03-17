# /root/ukrsell_v4/scripts/deduplicate.py v1.5.0
import json
import re
import hashlib
import sys
import os
import time
import html
from collections import defaultdict, Counter

# ===========================
# Расширенная конфигурация
# ===========================
RAW_SUBTYPE_MAP = {
    "кігтеріз": "Grooming", "кігтеріз-сікатор": "Grooming", "пуходерка": "Grooming",
    "щітка": "Grooming", "тример": "Grooming", "колтуноріз": "Grooming",
    "поїлка": "Feeding", "миска": "Feeding", "контейнер для корму": "Feeding",
    "килимок под миску": "Feeding", "автогодівниця": "Feeding",
    "светр": "Apparel", "комбінезон": "Apparel", "взуття": "Apparel",
    "куртка": "Apparel", "дощовик": "Apparel", "футболка": "Apparel",
    "жилетка": "Apparel", "толстовка": "Apparel", "шапка": "Apparel",
    "намордник": "Walking", "повідець": "Walking", "нашийник": "Walking",
    "шлея": "Walking", "рулетка": "Walking", "сумка-переноска": "Walking",
    "лежак": "Beds & Furniture", "будиночок": "Beds & Furniture",
    "матрац": "Beds & Furniture", "подушка": "Beds & Furniture",
    "м'яч": "Toys", "кільце": "Toys", "канат": "Toys",
    "м'яка іграшка": "Toys", "інтерактивна іграшка": "Toys",
    "пелюшка": "Hygiene", "лоток": "Hygiene", "наповнювач": "Hygiene",
    "пакет для фекалій": "Hygiene", "мило-пінка": "Care",
    "догляд за лапами": "Care", "лосьйон для вух/очей": "Care",
    "краплі від паразитів": "Meds", "краплі от паразитів": "Meds",
    "нашийник від бліх": "Meds", "таблетки від глистів": "Meds",
    "гігієнічні труси": "Apparel", "кофта": "Apparel", "худі": "Apparel",
    "жилет": "Apparel", "попона": "Apparel", "костюм": "Apparel",
    "автогамак": "Walking", "контейнер-переноска": "Walking", "переноска": "Walking",
    "поводок-рулетка": "Walking", "повідець-рулетка": "Walking",
    "сворка": "Walking", "ремінь безпеки": "Walking", "автокрісло": "Walking",
    "літаюча тарілка": "Toys", "гантель": "Toys", "кістка": "Toys",
    "іграшка": "Toys", "охолоджувальна іграшка": "Toys",
    "палиця": "Toys", "тренувальне кільце": "Toys",
    "шампунь": "Care", "лапомийка": "Care", "фурмінатор": "Care",
    "килимок для злизування": "Feeding", "ласощі": "Feeding",
    "котяча м'ята": "Meds", "м'ята": "Meds",
    "кігтеріз-секатор": "Grooming",
"пуходірка": "Grooming"
}

def _normalize_apostrophe(s: str) -> str:
    return s.replace("\u02bc", "'").replace("\u2019", "'").replace("`", "'").replace("\u0060", "'")

SUBTYPE_CONF_MAP = {_normalize_apostrophe(k.lower()): v for k, v in RAW_SUBTYPE_MAP.items()}

PATTERNS = [
    (r'(\d+(?:[\.,]\d+)?\s*[хx\*]\s*\d+(?:[\.,]\d+)?\s*[хx\*]\s*\d+(?:[\.,]\d+)?)', 'size'),
    (r'(\d+(?:[\.,]\d+)?\s*[хx\*]\s*\d+(?:[\.,]\d+)?)', 'size'),
    (r'[dд]\s*(\d+(?:[\.,]\d+)?)', 'diameter'),
    (r'(\d+(?:[\.,]\d+)?)\s*(?:см|mm|мм)', 'size'),
    (r'\b(XS|S|M|L|XL|XXL|3XL|4XL|Small|Medium|Large|X-Large|XX-Large)\b', 'apparel_size'),
    (r'[\s\-]+(XS|S|M|L|XL|XXL|3XL|4XL)\b', 'apparel_size'),
    (r'(\d+\s*-\s*\d+\s*(?:см|мм)?)', 'size')
]

STOP_WORDS_GLOBAL = {
    'null', 'none', 'other', 'v2', 'test', 'см', 'мм', 'для', 'та', 'і', 'з', 'в',
    'на', 'із', 'по', 'шт', 'у', 'is', 'with', 'the', 'of', 'and', 'for', 'in'
}


def deep_clean(val):
    if val is None:
        return ""
    if not isinstance(val, str):
        val = str(val)
    val = html.unescape(html.unescape(val))
    val = re.sub(r'<[^>]+>', '', val)
    val = val.replace('\u00a0', ' ').replace('\xa0', ' ')
    return val.strip()


def get_base_identity(title):
    if not title:
        return "generic_item"

    text = _normalize_apostrophe(title.lower())

    for pattern, _ in PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    text = re.sub(
        r'\b(зелений|червоний|синій|білий|чорний|рожевий|green|red|blue|white|black|pink)\b',
        '', text
    )

    tokens = re.findall(r'\w+', text)
    meaningful = [t for t in tokens if t not in STOP_WORDS_GLOBAL and len(t) > 2]
    return "_".join(meaningful[:4]) if meaningful else "item"


def normalize_animal(val):
    if not val:
        return "universal"
    if isinstance(val, list):
        parts = [_normalize_apostrophe(deep_clean(a).lower()) for a in val if a]
    else:
        parts = [_normalize_apostrophe(a.strip().lower())
                 for a in str(val).replace(';', ',').split(',') if a.strip()]
    return "|".join(sorted(list(set(parts))))


def clean_and_extract(title):
    title = deep_clean(title)
    extracted = {}
    working_title = _normalize_apostrophe(title.lower())
    for pattern, attr_name in PATTERNS:
        matches = re.findall(pattern, working_title, re.IGNORECASE)
        for m in matches:
            if isinstance(m, tuple):
                m = m[0]
            val = m.strip().replace(' ', '').replace(',', '.')
            if attr_name not in extracted:
                extracted[attr_name] = val
            working_title = working_title.replace(m.lower(), '', 1)

    clean_t = re.sub(r'\s+', ' ', working_title).strip(' ,-()./').capitalize()
    return clean_t, extracted


def _best_representative(items: list) -> dict:
    """
    Выбирает эталонный товар из группы — тот у которого
    заголовок длиннее всего (больше информации).
    """
    return max(items, key=lambda x: len(x.get('title', '')))


def process_pipeline(slug):
    start_time = time.time()
    base_path = f"/root/ukrsell_v4/stores/{slug}"
    input_file = f"{base_path}/normalized_products_final.json"
    output_file = f"{base_path}/deduplicated_products.json"
    report_file = f"{base_path}/processing_report.json"
    error_log_path = f"{base_path}/critical_errors.jsonl"

    if not os.path.exists(input_file):
        print(f"Input not found: {input_file}")
        sys.exit(1)

    with open(input_file, 'r', encoding='utf-8') as f:
        products = json.load(f)

    groups = defaultdict(list)
    unmatched_subtypes = set()
    critical_errors = []

    for p in products:
        try:
            p['title'] = deep_clean(p.get('title', ''))
            p['brand'] = deep_clean(p.get('brand', 'No Brand'))

            p['animal_norm'] = normalize_animal(p.get('animal'))

            clean_title, var_attrs = clean_and_extract(p['title'])

            attrs = p.get('attributes', {})
            raw_subtype = _normalize_apostrophe(deep_clean(attrs.get('subtype', 'Unknown')))
            category = SUBTYPE_CONF_MAP.get(raw_subtype.lower(), "Other")
            if category == "Other" and raw_subtype.lower() not in ("unknown", "other", ""):
                unmatched_subtypes.add(raw_subtype)

            base_id = get_base_identity(clean_title)
            seed = f"{category}|{p['brand']}|{p['animal_norm']}|{base_id}".lower()
            group_key = hashlib.md5(seed.encode()).hexdigest()[:12]

            p['_clean_title'] = clean_title
            p['_var_attrs'] = var_attrs
            p['_canonical_category'] = category
            p['_norm_subtype'] = raw_subtype
            groups[group_key].append(p)

        except Exception as e:
            critical_errors.append({"id": p.get('product_id'), "error": str(e)})

    final_output = []
    for g_id, items in groups.items():
        st_counts = Counter([x['_norm_subtype'] for x in items if x['_norm_subtype'].lower() != 'unknown'])
        best_subtype = st_counts.most_common(1)[0][0] if st_counts else items[0]['_norm_subtype']

        prices = []
        for x in items:
            try:
                val = float(str(x.get('price', 0)).replace(',', '.'))
                if val > 0:
                    prices.append(val)
            except Exception:
                continue

        representative = _best_representative(items)

        all_text = f"{representative['brand']} {representative['_canonical_category']} {best_subtype}"
        for x in items:
            all_text += f" {x['title']} {x.get('search_blob', '')}"

        words = re.findall(r'\w+', _normalize_apostrophe(all_text.lower()))
        unique_words = sorted(list(set(w for w in words if w not in STOP_WORDS_GLOBAL and len(w) > 1)))

        parent_obj = {
            "product_id": f"grp_{g_id}",
            "title": representative['_clean_title'],
            "category": representative['_canonical_category'],
            "subtype": best_subtype if best_subtype.lower() != 'unknown' else None,
            "brand": representative['brand'] if representative['brand'] != 'No Brand' else None,
            "animal": representative['animal_norm'].replace('|', ', '),
            "price_min": min(prices) if prices else 0.0,
            "price_max": max(prices) if prices else 0.0,
            "image_url": representative.get('image_url'),
            "search_blob": " ".join(unique_words),
            "attributes": representative.get('attributes', {}),
            "variants": []
        }

        v_seen = {}
        for x in items:
            orig_id = str(x.get('product_id', 'unknown'))
            if orig_id in v_seen:
                v_hash = hashlib.md5(x.get('product_url', orig_id).encode()).hexdigest()[:4]
                orig_id = f"{orig_id}_{v_hash}"
            v_seen[orig_id] = True

            var_entry = {
                "variant_id": orig_id,
                "price": x.get('price'),
                "availability": x.get('availability'),
                "image_url": x.get('image_url'),
                "product_url": x.get('product_url')
            }
            var_entry.update(x.get('_var_attrs', {}))
            parent_obj['variants'].append(var_entry)

        final_output.append(parent_obj)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    if critical_errors:
        with open(error_log_path, 'w', encoding='utf-8') as f:
            for err in critical_errors:
                f.write(json.dumps(err, ensure_ascii=False) + "\n")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "1.5.0",
        "slug": slug,
        "stats": {
            "total_input": len(products),
            "total_output": len(final_output),
            "dedup_ratio": round(1 - len(final_output) / max(len(products), 1), 3),
            "critical_errors": len(critical_errors),
            "unmatched_subtypes": sorted(list(unmatched_subtypes))
        },
        "performance": f"{round(time.time() - start_time, 2)}s"
    }
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"COMPLETED: {len(final_output)} stable groups from {len(products)} products "
          f"(dedup ratio: {report['stats']['dedup_ratio']}). "
          f"Errors: {len(critical_errors)}")
    if unmatched_subtypes:
        print(f"WARN: unmatched subtypes ({len(unmatched_subtypes)}): "
              f"{', '.join(sorted(unmatched_subtypes)[:20])}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_pipeline(sys.argv[1])
    else:
        print("Usage: python3 deduplicate.py <slug>")