# UkrSell v4 — Architecture Guide for AI Agents

## CRITICAL: Universal SaaS Platform

This is a **multi-tenant SaaS platform**. The kernel serves ANY type of store:
electronics, tea, auto parts, pet supplies, clothing, etc.

**NEVER hardcode store-specific logic into core/.**

---

## Directory Structure

```
/root/ukrsell_v4/
├── core/                  ← UNIVERSAL kernel (store-agnostic)
│   ├── confidence.py      ← scoring engine, NO store logic
│   ├── retrieval.py       ← FAISS search, NO store logic
│   ├── analyzer.py        ← LLM synthesis, NO store logic
│   ├── intelligence.py    ← entity filter, NO store logic
│   └── ...
├── kernel.py              ← orchestrator, NO store logic
└── stores/
    └── luckydog/          ← pet store specific data ONLY
        ├── category_map.json      ← luckydog categories
        ├── store_profile.json     ← luckydog profile, intent_mapping
        ├── normalized_products_final.json
        └── faiss.index
```

---

## The Golden Rule

| Where to put it | What goes there |
|---|---|
| `core/` | Logic that works for ANY store (phones, tea, auto parts) |
| `stores/{slug}/` | Data specific to ONE store |

---

## How Store-Specific Data Reaches the Kernel

Everything store-specific is loaded from `stores/{slug}/store_profile.json`
and `stores/{slug}/category_map.json` at startup into `StoreContext`.

The kernel reads it via `ctx.profile` and `ctx.category_map`. It never
imports or references any store slug directly.

**Example — animal translation is luckydog-specific:**

```json
// stores/luckydog/store_profile.json
{
  "intent_mapping": {
    "animal": {
      "field": "animal",
      "translations": {
        "кіт": "cat",
        "кішка": "cat",
        "собака": "dog",
        "пес": "dog",
        "кролик": "rabbit"
      }
    }
  }
}
```

The kernel reads `ctx.profile["intent_mapping"]` and applies translations
generically — without knowing it's a pet store.

---

## Correct Pattern for entity_filter (intelligence.py)

```python
# WRONG — hardcoded for pet stores only
ANIMAL_TRANSLATIONS = {"кіт": "cat", "собака": "dog"}

# CORRECT — driven by store profile
def entity_filter(products, entities, intent_mapping, category_map):
    for field, mapping in intent_mapping.items():
        translations = mapping.get("translations", {})
        value = entities.get(field, "")
        translated = translations.get(value.lower(), value)
        # filter products by translated value
```

---

## Correct Pattern for retrieval.py stemming

```python
# WRONG — Ukrainian-specific stemmer hardcoded in core
from uk_stemmer import UkStemmer

# CORRECT — stemming config comes from store_profile
stem_config = ctx.profile.get("stemming", {})
# stem_config = {"enabled": true, "language": "uk"}
# Core just checks stem_config["enabled"] — works for any language
```

---

## Checklist Before Editing core/

- [ ] Would this logic work for a phone store? A tea store?
- [ ] If NO → it belongs in `stores/{slug}/store_profile.json`, not in `core/`
- [ ] Am I importing or referencing a slug name? → WRONG
- [ ] Am I reading from `ctx.profile` or `ctx.category_map`? → CORRECT

---

## Current Store: luckydog

- 696 SKUs, dog/cat apparel and accessories
- Language: Ukrainian
- Currency: грн
- Store-specific fields: `animal`, `breed`, `size`
- These fields are declared in `stores/luckydog/store_profile.json → intent_mapping`

Any fix for luckydog behavior goes into `stores/luckydog/` — not into `core/`.
