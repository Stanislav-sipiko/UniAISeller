# LuckyDog AI Assistant (v11.0.0)

Production-grade AI assistant for a pet products store (Lucky Dog).  
Implements modular LLM orchestration, hybrid retrieval, follow-up handling, user session memory, and FSM self-learning.

---

## 1. Overview

LuckyDog v11.0.0 is a Telegram-based assistant that:

- Answers product-related queries (availability, price, attributes).
- Handles follow-up questions (e.g. "and cheaper?", "what else?", "size?").
- Uses LLMs for intent detection, query planning, and answer generation.
- Combines Meilisearch, normalized product data, and legacy JSON search.
- Learns from its own mistakes via a soft-patching mechanism (FSM self-learning).

---

## 2. Project structure

```text
ukrsell_project_v3/
│
├── main.py                  # Main orchestrator (Telegram bot + HTTP server)
├── llm_selector.py          # Centralized LLM selection (fast / heavy)
├── config.py                # Tokens, API keys, configuration
│
├── engine/
│   ├── followup_engine.py   # Follow-up query handling
│   ├── session_memory.py    # Per-user session context (history, last products, profile)
│   ├── retrieval_engine.py  # Meilisearch + products_normalized.json
│   ├── planner_llm.py       # LLM-based query decomposition
│   ├── intent_gate.py       # Intent classification (product_query / objection / offtopic)
│   ├── fact_checker.py      # LLM-based final answer generation
│   ├── objection_handler.py # Rule-based objection handling
│   ├── auto_fixer.py        # FSM soft-patching and negative examples
│   ├── policy_controller.py # CTA strategy (soft/direct/urgent/consultative)
│   └── utils.py             # Shared utilities (ID normalization, etc.)
│
├── stores/
│   └── lucky_dog/
│       ├── products_search_optimized.json
│       ├── products_normalized.json
│       ├── objection_examples.json
│       ├── chroma_db/chroma.sqlite3
│       └── users.db
│
└── sessions/                # JSON-based per-user session files
