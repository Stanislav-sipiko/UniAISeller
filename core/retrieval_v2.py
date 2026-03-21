# -*- coding: utf-8 -*-
# /root/ukrsell_v4/core/retrieval_v2.py v3.3.3
"""
LLM Reasoning Engine v3.3.1 — The single "brain" of the system.

This module is the core intelligence of the Ukrsell V4 project. 
It replaces multiple legacy stages (classification, rewriting, entity extraction) 
with a streamlined two-step LLM reasoning process:

1. Intent Pre-check:
   - Analyzes conversation history and the raw user query.
   - Detects if the user is "trolling", asking an absurd question, or needs clarification.
   - Identifies the product category to enrich the subsequent vector search.
   - Detects "availability" requests (e.g., "do you have...").

2. FAISS Retrieval & LLM Reranking:
   - Fetches potential candidates from the FAISS vector index.
   - Uses a "heavy" LLM to filter, rank, and select the most relevant products.
   - Formulates a natural, friendly response as a pet store consultant.

Changelog v3.1.0:
  - Added clarify loop guard: counts only '?' in consultant lines to prevent endless loops.
  - Introduced force_search: strictly blocks 'clarify' or 'empty' modes at the pre_check level.
  - Enhanced faiss_query: now enriched with 'category' detected by LLM.
  - Normalized category mapping: used _CAT_MAP to ensure stability.
  - Sanitized product_ids: regex-based extraction to handle various LLM output formats.
  - Improved _trim_history: now trims by lines to maintain structure.

Changelog v3.2.0:
  - Fixed a bug where product_ids extraction was too restrictive.
  - Improved logging for force_search activation.

Changelog v3.3.3:
  - _llm_pre_check: get_heavy() → get_fast() — классификация не требует тяжёлой модели,
    экономим ресурсы и снижаем latency на ~1.5-2s для troll/clarify путей
  - Все asyncio.wait_for таймауты унифицированы до 25.0s (pre_check и rerank)
  - _DEFAULT_CATALOG_FIELDS = ["category"] — ядро agnostic,
    animal/subtype/brand подтягиваются только из search_config.json магазина

Changelog v3.3.2:
  - _build_non_search_result: elapsed_ms стал опциональным (default=0.0) —
    rerank больше не передаёт заглушку 0
  - _llm_rerank: возвращён asyncio.wait_for(timeout=20.0) — защита от зависания LLM
  - _llm_pre_check: возвращён asyncio.wait_for(timeout=10.0) — аналогично
"""

import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Core project components
# We assume core.logger is properly configured in the environment
try:
    from core.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger("retrieval_v2")

# Configuration constants for reasoning and retrieval
_LLM_MAX_TOKENS  = 1024
_FAISS_TOP_K     = 15
_HISTORY_MAX_LEN = 3000

# Default fields to include in the LLM catalog context
_DEFAULT_CATALOG_FIELDS = ["category"]


def _trim_history(history: str, max_chars: int = _HISTORY_MAX_LEN) -> str:
    """
    Trims the conversation history to stay within context limits.
    Processes the history from the most recent message upwards (bottom-up).
    
    Safety features:
    - Maintains whole lines to avoid breaking JSON or text structures.
    - If a single line is longer than max_chars, it's forcibly truncated.
    """
    if not history:
        return ""
    
    # Fast path: no trimming needed
    if len(history) <= max_chars:
        return history

    lines = history.strip().split('\n')
    result = []
    current_length = 0

    # Iterate backwards to keep the most recent context
    for line in reversed(lines):
        # +1 accounts for the newline character
        line_len = len(line) + 1 
        
        if current_length + line_len > max_chars:
            # Case: the very first (most recent) line is already too long
            if not result:
                # Truncate the line and add ellipsis
                return line[-(max_chars - 3):] + "..."
            break
            
        result.append(line)
        current_length += line_len

    # Re-reverse to restore chronological order
    return '\n'.join(reversed(result))


class LLMRetrievalEngine:
    """
    The main reasoning engine class.
    Coordinates between vector indices, database metadata, and LLM providers.
    """

    def __init__(self, base_engine: Any, selector: Any):
        """
        Initializes the engine with necessary dependencies.
        
        :param base_engine: The underlying engine containing FAISS index and metadata.
        :param selector: LLM selector instance (provides heavy/light models).
        """
        self._base             = base_engine
        self.slug              = base_engine.slug
        self.selector          = selector
        self.language          = getattr(base_engine, 'language', 'Ukrainian')
        
        # Internal cache for instructions and configuration
        self._prompt           = None
        self._prompt_path      = os.path.join("/root/ukrsell_v4/stores", self.slug, "llm_retrieval_prompt.md")
        
        # Category hints mapping: category_name -> list of subtypes
        self.category_hints: Dict[str, List[str]] = {}
        
        # Search parameters loaded from search_config.json
        self.catalog_fields: List[str] = _DEFAULT_CATALOG_FIELDS
        self._consultant_label  = "консультант"  # Used for the loop guard

    async def warm_up(self):
        """
        Warms up the engine by loading configurations and pre-calculating category hints.
        Should be called during store initialization.
        """
        start = time.perf_counter()
        
        # Load custom search settings (fields, labels)
        await self._load_search_config()
        
        # Analyze product DB to build category-subtype suggestions
        await self._load_category_subtypes()
        
        elapsed = round((time.perf_counter() - start) * 1000, 1)
        logger.info(f"[{self.slug}] LLMRetrievalEngine warm_up finished in {elapsed}ms")

    async def _load_search_config(self):
        """
        Loads store-specific search configuration from search_config.json.
        Configurable fields:
        - catalog_fields: attributes to show to the LLM.
        - label_consultant: the name the AI uses in the chat history.
        """
        config_path = os.path.join("/root/ukrsell_v4/stores", self.slug, "search_config.json")
        if not os.path.exists(config_path):
            logger.warning(f"[{self.slug}] search_config.json missing — using project defaults")
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.catalog_fields     = config.get("catalog_fields", _DEFAULT_CATALOG_FIELDS)
            self._consultant_label  = config.get("label_consultant", "консультант").lower()
            
            logger.info(f"[{self.slug}] Search config applied. Catalog fields: {self.catalog_fields}")
        except Exception as e:
            logger.error(f"[{self.slug}] CRITICAL: Failed to load search_config: {e}")

    async def _load_category_subtypes(self):
        """
        Queries the local SQLite database to find the most frequent subtypes for each category.
        This data is used to provide 'hints' when the user's query is too broad.
        """
        db_path = os.path.join("/root/ukrsell_v4/stores", self.slug, "products.db")
        if not os.path.exists(db_path):
            logger.debug(f"[{self.slug}] No products.db found for subtype analysis")
            return

        try:
            import aiosqlite
            from collections import defaultdict
            temp_hints = defaultdict(list)
            
            async with aiosqlite.connect(db_path, timeout=15.0) as db:
                db.row_factory = aiosqlite.Row
                # Select top 3 subtypes per category by product count
                async with db.execute("""
                    SELECT category, subtype, COUNT(*) as cnt
                    FROM products
                    WHERE category IS NOT NULL AND subtype IS NOT NULL
                    GROUP BY category, subtype
                    ORDER BY category, cnt DESC
                """) as cursor:
                    rows = await cursor.fetchall()
                    
            for row in rows:
                category = str(row['category']).lower().strip()
                subtype  = str(row['subtype']).strip()
                
                # We limit hints to 3 items per category to keep the prompt clean
                if len(temp_hints[category]) < 3 and subtype not in temp_hints[category]:
                    temp_hints[category].append(subtype)
            
            self.category_hints = dict(temp_hints)
            logger.info(f"[{self.slug}] Category hint map built for {len(self.category_hints)} categories")
            
        except Exception as e:
            logger.warning(f"[{self.slug}] Category subtype analysis failed: {e}")

    async def search(
        self,
        query:    str,
        entities: Dict[str, Any] = None,
        top_k:    int = 5,
        history:  str = "",
    ) -> Dict:
        """
        The primary search interface. Executes the full reasoning pipeline.
        
        :param query: Raw user input string.
        :param entities: Pre-extracted entities (optional, mostly for logging/legacy).
        :param top_k: Desired number of products to return.
        :param history: Formatted chat history.
        :return: A dictionary containing the final response status, products, and message.
        """
        start_time = time.perf_counter()
        
        # --- Step 0: Clarify Loop Guard (Safety) ---
        # If the consultant has asked questions several times in a row, we force a search
        # to prevent "clarification hell".
        force_search = False
        if history:
            # We check the last 6 segments of history
            recent_segments = history.strip().split('\n')[-6:]
            # A 'clarify' action usually contains the consultant's name and a question mark
            clarify_count = sum(1 for line in recent_segments 
                                if self._consultant_label in line.lower() and '?' in line)
            
            if clarify_count >= 2:
                force_search = True
                logger.info(f"[{self.slug}] Loop Guard: force_search=True due to {clarify_count} clarifications")

        # --- Step 1: LLM Pre-check (Intent & Routing) ---
        # We determine if the user is looking for products or just chatting.
        pre_check = await self._llm_pre_check(query, history)
        
        # Override based on force_search flag
        if force_search:
            pre_check["force_search"] = True
            if pre_check.get("mode") in ("clarify", "empty"):
                logger.info(f"[{self.slug}] Overriding {pre_check['mode']} mode to 'search' due to Loop Guard")
                pre_check["mode"] = "search"

        mode = pre_check.get("mode", "search")

        # Immediate return for non-retrieval intents
        if mode in ("troll", "clarify", "empty"):
            elapsed = round((time.perf_counter() - start_time) * 1000, 1)
            return self._build_non_search_result(pre_check, elapsed)

        # --- Step 2: Vector Search (FAISS) ---
        # We enrich the vector query with the LLM-detected category to improve recall.
        enriched_query = f"{query} {pre_check.get('category') or ''}".strip()
        candidates = await self._faiss_raw(enriched_query, _FAISS_TOP_K)
        
        if not candidates:
            logger.warning(f"[{self.slug}] Vector search returned zero candidates for: '{enriched_query}'")
            return {
                "status":       "EMPTY",
                "products":     [],
                "all_products": [],
                "is_empty":     True,
                "explanation":  "На жаль, за вашим запитом нічого не знайдено. Спробуйте змінити опис. 🐾",
            }

        # --- Step 3: LLM Reranking & Final Decision ---
        # The 'heavy' model looks at actual product data and decides what to show.
        result = await self._llm_rerank(
            query=query,
            candidates=candidates,
            history=history,
            top_k=top_k,
            pre_check=pre_check
        )

        final_elapsed = round((time.perf_counter() - start_time) * 1000, 1)
        logger.info(f"[{self.slug}] Reasoning pipe finished: mode={result.get('response_mode')} | {final_elapsed}ms")
        
        return result

    async def _faiss_raw(self, query: str, top_k: int) -> List[Dict]:
        """
        Executes raw FAISS search using the store's shared index.
        Thread-safe execution using the base engine's executor.
        """
        base = self._base
        
        # Check if index is ready (might be loading or updating)
        if not base._ready_event.is_set():
            logger.info(f"[{self.slug}] FAISS index not ready, waiting...")
            try:
                await asyncio.wait_for(base._ready_event.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                logger.error(f"[{self.slug}] Timeout waiting for FAISS index readiness")
                return []

        try:
            import numpy as np
            loop = asyncio.get_running_loop()
            
            # 1. Encode query (asymmetric search: 'query: ' prefix is used for E5/BGE models)
            # We use the shared thread pool executor to avoid blocking the event loop.
            q_emb = await loop.run_in_executor(
                base.executor,
                lambda: base.model.encode(f"query: {query}", normalize_embeddings=True).astype('float32')
            )
            
            # 2. Search index
            scores, idxs = await loop.run_in_executor(
                base.executor,
                lambda: base.index.search(np.expand_dims(q_emb, axis=0), top_k)
            )
            
            # Basic validation of FAISS output
            if scores is None or idxs is None or len(scores) == 0:
                return []

            # 3. Map indices to metadata
            results = []
            id_list_snap = list(base.id_list) # Snapshot to prevent mutation issues
            meta_snap    = dict(base.metadata)
            
            for score, idx in zip(scores[0], idxs[0]):
                if idx == -1 or idx >= len(id_list_snap):
                    continue
                
                pid = id_list_snap[idx]
                product_data = meta_snap.get(pid)
                if product_data:
                    results.append({
                        "product": product_data,
                        "score":   float(score),
                        "id":      pid
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"[{self.slug}] FAISS execution error: {e}")
            return []

    async def _llm_pre_check(self, query: str, history: str) -> Dict:
        """
        The first stage LLM call. Performs intent classification and routing.
        """
        store_instructions = await self._get_store_prompt()
        lang_note = "Respond in Ukrainian." if self.language == "Ukrainian" else "Respond in Russian."
        
        history_block = f"CONVERSATION HISTORY:\n{_trim_history(history)}\n\n" if history else ""
        
        prompt = f"""{lang_note}
{store_instructions}

{history_block}USER QUERY: "{query}"

Analyze the user's intent. 
Respond ONLY with a JSON object.

JSON structure:
{{
  "mode": "search|availability|clarify|troll|empty",
  "category": "apparel|walking|beds & furniture|feeding|grooming|toys|null",
  "message": "a friendly short message for the user",
  "reason": "internal log explaining your decision"
}}

Definitions:
- 'search': looking for specific items or types.
- 'availability': checking if a generic item type exists.
- 'clarify': the query is too vague (e.g., "for dog") and needs more info.
- 'troll': query is offensive, spam, or totally unrelated to pets.
- 'empty': query is related but obviously not in a pet store catalog.
"""

        # Map for normalizing LLM category output to internal IDs
        _CAT_MAP = {
            "одяг": "apparel", "одежда": "apparel", "apparel": "apparel",
            "walking": "walking", "прогулянки": "walking", "прогулки": "walking",
            "beds & furniture": "beds & furniture", "лежаки": "beds & furniture", "мебель": "beds & furniture",
            "feeding": "feeding", "харчування": "feeding", "корм": "feeding",
            "grooming": "grooming", "грумінг": "grooming", "уход": "grooming",
            "toys": "toys", "іграшки": "toys", "игрушки": "toys"
        }

        try:
            client, model = await self.selector.get_fast()
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": "You are a pet store router."}, 
                              {"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=250
                ),
                timeout=25.0
            )
            
            content = response.choices[0].message.content
            # Cleanup Markdown code blocks if LLM included them
            clean_json = re.sub(r'```json\s*|\s*```', '', content).strip()
            result = json.loads(clean_json)
            
            # Validate 'mode'
            mode = result.get("mode")
            if mode not in ("search", "availability", "clarify", "troll", "empty"):
                logger.warning(f"[{self.slug}] LLM returned unknown mode: {mode}. Defaulting to 'search'.")
                result["mode"] = "search"

            # Normalize and log unexpected categories
            raw_cat = str(result.get("category") or "").lower().strip()
            mapped_cat = _CAT_MAP.get(raw_cat)
            if raw_cat and raw_cat != "null" and not mapped_cat:
                logger.warning(f"[{self.slug}] LLM returned unmapped category: '{raw_cat}'")
            
            result["category"] = mapped_cat
            return result
            
        except Exception as e:
            logger.warning(f"[{self.slug}] Pre-check LLM call failed: {e}. Falling back to 'search' mode.")
            return {"mode": "search", "category": None, "message": "", "reason": "error_fallback"}

    def _build_non_search_result(self, pre: Dict, elapsed_ms: float = 0.0) -> Dict:
        """
        Constructs standard response dictionaries for bypass modes (troll, clarify, empty).
        """
        mode = pre.get("mode", "clarify")
        message = pre.get("message") or "Будь ласка, розкажіть детальніше, що саме ви шукаєте. 🐾"
        
        logger.info(f"[{self.slug}] Intent Bypass triggered: mode={mode} | {elapsed_ms}ms")
        
        if mode == "troll":
            return {"status": "TROLL", "is_empty": True, "message": message}
            
        if mode == "empty":
            return {"status": "EMPTY", "is_empty": True, "explanation": message}

        # Handle 'clarify' mode with hints
        category = pre.get("category")
        hints = ""
        if category and category in self.category_hints:
            hints = ", ".join(self.category_hints[category])
            
        return {
            "status":   "CLARIFY",
            "is_empty": True,
            "question": message,
            "category": category,
            "hints":    hints
        }

    async def _get_store_prompt(self) -> str:
        """
        Retrieves the customized store persona prompt from disk.
        Results are cached in memory for performance.
        """
        if self._prompt:
            return self._prompt
            
        if not os.path.exists(self._prompt_path):
            logger.debug(f"[{self.slug}] Custom persona prompt not found at {self._prompt_path}")
            return ""

        try:
            with open(self._prompt_path, 'r', encoding='utf-8') as f:
                self._prompt = f.read()
            return self._prompt
        except Exception as e:
            logger.error(f"[{self.slug}] Error reading persona file: {e}")
            return ""

    async def _llm_rerank(
        self,
        query: str,
        candidates: List[Dict],
        history: str,
        top_k: int,
        pre_check: Dict
    ) -> Dict:
        """
        The second stage LLM call (Reranker). 
        Evaluates physical product attributes against user preferences.
        """
        store_instructions = await self._get_store_prompt()
        lang_note = "Respond in Ukrainian." if self.language == "Ukrainian" else "Respond in Russian."
        force_search = pre_check.get("force_search", False)

        # 1. Prepare Mini-Catalog for the LLM
        # We use numeric IDs (1, 2, 3...) to reduce token usage and improve LLM reference accuracy.
        catalog_for_llm = []
        idx_map = {}
        
        for i, hit in enumerate(candidates, start=1):
            product = hit.get("product") or hit
            str_idx = str(i)
            
            # Essential data
            entry = {
                "id":    str_idx,
                "title": product.get("title", "Unknown Product"),
                "price": product.get("price", 0)
            }
            
            # Add configured metadata fields
            for field in self.catalog_fields:
                val = product.get(field) or (product.get("attributes") or {}).get(field)
                if val:
                    entry[field] = val
                    
            catalog_for_llm.append(entry)
            idx_map[str_idx] = hit

        history_block = f"CONVERSATION HISTORY:\n{_trim_history(history)}\n\n" if history else ""
        
        # Inject Force Search rule if Loop Guard is active
        force_rule = ""
        if force_search:
            force_rule = "⚠️ CRITICAL: You have clarified too many times. You MUST pick products now. Do NOT return 'clarify'."

        prompt = f"""{lang_note}
{store_instructions}
{force_rule}

{history_block}USER QUERY: "{query}"

CANDIDATE PRODUCTS:
{json.dumps(catalog_for_llm, ensure_ascii=False)}

TASK:
1. Select the most relevant product IDs from the list.
2. If multiple products fit, list them all.
3. Respond ONLY in JSON.

JSON Schema:
{{
  "mode": "products|availability|clarify|empty",
  "message": "Friendly response mentioning selected items",
  "product_ids": ["1", "2"],
  "reason": "logic explanation"
}}

Rules:
- 'product_ids' MUST be a list of strings containing ONLY the candidate IDs provided.
- If force search is active, mode MUST be 'products' if any candidates are even remotely relevant.
"""

        try:
            client, model = await self.selector.get_heavy()
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=_LLM_MAX_TOKENS
                ),
                timeout=25.0
            )
            
            raw_content = response.choices[0].message.content
            clean_json = re.sub(r'```json\s*|\s*```', '', raw_content).strip()
            result = json.loads(clean_json)
            
            # --- Logic Decision Tree ---
            mode        = result.get("mode", "products")
            message     = result.get("message", "")
            product_ids = result.get("product_ids", [])
            
            # Apply Loop Guard override
            if force_search and mode in ("clarify", "empty") and catalog_for_llm:
                logger.info(f"[{self.slug}] Loop Guard override in Rerank phase: {mode} -> 'products'")
                mode = "products"
                if not product_ids:
                    product_ids = ["1"] # Fallback to first candidate

            # Handle clarification requests
            if mode == "clarify" and not force_search:
                return self._build_non_search_result({
                    "mode": "clarify", 
                    "message": message, 
                    "category": pre_check.get("category")
                }, 0)
                
            # Handle empty results
            if mode == "empty" or (not product_ids and mode in ("products", "availability")):
                return {
                    "status":       "EMPTY",
                    "products":     [],
                    "all_products": [],
                    "is_empty":     True,
                    "explanation":  message or "На жаль, за цими параметрами нічого не підійшло. 🐾",
                }

            # Map ordinal IDs back to original product metadata and FAISS scores
            final_selection = []
            seen_ids = set()
            
            for raw_pid in product_ids:
                # Sanitize: extract digits (e.g., handle "ID 1" or "item 2")
                match = re.search(r'\d+', str(raw_pid))
                if not match:
                    continue
                
                clean_pid = match.group()
                if clean_pid in idx_map and clean_pid not in seen_ids:
                    hit = idx_map[clean_pid]
                    final_selection.append({
                        "product": hit.get("product"),
                        "score":   hit.get("score", 0.5),
                        "id":      hit.get("id")
                    })
                    seen_ids.add(clean_pid)

            # Verification of final list
            if not final_selection:
                logger.warning(f"[{self.slug}] Reranker returned invalid IDs {product_ids}. Mapping failed.")
                return {
                    "status":       "EMPTY",
                    "is_empty":     True,
                    "explanation":  message or "Технічна помилка при виборі товарів.",
                }

            return {
                "status":          "FOUND",
                "products":        final_selection,
                "all_products":    final_selection,
                "is_empty":        False,
                "explanation":     message,
                "response_mode":   mode,
                "is_alternative":  False
            }

        except Exception as e:
            logger.error(f"[{self.slug}] CRITICAL Rerank error: {e}")
            # Safe recovery: return the top 3 FAISS results
            recovery_hits = []
            for candidate in candidates[:3]:
                recovery_hits.append({
                    "product": candidate.get("product"),
                    "score":   candidate.get("score", 0.0),
                    "id":      candidate.get("id")
                })
                
            return {
                "status":       "FOUND",
                "products":     recovery_hits,
                "all_products": recovery_hits,
                "is_empty":     False,
                "explanation":  "Ось найкращі варіанти, які мені вдалося знайти: 🐾",
                "reason":       "exception_recovery"
            }

def get_version() -> str:
    """Returns the module version string."""
    return "3.3.3"

# End of retrieval_v2.py