# /root/ukrsell_v4/core/retrieval.py v9.5.6

from typing import List, Dict, Optional, Any, Union
import os
import json
import faiss
import numpy as np
import re
import asyncio
import sqlite3
import aiosqlite
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from core.logger import logger, log_event, log_retrieval
from core.intelligence import entity_filter, get_stem

# Минимальный final_score для включения результата в выдачу
RELEVANCE_SCORE_THRESHOLD = 0.30


class RetrievalEngine:
    """
    Universal SaaS Retrieval Engine v9.5.6.

    v9.5.6:
      - FIX #2: category=null gate — если category не задана и action=SEARCH,
        возвращает NO_CATEGORY без FAISS. Останавливает поиск мусора.
      - FIX #3: Score threshold RELEVANCE_SCORE_THRESHOLD=0.30 — результаты
        с final_score < порога отсекаются после _semantic_rerank.
        Soft fallback (L1/L2/L3) применяется только для action=SEARCH.
    """

    def __init__(self, ctx: Any, shared_model: Any, shared_translator: Any = None):
        self.ctx = ctx
        self.slug = getattr(ctx, 'slug', 'unknown')
        self.model = shared_model
        self.translator = shared_translator
        self.executor = ThreadPoolExecutor(max_workers=4)

        self.threshold = getattr(ctx, "threshold_main", 0.35)
        self.semantic_filter_threshold = 0.55
        self.troll_threshold_strict = 0.95
        self.troll_threshold_soft = 0.75

        self.language = getattr(ctx, "language", "Ukrainian")

        self.id_list: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.index: Optional[faiss.IndexFlatIP] = None
        self.category_patterns: Dict[str, re.Pattern] = {}
        self.top_brands: List[str] = []
        self.negative_patterns: List[Dict[str, Any]] = []
        self.schema_keys: List[str] = []
        self.intent_mapping: Dict[str, Any] = {}
        self.enrichment_map: Dict[str, List[str]] = {}

        self.products_db_path = os.path.join(ctx.base_path, "products.db")
        self._price_cleaner = re.compile(r'[^\d.,]')
        self._init_error: Optional[Exception] = None

        self._init_task: Optional[asyncio.Task] = asyncio.create_task(
            self._initialize_assets_async()
        )
        self._init_task.add_done_callback(lambda _t: setattr(self, '_init_task', None))

    # ── Инициализация ─────────────────────────────────────────────────────────

    async def _initialize_assets_async(self):
        loop = asyncio.get_running_loop()
        init_start = time.perf_counter()
        try:
            await self._load_id_list_from_db()
            await loop.run_in_executor(self.executor, self._load_assets)
            await loop.run_in_executor(self.executor, self._load_negative_patches)
            await loop.run_in_executor(self.executor, self._load_store_profile)
            await loop.run_in_executor(self.executor, self._load_enrichment_data)

            if not self.id_list:
                await self._finalize_id_mapping()

            init_elapsed = round(time.perf_counter() - init_start, 3)
            if init_elapsed > 5.0:
                logger.warning(
                    f"⚠️ [{self.slug}] Slow init detected: {init_elapsed}s "
                    f"(metadata={len(self.metadata)}, id_list={len(self.id_list)})"
                )
            else:
                logger.info(
                    f"🚀 [{self.slug}] Retrieval Engine v9.5.6 READY in {init_elapsed}s "
                    f"(items={len(self.metadata)}, ids={len(self.id_list)})"
                )

            if hasattr(self.ctx, 'data_ready') and isinstance(self.ctx.data_ready, asyncio.Event):
                self.ctx.data_ready.set()

        except asyncio.CancelledError:
            logger.warning(f"[{self.slug}] Asset initialization cancelled.")
            raise
        except Exception as e:
            self._init_error = e
            logger.error(
                f"[{self.slug}] Async Init Failed: {e}. "
                f"All subsequent searches will return INIT_FAILED.",
                exc_info=True,
            )

    async def _load_id_list_from_db(self, retries: int = 3):
        if not os.path.exists(self.products_db_path):
            logger.warning(
                f"[{self.slug}] products.db not found at {self.products_db_path}. "
                f"Will fall back to id_map.json or metadata.keys()."
            )
            return

        for attempt in range(retries):
            try:
                async with aiosqlite.connect(self.products_db_path, timeout=5.0) as db:
                    async with db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='faiss_map'"
                    ) as cursor:
                        if not await cursor.fetchone():
                            logger.warning(
                                f"[{self.slug}] Table 'faiss_map' missing in products.db."
                            )
                            return

                    async with db.execute(
                        "SELECT product_id FROM faiss_map ORDER BY position ASC"
                    ) as cursor:
                        rows = await cursor.fetchall()
                        self.id_list = [
                            str(row[0]).strip() for row in rows if row[0] is not None
                        ]
                        logger.info(
                            f"[{self.slug}] Loaded {len(self.id_list)} IDs "
                            f"from faiss_map (SQLite)."
                        )
                        return

            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < retries - 1:
                    logger.warning(
                        f"[{self.slug}] DB locked on id_list load, "
                        f"retry {attempt + 1}/{retries}..."
                    )
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                logger.error(f"[{self.slug}] SQLite OperationalError on id_list load: {e}")
                break
            except Exception as e:
                logger.error(f"[{self.slug}] Unexpected error loading faiss_map: {e}")
                break

    async def _finalize_id_mapping(self):
        store_path  = Path(self.ctx.base_path)
        id_map_file = store_path / "id_map.json"

        if id_map_file.exists():
            try:
                def _load_json():
                    with open(id_map_file, "r", encoding="utf-8") as f:
                        return json.load(f)

                data = await asyncio.get_running_loop().run_in_executor(
                    self.executor, _load_json
                )
                loaded = [
                    str(x).strip()
                    for x in (data.get("ids", []) if isinstance(data, dict) else data)
                    if x
                ]
                if loaded:
                    self.id_list = loaded
                    logger.info(
                        f"[{self.slug}] Fallback: loaded {len(self.id_list)} IDs "
                        f"from id_map.json."
                    )
                    return
            except Exception as e:
                logger.error(f"[{self.slug}] id_map.json load error: {e}")

        if self.metadata:
            self.id_list = list(self.metadata.keys())
            logger.warning(
                f"[{self.slug}] Fallback: using metadata.keys() for ID mapping "
                f"({len(self.id_list)} items). FAISS position alignment not guaranteed."
            )

    # ── Загрузка ассетов ──────────────────────────────────────────────────────

    def _load_assets(self):
        """
        FIX v9.5.5: источник данных → deduplicated_products.json.
        Fallback на normalized_products_final.json если deduplicated не найден.
        attributes десериализуется один раз при загрузке.
        """
        store_path = Path(self.ctx.base_path)

        # FIX: приоритет deduplicated_products.json как финального артефакта
        data_file = store_path / "deduplicated_products.json"
        if not data_file.exists():
            data_file = store_path / "normalized_products_final.json"
            if data_file.exists():
                logger.warning(
                    f"[{self.slug}] deduplicated_products.json not found, "
                    f"falling back to normalized_products_final.json"
                )

        index_file = store_path / "faiss.index"

        if data_file.exists():
            try:
                with open(data_file, "r", encoding="utf-8") as f:
                    products_list = json.load(f)
                if isinstance(products_list, list):
                    temp_metadata: Dict[str, Any] = {}
                    skipped      = 0
                    attrs_parsed = 0
                    for item in products_list:
                        raw_id = item.get("product_id") or item.get("id")
                        if raw_id is None or str(raw_id).strip().lower() == "none":
                            skipped += 1
                            continue
                        p_id = str(raw_id).strip()

                        raw_attrs = item.get("attributes")
                        if isinstance(raw_attrs, str) and raw_attrs.strip():
                            try:
                                parsed = json.loads(raw_attrs)
                                item["attributes"] = parsed if isinstance(parsed, dict) else {}
                                attrs_parsed += 1
                            except (json.JSONDecodeError, ValueError):
                                item["attributes"] = {}
                        elif not isinstance(raw_attrs, dict):
                            item["attributes"] = {}

                        temp_metadata[p_id] = item

                    self.metadata = temp_metadata

                    if skipped:
                        logger.warning(
                            f"[{self.slug}] Skipped {skipped} products with invalid ID."
                        )
                    if attrs_parsed:
                        logger.info(
                            f"[{self.slug}] Pre-deserialized attributes for "
                            f"{attrs_parsed} products."
                        )

                self.schema_keys = getattr(self.ctx, "schema_keys", [])

                categories = {
                    str(p.get("category", "")).lower()
                    for p in self.metadata.values()
                    if p.get("category")
                }
                for cat in categories:
                    if cat:
                        self.category_patterns[cat] = re.compile(
                            rf"\b{re.escape(cat)}\b", re.IGNORECASE
                        )
                logger.info(f"[{self.slug}] Metadata loaded: {len(self.metadata)} items.")
            except Exception as e:
                logger.error(f"[{self.slug}] Metadata load error: {e}")

        if index_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                logger.info(f"[{self.slug}] FAISS index loaded.")
            except Exception as e:
                logger.error(f"[{self.slug}] FAISS read error: {e}")

    def _load_enrichment_data(self):
        path = Path(self.ctx.base_path) / "enrichment.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.enrichment_map = json.load(f)
                logger.info(f"[{self.slug}] Enrichment map loaded.")
            except Exception as e:
                logger.error(f"[{self.slug}] Failed to load enrichment.json: {e}")

    def _load_negative_patches(self):
        p = Path(self.ctx.base_path) / "fsm_soft_patch.json"
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    self.negative_patterns = (
                        d.get("fsm_errors", []) or d.get("troll_patterns", [])
                    )
                logger.info(
                    f"[{self.slug}] Negative patches loaded: "
                    f"{len(self.negative_patterns)} patterns."
                )
            except Exception as e:
                logger.error(
                    f"[{self.slug}] Failed to load fsm_soft_patch.json: {e}. "
                    f"Negative intent detection disabled."
                )

    def _load_store_profile(self):
        if hasattr(self.ctx, 'profile') and self.ctx.profile:
            p = self.ctx.profile
            self.top_brands = [
                b.lower()
                for b in p.get("brand_matrix", {}).get("top_brands", [])
            ]
            self.intent_mapping = p.get(
                "intent_mapping", {"brand": ["brand"], "category": ["category"]}
            )

    # ── Утилиты ───────────────────────────────────────────────────────────────

    def _parse_price(self, raw: str) -> Optional[float]:
        cleaned = self._price_cleaner.sub('', str(raw))
        if not cleaned:
            return None
        try:
            has_dot   = '.' in cleaned
            has_comma = ',' in cleaned

            if has_dot and has_comma:
                last_dot   = cleaned.rfind('.')
                last_comma = cleaned.rfind(',')
                if last_comma > last_dot:
                    cleaned = cleaned.replace('.', '').replace(',', '.')
                else:
                    cleaned = cleaned.replace(',', '')
            elif has_comma:
                cleaned = cleaned.replace(',', '.')

            return float(cleaned)
        except (ValueError, TypeError):
            return None

    def _enrich_query(self, query: str) -> str:
        if not self.enrichment_map:
            return query
        tokens    = set(query.lower().split())
        additions = []
        for key, synonyms in self.enrichment_map.items():
            if key in tokens:
                additions.extend(synonyms)
        if not additions:
            return query
        return f"{query} {' '.join(additions)}".strip()

    def _keyword_search(self, query: str, limit: int = 40) -> List[Dict[str, Any]]:
        q_words = [get_stem(w) for w in query.lower().split() if len(w) > 2]
        if not q_words:
            return []

        hits           = []
        items_snapshot = list(self.metadata.items())
        for pid, p in items_snapshot:
            blob = str(
                p.get("search_blob") or f"{p.get('title', '')} {p.get('description', '')}"
            ).lower()
            score = sum(1 for w in q_words if w in blob)
            if score > 0:
                hits.append({"id": pid, "keyword_score": score})

        hits.sort(key=lambda x: x["keyword_score"], reverse=True)
        return hits[:limit]

    def _rrf_merge(
        self,
        vector_hits:  List[Dict],
        keyword_hits: List[Dict],
        k: int = 60,
    ) -> List[str]:
        scores: Dict[str, float] = {}
        for rank, hit in enumerate(vector_hits):
            pid = hit["id"]
            scores[pid] = scores.get(pid, 0) + 1.0 / (k + rank)
        for rank, hit in enumerate(keyword_hits):
            pid = hit["id"]
            scores[pid] = scores.get(pid, 0) + 1.0 / (k + rank)
        return [
            pid
            for pid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:80]
        ]

    def _check_negative_intent(self, q_emb: np.ndarray) -> Optional[str]:
        if not self.negative_patterns:
            return None
        q_norm = q_emb.flatten()
        for patch in self.negative_patterns:
            t_vec = patch.get("vector")
            if not t_vec:
                continue
            similarity = np.dot(q_norm, np.array(t_vec).astype('float32').flatten())
            if similarity > self.troll_threshold_strict:
                return "STRICT_REJECT"
            if similarity > self.troll_threshold_soft:
                return "GRAY_ZONE"
        return None

    def _apply_unified_filters(
        self,
        product:     Dict[str, Any],
        entities:    Dict[str, Any],
        price_limit: Optional[float] = None,
    ) -> bool:
        # 1. Availability
        avail = str(product.get("availability", "instock")).lower()
        if avail in ["outofstock", "немає в наявності", "нет в наличии"]:
            return False

        # 2. Price
        if price_limit is not None:
            raw_price    = product.get("price", "0")
            parsed_price = self._parse_price(str(raw_price))
            if parsed_price is not None and parsed_price > price_limit:
                return False

        # 3. Entity & attribute filters
        if not entities:
            return True

        raw_attrs = product.get('attributes')
        if isinstance(raw_attrs, dict):
            p_attrs = raw_attrs
        elif isinstance(raw_attrs, str) and raw_attrs.strip():
            try:
                parsed_attrs = json.loads(raw_attrs)
                p_attrs = parsed_attrs if isinstance(parsed_attrs, dict) else {}
            except (json.JSONDecodeError, ValueError):
                p_attrs = {}
        else:
            p_attrs = {}

        for key, val in entities.items():
            if val is None or str(val).lower() in [
                "none", "any", "null", "search", "troll", "chat"
            ]:
                continue
            if key in [
                "resolved_product", "target", "action", "entities",
                "properties", "price_limit", "max_price",
            ]:
                continue

            k_lower = str(key).lower()
            v_lower = str(val).lower()

            if v_lower in str(product.get(k_lower, "")).lower():
                continue

            if v_lower in str(p_attrs.get(key, p_attrs.get(k_lower, ""))).lower():
                continue

            if self.schema_keys and any(sk.lower() == k_lower for sk in self.schema_keys):
                return False

        return True

    def _apply_unified_filters_relaxed(
        self,
        product:     Dict[str, Any],
        entities:    Dict[str, Any],
        price_limit: Optional[float] = None,
        skip_properties: bool = False,
    ) -> bool:
        """
        FIX v9.5.5: relaxed фильтр для Search Fallback.
        skip_properties=True — игнорирует properties (цвет, размер), оставляет category.
        Используется в L1 fallback.
        """
        avail = str(product.get("availability", "instock")).lower()
        if avail in ["outofstock", "немає в наявності", "нет в наличии"]:
            return False

        if price_limit is not None:
            raw_price    = product.get("price", "0")
            parsed_price = self._parse_price(str(raw_price))
            if parsed_price is not None and parsed_price > price_limit:
                return False

        if not entities:
            return True

        for key, val in entities.items():
            if val is None or str(val).lower() in [
                "none", "any", "null", "search", "troll", "chat"
            ]:
                continue
            if key in [
                "resolved_product", "target", "action", "entities",
                "properties", "price_limit", "max_price",
            ]:
                continue

            # skip_properties=True: пропускаем все ключи кроме category и brand
            if skip_properties and key not in ("category", "brand"):
                continue

            k_lower = str(key).lower()
            v_lower = str(val).lower()

            if v_lower in str(product.get(k_lower, "")).lower():
                continue

            raw_attrs = product.get('attributes', {})
            p_attrs   = raw_attrs if isinstance(raw_attrs, dict) else {}
            if v_lower in str(p_attrs.get(key, p_attrs.get(k_lower, ""))).lower():
                continue

            if self.schema_keys and any(sk.lower() == k_lower for sk in self.schema_keys):
                return False

        return True

    def _semantic_rerank(
        self,
        hits:         List[Dict[str, Any]],
        target:       str,
        detected_cat: str,
        target_vec:   Optional[np.ndarray] = None,
        query_words:  Optional[List[str]]  = None,
    ) -> List[Dict[str, Any]]:
        t_lower     = target.lower() if target else ""
        c_lower     = detected_cat.lower() if detected_cat else ""
        query_words = query_words or []

        cat_map    = getattr(self.ctx, "category_map", {})
        valid_cats = {c_lower}
        if c_lower in cat_map:
            m = cat_map[c_lower]
            if isinstance(m, list):
                valid_cats.update(str(x).lower() for x in m)
            else:
                valid_cats.add(str(m).lower())

        for hit in hits:
            prod  = hit["product"]
            title = str(prod.get("title") or prod.get("name", "")).lower()
            brand = str(prod.get("brand", "")).lower()
            p_cat = str(prod.get("category", "")).lower()

            vec_sim = float(hit.get("score", 0))
            if (
                target_vec is not None
                and t_lower
                and t_lower not in str(prod.get("search_blob") or title).lower()
            ):
                vec_sim *= 0.7

            t_bonus = 0.4 if t_lower and (t_lower in title or t_lower in brand) else 0.0
            c_bonus = 0.3 if any(vc == p_cat or vc in p_cat for vc in valid_cats) else 0.0
            b_mult  = 1.2 if brand in self.top_brands else 1.0

            d_count = sum(
                1 for w in query_words
                if len(w) > 2 and (w in title or get_stem(w) in title)
            )
            d_bonus = min(d_count * 0.15, 0.45)

            hit["final_score"] = round(
                ((vec_sim * 0.2) + t_bonus + c_bonus + d_bonus) * b_mult, 4
            )

        hits.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        return hits

    # ── Основной поиск ────────────────────────────────────────────────────────

    async def search(
        self, query: str, entities: Dict[str, Any] = None, top_k: int = 5
    ) -> Dict:
        """
        Production Hybrid Search Pipeline v9.5.5.

        FIX v9.5.5: Three-level Search Fallback при Candidates=0:
          L0: полный поиск с entities + properties (штатный путь).
          L1: поиск без properties, только category + price_limit.
          L2: поиск без category и properties, только FAISS по тексту запроса.
          L3: топ reranked без entity-фильтров (last resort).
        """
        start_time = time.perf_counter()

        if self._init_task is not None and not self._init_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._init_task), timeout=10.0)
            except asyncio.TimeoutError:
                logger.error(
                    f"🛑 [{self.slug}] Search timeout: Assets not ready after 10s."
                )
                return {
                    "status":       "NO_ASSETS",
                    "products":     [],
                    "all_products": [],
                    "is_empty":     True,
                }
            except Exception as e:
                logger.error(f"[{self.slug}] Init task wait error: {e}")

        if self._init_error is not None:
            logger.error(
                f"🛑 [{self.slug}] Search aborted: init failed with "
                f"{type(self._init_error).__name__}: {self._init_error}"
            )
            return {
                "status":       "INIT_FAILED",
                "products":     [],
                "all_products": [],
                "is_empty":     True,
            }

        id_list_snap:  List[str]                   = self.id_list
        metadata_snap: Dict[str, Any]              = self.metadata
        index_snap:    Optional[faiss.IndexFlatIP] = self.index

        if not index_snap or not id_list_snap:
            logger.error(f"🛑 [{self.slug}] Search failed: Index/ID list missing.")
            return {"status": "NO_ASSETS", "products": [], "all_products": []}

        if index_snap.ntotal != len(id_list_snap):
            logger.error(
                f"🛑 [{self.slug}] FAISS/ID mapping desync: "
                f"index.ntotal={index_snap.ntotal}, id_list={len(id_list_snap)}. "
                f"Rebuild the FAISS index and faiss_map table."
            )
            return {
                "status":       "MAPPING_ERROR",
                "products":     [],
                "all_products": [],
                "is_empty":     True,
            }

        loop = asyncio.get_running_loop()
        try:
            ent    = entities or {}
            target = str(ent.get("resolved_product") or ent.get("target") or "").lower()
            action = ent.get("action", "SEARCH")

            logger.info(
                f"🔎 [{self.slug}] RETRIEVAL_START | "
                f"Target: '{target}' | Action: {action} | Query: '{query[:50]}'"
            )

            if action == "TROLL":
                return {
                    "status":       "SEMANTIC_REJECT",
                    "products":     [],
                    "all_products": [],
                    "is_troll":     True,
                }

            # FIX #2: category=null gate — нет категории = нет поиска
            # Исключение: если есть явный target или brand — ищем по тексту
            det_cat_from_ent = ent.get("category")
            has_target       = bool(ent.get("resolved_product") or ent.get("target"))
            has_brand        = bool(ent.get("brand"))
            if (
                det_cat_from_ent is None
                and not has_target
                and not has_brand
                and action not in ("CONSULT",)
            ):
                logger.info(
                    f"[{self.slug}] category=null gate: no category/target/brand. "
                    f"Returning NO_CATEGORY."
                )
                return {
                    "status":          "NO_CATEGORY",
                    "products":        [],
                    "all_products":    [],
                    "is_empty":        True,
                    "fallback_level":  0,
                    "fallback_reason": "no_category",
                }

            enriched_q = self._enrich_query(query)
            det_cat    = ent.get("category") or self._detect_category(query)

            search_text = target if len(target) > 2 else enriched_q
            q_emb = await loop.run_in_executor(
                self.executor,
                lambda: self.model.encode(
                    f"query: {search_text}", normalize_embeddings=True
                ).astype('float32'),
            )

            if self._check_negative_intent(q_emb) == "STRICT_REJECT":
                logger.warning(f"🛡️ [{self.slug}] Negative intent blocked (Strict).")
                return {"status": "SEMANTIC_REJECT", "products": [], "all_products": []}

            scores, idxs = await loop.run_in_executor(
                self.executor,
                lambda: index_snap.search(np.expand_dims(q_emb, axis=0), 60),
            )

            vector_hits: List[Dict]      = []
            v_map:       Dict[str, float] = {}
            for score, idx in zip(scores[0], idxs[0]):
                if idx != -1 and idx < len(id_list_snap):
                    pid = id_list_snap[idx]
                    vector_hits.append({"id": pid, "score": float(score)})
                    v_map[pid] = float(score)

            kw_hits    = await loop.run_in_executor(
                self.executor, self._keyword_search, enriched_q, 40
            )
            merged_ids = self._rrf_merge(vector_hits, kw_hits)

            t_vec = (
                await loop.run_in_executor(
                    self.executor,
                    lambda: self.model.encode(
                        f"query: {target}", normalize_embeddings=True
                    ),
                )
                if target
                else None
            )
            q_words = [
                w.lower()
                for w in query.split()
                if len(w) > 2
                and w.lower() not in {"я", "мені", "хочу", "треба", "купити"}
            ]

            raw_p_limit  = ent.get("price_limit") or ent.get("max_price")
            price_limit: Optional[float] = None
            if raw_p_limit:
                price_limit = self._parse_price(str(raw_p_limit))
                if price_limit is None:
                    logger.warning(
                        f"[{self.slug}] Could not parse price_limit: '{raw_p_limit}'"
                    )

            # ── L0: полный поиск с entities + properties ──────────────────────
            candidate_list: List[Dict] = []
            for pid in merged_ids:
                p = metadata_snap.get(pid)
                if p and self._apply_unified_filters(p, ent, price_limit):
                    candidate_list.append({
                        "product": p,
                        "score":   v_map.get(pid, 0.1),
                        "id":      pid,
                    })

            reranked = self._semantic_rerank(
                candidate_list, target, det_cat or "", t_vec, q_words
            )

            # FIX #3: отсекаем результаты ниже порога релевантности
            reranked = [
                r for r in reranked
                if r.get("final_score", 0) >= RELEVANCE_SCORE_THRESHOLD
            ]
            if len(candidate_list) > 0 and not reranked:
                logger.info(
                    f"[{self.slug}] Score threshold ({RELEVANCE_SCORE_THRESHOLD}) "
                    f"dropped all {len(candidate_list)} candidates."
                )

            intel_filtered = entity_filter(
                [r["product"] for r in reranked],
                ent,
                intent_mapping=self.intent_mapping,
                category_map=getattr(self.ctx, "category_map", {}),
            )
            intel_ids = {
                str(p.get("product_id") or p.get("id")) for p in intel_filtered
            }
            final_products = [
                r for r in reranked
                if str(r["product"].get("product_id") or r["product"].get("id"))
                in intel_ids
            ]

            # ── Soft Fallback (оригинальный) ──────────────────────────────────
            if not final_products and reranked:
                best_score = reranked[0].get("final_score", 0)
                if best_score > self.semantic_filter_threshold:
                    final_products = reranked[:top_k]
                    logger.info(
                        f"⚠️ [{self.slug}] Soft Fallback activated. "
                        f"Best score: {best_score}"
                    )

            fallback_level  = 0
            fallback_reason = None

            # FIX v9.5.6: Three-level Search Fallback только для SEARCH
            if not final_products and merged_ids and action == "SEARCH":

                # L1: убираем properties, оставляем category + price_limit
                has_properties = bool(
                    ent.get("properties") or
                    any(k not in ("category", "brand", "price_limit", "properties",
                                  "resolved_product", "target", "action")
                        for k in ent.keys())
                )

                if has_properties:
                    fallback_level  = 1
                    fallback_reason = "no_results_with_properties"
                    logger.info(
                        f"⚠️ [{self.slug}] Fallback L1: retrying without properties "
                        f"(category='{det_cat}')."
                    )
                    l1_candidates: List[Dict] = []
                    for pid in merged_ids:
                        p = metadata_snap.get(pid)
                        if p and self._apply_unified_filters_relaxed(
                            p, ent, price_limit, skip_properties=True
                        ):
                            l1_candidates.append({
                                "product": p,
                                "score":   v_map.get(pid, 0.1),
                                "id":      pid,
                            })
                    if l1_candidates:
                        final_products = self._semantic_rerank(
                            l1_candidates, target, det_cat or "", t_vec, q_words
                        )[:top_k]

                # L2: убираем category и properties, только FAISS по тексту
                if not final_products:
                    fallback_level  = 2
                    fallback_reason = "no_results_without_properties"
                    logger.info(
                        f"⚠️ [{self.slug}] Fallback L2: retrying FAISS-only "
                        f"(no category, no properties)."
                    )
                    l2_candidates: List[Dict] = []
                    for pid in merged_ids:
                        p = metadata_snap.get(pid)
                        if p:
                            avail = str(p.get("availability", "instock")).lower()
                            if avail in ["outofstock", "немає в наявності", "нет в наличии"]:
                                continue
                            if price_limit is not None:
                                raw_price = p.get("price", "0")
                                parsed    = self._parse_price(str(raw_price))
                                if parsed is not None and parsed > price_limit:
                                    continue
                            l2_candidates.append({
                                "product": p,
                                "score":   v_map.get(pid, 0.1),
                                "id":      pid,
                            })
                    if l2_candidates:
                        final_products = self._semantic_rerank(
                            l2_candidates, target, det_cat or "", t_vec, q_words
                        )[:top_k]

                # L3: топ reranked без фильтров (last resort)
                if not final_products and reranked:
                    fallback_level  = 3
                    fallback_reason = "no_results_faiss_only"
                    logger.info(
                        f"⚠️ [{self.slug}] Fallback L3: returning top reranked "
                        f"(last resort, no filters)."
                    )
                    final_products = reranked[:top_k]

            top_results    = final_products[:top_k]
            execution_time = round(time.perf_counter() - start_time, 3)

            logger.info(
                f"✅ [{self.slug}] RETRIEVAL_END | "
                f"Time: {execution_time}s | "
                f"FAISS: {len(vector_hits)} | "
                f"Candidates: {len(candidate_list)} | "
                f"After intel: {len(final_products)} | "
                f"Final: {len(top_results)}"
                + (f" | Fallback L{fallback_level}" if fallback_level else "")
            )

            log_retrieval(
                slug=self.slug,
                query_preview=query[:50],
                faiss_candidates=len(vector_hits),
                after_entity_filter=len(final_products),
                after_price_filter=len(candidate_list),
                final_count=len(top_results),
                detected_category=det_cat or "none",
            )

            return {
                "status":            "SUCCESS" if top_results else "NOT_FOUND",
                "products":          top_results,
                "all_products":      reranked[:20],
                "detected_category": det_cat,
                "is_empty":          not top_results,
                "execution_time":    execution_time,
                "fallback_level":    fallback_level,
                "fallback_reason":   fallback_reason,
            }

        except Exception as e:
            logger.error(f"[{self.slug}] Search Pipeline Crash: {e}", exc_info=True)
            return {"status": "ERROR", "products": [], "all_products": [], "is_empty": True}

    # ── Вспомогательные методы ────────────────────────────────────────────────

    def _detect_category(self, query: str) -> Optional[str]:
        for cat, pattern in self.category_patterns.items():
            if pattern.search(query):
                return cat
        return None

    # ── Закрытие ──────────────────────────────────────────────────────────────

    def close(self):
        if self._init_task is not None and not self._init_task.done():
            self._init_task.cancel()
            logger.info(f"[{self.slug}] Init task cancellation requested (best-effort).")

        self.executor.shutdown(wait=False)
        self.index = None
        self.id_list.clear()
        self.metadata.clear()
        logger.info(f"🛑 [{self.slug}] Retrieval Engine v9.5.6 closed.")

    def __repr__(self):
        return (
            f"<RetrievalEngine v9.5.6 slug={self.slug} items={len(self.metadata)}>"
        )