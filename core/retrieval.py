# -*- coding: utf-8 -*-
# /root/ukrsell_v4/core/retrieval.py v9.3.0
from typing import List, Dict, Optional, Any, Union
import os
import json
import faiss
import numpy as np
import re
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from core.logger import logger, log_event, log_retrieval
from core.intelligence import entity_filter, get_stem

class RetrievalEngine:
    """
    Universal SaaS Retrieval Engine v9.3.0 [ASYNC OPTIMIZED].
    - Исправлена блокировка Event Loop при загрузке и поиске.
    - Hybrid Search: Vector (FAISS) + Keyword (Local).
    - RRF Rank Fusion для объединения результатов.
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

        # Запуск асинхронной инициализации
        asyncio.create_task(self._initialize_assets_async())

    async def _initialize_assets_async(self):
        """Asynchronous assets loading to prevent Event Loop lag."""
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self.executor, self._load_assets)
            await loop.run_in_executor(self.executor, self._load_negative_patches)
            await loop.run_in_executor(self.executor, self._load_store_profile)
            await loop.run_in_executor(self.executor, self._load_enrichment_data)

            if hasattr(self.ctx, 'data_ready') and isinstance(self.ctx.data_ready, asyncio.Event):
                self.ctx.data_ready.set()
                logger.info(f"🚀 [{self.slug}] Retrieval Engine v9.3.0 is READY (Async Init).")
        except Exception as e:
            logger.error(f"[{self.slug}] Async Init Failed: {e}", exc_info=True)

    def _load_enrichment_data(self):
        path = Path(self.ctx.base_path) / "enrichment.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.enrichment_map = json.load(f)
                logger.info(f"[{self.slug}] Enrichment map loaded.")
            except Exception as e:
                logger.error(f"[{self.slug}] Failed to load enrichment.json: {e}")

    def _load_assets(self):
        store_path = Path(self.ctx.base_path)
        data_file  = store_path / "normalized_products_final.json"
        index_file = store_path / "faiss.index"

        if data_file.exists():
            try:
                with open(data_file, "r", encoding="utf-8") as f:
                    products_list = json.load(f)
                if isinstance(products_list, list):
                    for item in products_list:
                        p_id = str(item.get("product_id") or item.get("id")).strip()
                        self.metadata[p_id] = item
                
                self.schema_keys = getattr(self.ctx, "schema_keys", [])
                
                categories = {str(p.get("category", "")).lower() for p in self.metadata.values() if p.get("category")}
                for cat in categories:
                    if cat:
                        self.category_patterns[cat] = re.compile(rf"\b{re.escape(cat)}\b", re.IGNORECASE)
                logger.info(f"[{self.slug}] Metadata loaded: {len(self.metadata)} items.")
            except Exception as e:
                logger.error(f"[{self.slug}] Asset load error: {e}")

        if index_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                logger.info(f"[{self.slug}] FAISS index loaded.")
            except Exception as e:
                logger.error(f"[{self.slug}] FAISS read error: {e}")

        id_map_file = store_path / "id_map.json"
        if id_map_file.exists():
            try:
                with open(id_map_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.id_list = [str(x).strip() for x in (data.get("ids", []) if isinstance(data, dict) else data)]
            except Exception as e:
                logger.error(f"[{self.slug}] ID map error: {e}")
                self.id_list = list(self.metadata.keys())
        else:
            self.id_list = list(self.metadata.keys())

    def _enrich_query(self, query: str) -> str:
        if not self.enrichment_map: return query
        tokens = set(query.lower().split())
        additions = []
        for key, synonyms in self.enrichment_map.items():
            if key in tokens: additions.extend(synonyms)
        return f"{query} {' '.join(additions)}".strip()

    def _keyword_search(self, query: str, limit: int = 40) -> List[Dict[str, Any]]:
        q_words = [get_stem(w) for w in query.lower().split() if len(w) > 2]
        if not q_words: return []
        hits = []
        for pid, p in self.metadata.items():
            blob = str(p.get("search_blob") or f"{p.get('title','')} {p.get('description','')}").lower()
            score = sum(1 for w in q_words if w in blob)
            if score > 0: hits.append({"id": pid, "keyword_score": score})
        hits.sort(key=lambda x: x["keyword_score"], reverse=True)
        return hits[:limit]

    def _rrf_merge(self, vector_hits: List[Dict], keyword_hits: List[Dict], k: int = 60) -> List[str]:
        scores = {}
        for rank, hit in enumerate(vector_hits):
            pid = hit["id"]
            scores[pid] = scores.get(pid, 0) + 1.0 / (k + rank)
        for rank, hit in enumerate(keyword_hits):
            pid = hit["id"]
            scores[pid] = scores.get(pid, 0) + 1.0 / (k + rank)
        return [pid for pid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:80]]

    def _check_negative_intent(self, q_emb: np.ndarray) -> Optional[str]:
        if not self.negative_patterns: return None
        q_norm = q_emb.flatten()
        for patch in self.negative_patterns:
            t_vec = patch.get("vector")
            if not t_vec: continue
            similarity = np.dot(q_norm, np.array(t_vec).astype('float32').flatten())
            if similarity > self.troll_threshold_strict: return "STRICT_REJECT"
            if similarity > self.troll_threshold_soft: return "GRAY_ZONE"
        return None

    def _apply_invisible_filters(self, product: Dict[str, Any], properties: Dict[str, Any]) -> bool:
        if not properties: return True
        p_attrs = product.get('attributes', {})
        if not isinstance(p_attrs, dict): p_attrs = {}
        for key, val in properties.items():
            if val is None or str(val).lower() in ["none", "any", "null"]: continue
            k_lower, v_lower = str(key).lower(), str(val).lower()
            if self.schema_keys and any(sk.lower() == k_lower for sk in self.schema_keys):
                attr_val = str(p_attrs.get(key, product.get(k_lower, ""))).lower()
                if attr_val and attr_val != "none" and v_lower not in attr_val: return False
        return True

    def _semantic_rerank(self, products: List[Dict[str, Any]], target: str, detected_cat: str, target_vec: Optional[np.ndarray] = None, query_words: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        reranked = []
        t_lower, c_lower = target.lower() if target else "", detected_cat.lower() if detected_cat else ""
        query_words = query_words or []
        cat_map = getattr(self.ctx, "category_map", {})
        valid_cats = [c_lower]
        if c_lower in cat_map:
            m = cat_map[c_lower]
            valid_cats.extend([x.lower() for x in m] if isinstance(m, list) else [str(m).lower()])

        for p in products:
            prod = p["product"]
            title, brand, p_cat = str(prod.get("title") or prod.get("name", "")).lower(), str(prod.get("brand", "")).lower(), str(prod.get("category", "")).lower()
            vec_sim = float(p.get("score", 0))
            if target_vec is not None and t_lower and t_lower not in str(prod.get("search_blob") or title).lower(): vec_sim *= 0.7
            
            t_bonus = 0.4 if t_lower and (t_lower in title or t_lower in brand) else 0.0
            c_bonus = 0.3 if any(vc == p_cat or vc in p_cat for vc in valid_cats) else 0.0
            b_mult = 1.2 if brand in self.top_brands else 1.0
            d_bonus = min(sum(1 for w in query_words if len(w) > 2 and (w in title or get_stem(w) in title)) * 0.15, 0.45) if query_words else 0.0

            p["final_score"] = round(((vec_sim * 0.2) + t_bonus + c_bonus + d_bonus) * b_mult, 4)
            reranked.append(p)
        return sorted(reranked, key=lambda x: x["final_score"], reverse=True)

    async def search(self, query: str, entities: Dict[str, Any] = None, top_k: int = 5) -> Dict:
        """Production Hybrid Search Pipeline v9.3.0 (Non-blocking)."""
        if not self.index or not self.id_list:
            return {"status": "NO_ASSETS", "products": [], "all_products": []}

        loop = asyncio.get_running_loop()
        try:
            ent = entities or {}
            target = str(ent.get("resolved_product") or ent.get("target") or "").lower()
            props = ent.get("properties", {})
            enriched_q = self._enrich_query(query)
            det_cat = props.get("category") or ent.get("category") or self._detect_category(query)
            
            if ent.get("action") == "TROLL": return {"status": "SEMANTIC_REJECT", "products": [], "all_products": []}

            # Non-blocking encoding
            search_text = target if len(target) > 2 else enriched_q
            q_emb = await loop.run_in_executor(self.executor, lambda: self.model.encode(f"query: {search_text}", normalize_embeddings=True).astype('float32'))

            if self._check_negative_intent(q_emb) == "STRICT_REJECT":
                return {"status": "SEMANTIC_REJECT", "products": [], "all_products": []}

            # Non-blocking Vector Search
            scores, idxs = await loop.run_in_executor(self.executor, lambda: self.index.search(np.expand_dims(q_emb, axis=0), 60))
            
            vector_hits, v_map = [], {}
            for score, idx in zip(scores[0], idxs[0]):
                if idx != -1 and idx < len(self.id_list):
                    pid = self.id_list[idx]
                    vector_hits.append({"id": pid, "score": float(score)})
                    v_map[pid] = float(score)

            kw_hits = self._keyword_search(enriched_q, limit=40)
            merged_ids = self._rrf_merge(vector_hits, kw_hits)

            t_vec = await loop.run_in_executor(self.executor, lambda: self.model.encode(f"query: {target}", normalize_embeddings=True)) if target else None
            q_words = [w.lower() for w in query.split() if len(w) > 2 and w.lower() not in {"я","мені","хочу","треба","купити"}]
            price_limit = props.get("price_limit") or ent.get("price_limit")
            
            raw_hits, after_price = [], 0
            cat_stem = get_stem(det_cat) if det_cat else None

            for pid in merged_ids:
                p = self.metadata.get(pid)
                if not p: continue
                if cat_stem and cat_stem not in str(p.get("title","")).lower() and cat_stem not in str(p.get("category","")).lower(): continue
                if str(p.get("availability", "instock")).lower() in ["outofstock", "немає в наявності", "нет в наличии"]: continue
                if price_limit and float(p.get("price", 0)) > float(price_limit): continue
                
                after_price += 1
                if self._apply_invisible_filters(p, props):
                    raw_hits.append({"product": p, "score": v_map.get(pid, 0.1), "final_score": 0.0, "id": pid})

            reranked = self._semantic_rerank(raw_hits, target, det_cat or "", t_vec, q_words)
            
            # Интеграция с intel_filter для глубокой фильтрации
            intel_filtered = entity_filter([r["product"] for r in reranked], ent, intent_mapping=self.intent_mapping, category_map=getattr(self.ctx, "category_map", {}))

            intel_filtered_ids = {str(p.get("product_id") or p.get("id")) for p in intel_filtered}
            final_products = [r for r in reranked if str(r["product"].get("product_id") or r["product"].get("id")) in intel_filtered_ids]

            top_results = final_products[:top_k]
           
            log_retrieval(
                slug=self.slug, 
                query_preview=query[:50], 
                faiss_candidates=len(vector_hits), 
                after_entity_filter=len(intel_filtered),  # Добавлен этот обязательный аргумент
                after_price_filter=after_price, 
                final_count=len(top_results), 
                detected_category=det_cat or "none"
            )

            return {"status": "SUCCESS" if top_results else "NOT_FOUND", "products": top_results, "all_products": reranked[:20], "detected_category": det_cat, "is_empty": not top_results}
        except Exception as e:
            logger.error(f"[{self.slug}] Search Pipeline Crash: {e}", exc_info=True)
            return {"status": "ERROR", "products": [], "all_products": [], "is_empty": True}

    def _detect_category(self, query: str) -> Optional[str]:
        for cat, pattern in self.category_patterns.items():
            if pattern.search(query): return cat
        return None

    def _load_negative_patches(self):
        p = Path(self.ctx.base_path) / "fsm_soft_patch.json"
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    self.negative_patterns = d.get("fsm_errors", []) or d.get("troll_patterns", [])
            except: pass

    def _load_store_profile(self):
        if hasattr(self.ctx, 'profile') and self.ctx.profile:
            p = self.ctx.profile
            self.top_brands = [b.lower() for b in p.get("brand_matrix", {}).get("top_brands", [])]
            self.intent_mapping = p.get("intent_mapping", {"brand": ["brand"], "category": ["category"]})

    def close(self):
        self.executor.shutdown(wait=False)
        self.index = None
        self.id_list.clear()
        self.metadata.clear()
        logger.info(f"🛑 [{self.slug}] Retrieval Engine v9.3.0 closed.")