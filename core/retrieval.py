# /root/ukrsell_v4/core/retrieval.py v7.5.1
"""
Universal SaaS Retrieval Engine v7.5.0

Changelog:
    v7.4.0  Normalized layer, entity_filter интеграция, Intent Mapping
    v7.5.0  [ФИКС для confidence.py v1.7.0]
            - final_score теперь явно передаётся в результирующих dict'ах
              (раньше confidence._extract_sim_score не находил его и падал на FAISS cosine)
            - log_retrieval: after_price_filter теперь считается правильно (отдельный счётчик)
            - _semantic_rerank: добавлен direct_title_bonus — если query words в title
            - search(): возвращает all_products_unfiltered для передачи в ASK_CLARIFICATION
"""

import os
import json
import faiss
import numpy as np
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any
from core.logger import logger, log_event, log_retrieval
from core.intelligence import entity_filter, get_stem


class RetrievalEngine:
    """
    Universal SaaS Retrieval Engine v7.5.0.
    - Optimized for normalized_products_final.json.
    - Flat Attribute Filtering: Direct access to attributes dictionary.
    - Search Blob Awareness: Enhanced text indexing.
    - Intent Mapping: Dynamically links AI entities to store-specific DB keys.
    - final_score передаётся в results для корректной работы confidence.py v1.7.0.
    """

    def __init__(self, ctx: Any, shared_model: Any, shared_translator: Any):
        self.ctx = ctx
        self.slug = getattr(ctx, 'slug', 'unknown')
        self.model = shared_model
        self.translator = shared_translator

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

        self._initialize_assets()

    def _initialize_assets(self):
        """Load all data and set readiness flag."""
        self._load_assets()
        self._load_negative_patches()
        self._load_store_profile()

        if hasattr(self.ctx, 'data_ready'):
            self.ctx.data_ready.set()
            logger.info(f"🚀 [{self.slug}] Retrieval Engine v7.5.0 is READY (Normalized Layer).")

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
                        p_id = str(item.get("product_id")).strip()
                        self.metadata[p_id] = item

                    self.schema_keys = getattr(self.ctx, "schema_keys", [])

                categories = {
                    str(p.get("category", "")).lower()
                    for p in self.metadata.values()
                    if isinstance(p, dict) and p.get("category")
                }
                for cat in categories:
                    if cat:
                        self.category_patterns[cat] = re.compile(
                            rf"\b{re.escape(cat)}\b", re.IGNORECASE
                        )

                logger.info(f"[{self.slug}] Assets loaded: {len(self.metadata)} items.")
            except Exception as e:
                logger.error(f"[{self.slug}] Error loading assets: {e}")

        if index_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                logger.info(f"[{self.slug}] FAISS index loaded: {self.index.ntotal} vectors.")
            except Exception as e:
                logger.error(f"[{self.slug}] FAISS error: {e}")

        id_map_file = store_path / "id_map.json"
        if id_map_file.exists():
            try:
                with open(id_map_file, "r", encoding="utf-8") as f:
                    self.id_list = json.load(f)
                if not isinstance(self.id_list, list):
                    self.id_list = self.id_list.get("ids", [])
                self.id_list = [str(x).strip() for x in self.id_list]
                logger.info(f"[{self.slug}] id_map.json loaded: {len(self.id_list)} positions.")
            except Exception as e:
                logger.error(f"[{self.slug}] id_map.json load error: {e}. Falling back to JSON order.")
                self.id_list = list(self.metadata.keys())
        else:
            logger.warning(
                f"[{self.slug}] id_map.json not found — using JSON order as id_list. "
                f"Run rebuild_faiss_index.py to fix FAISS↔product alignment."
            )
            self.id_list = list(self.metadata.keys())

    def _check_negative_intent(self, q_emb: np.ndarray) -> Optional[str]:
        """Check against negative examples from fsm_soft_patch."""
        if not self.negative_patterns:
            return None
        q_norm = q_emb.flatten()
        for patch in self.negative_patterns:
            troll_vec = patch.get("vector")
            if not troll_vec:
                continue
            t_emb = np.array(troll_vec).astype('float32').flatten()
            similarity = np.dot(q_norm, t_emb)
            if similarity > self.troll_threshold_strict:
                return "STRICT_REJECT"
            if similarity > self.troll_threshold_soft:
                return "GRAY_ZONE"
        return None

    def _apply_invisible_filters(self, product: Dict[str, Any], properties: Dict[str, Any]) -> bool:
        """Deep check product attributes against extracted properties."""
        if not properties:
            return True

        p_attrs = product.get('attributes', {})
        if not isinstance(p_attrs, dict):
            p_attrs = {}

        for key, val in properties.items():
            k_lower = str(key).lower()
            v_lower = str(val).lower()

            if self.schema_keys and any(sk.lower() == k_lower for sk in self.schema_keys):
                attr_val = str(p_attrs.get(key, product.get(k_lower, ""))).lower()
                if attr_val and attr_val != "none" and v_lower not in attr_val:
                    return False
        return True

    def _semantic_rerank(
        self,
        products: List[Dict[str, Any]],
        target: str,
        detected_cat: str,
        target_vec: Optional[np.ndarray] = None,
        query_words: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Reranks FAISS results using semantic importance.

        v7.5.0: добавлен direct_title_bonus — прямое совпадение слов запроса с title.
        final_score теперь всегда присутствует в результирующем dict для confidence.py.
        """
        reranked = []
        target_lower = target.lower() if target else ""
        cat_lower = detected_cat.lower() if detected_cat else ""
        query_words = query_words or []

        for p in products:
            prod = p["product"]
            title  = str(prod.get("title", "")).lower()
            brand  = str(prod.get("brand", "") or "").lower()
            p_cat  = str(prod.get("category", "")).lower()
            vec_sim = float(p.get("score", 0))

            if target_vec is not None:
                blob = str(prod.get("search_blob") or title).lower()
                blob_vec = self.model.encode(f"passage: {blob}", normalize_embeddings=True)
                intent_match = np.dot(target_vec, blob_vec)
                if intent_match < self.semantic_filter_threshold:
                    vec_sim *= 0.6

            target_bonus = 0.4 if target_lower and (target_lower in title or target_lower in brand) else 0.0
            cat_bonus    = 0.3 if cat_lower and (cat_lower == p_cat or cat_lower in p_cat) else 0.0
            brand_mult   = 1.15 if brand in self.top_brands else 1.0

            # v7.5.0: прямое совпадение слов запроса в title (v7.5.1: stem-aware)
            direct_title_bonus = 0.0
            if query_words:
                matched = sum(
                    1 for w in query_words
                    if len(w) > 2 and (w in title or get_stem(w) in title)
                )
                direct_title_bonus = min(matched * 0.1, 0.3)

            final_score = round(
                ((vec_sim * 0.3) + target_bonus + cat_bonus + direct_title_bonus) * brand_mult,
                4
            )
            p["final_score"] = final_score  # ← явная передача для confidence.py v1.7.0
            reranked.append(p)

        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked

    async def search(self, query: str, entities: Dict[str, Any] = None, top_k: int = 5) -> Dict:
        """
        Hybrid Search Flow with Intelligence Pipeline integration.

        v7.5.0: log_retrieval считает after_price_filter отдельно.
                Возвращает all_products для передачи в ASK_CLARIFICATION.
        """
        if not self.index or not self.id_list:
            return {"status": "NO_ASSETS", "products": [], "all_products": []}

        try:
            ent = entities or {}
            target = str(ent.get("resolved_product") or ent.get("target") or "").lower()
            props  = ent.get("properties", {})

            if ent.get("action") == "TROLL":
                return {
                    "status": "SEMANTIC_REJECT",
                    "products": [],
                    "all_products": [],
                    "reason": "action_troll_detected",
                }

            detected_cat = (
                props.get("category") or
                ent.get("category") or
                self._detect_category(query)
            )
            search_text = target if len(target) > 2 else query

            q_emb = self.model.encode(
                f"query: {search_text}", normalize_embeddings=True
            ).astype('float32')

            neg_status = self._check_negative_intent(q_emb)
            if neg_status == "STRICT_REJECT":
                return {
                    "status": "SEMANTIC_REJECT",
                    "products": [],
                    "all_products": [],
                    "reason": "negative_patch_match",
                }

            target_vec = (
                self.model.encode(f"query: {target}", normalize_embeddings=True)
                if target else None
            )

            # Слова запроса для direct_title_bonus
            stopwords = {"я", "мені", "хочу", "треба", "купити", "знайти", "шукаю",
                         "що", "якийсь", "щось", "для", "і", "та", "й", "чи", "або"}
            query_words = [w.lower() for w in query.split()
                           if len(w) > 2 and w.lower() not in stopwords]

            price_limit = props.get("price_limit") or ent.get("price_limit")
            if price_limit and not isinstance(price_limit, (int, float)):
                match = re.search(r'\d+', str(price_limit).replace(" ", ""))
                price_limit = float(match.group(0)) if match else None

            scores, idxs = self.index.search(
                np.expand_dims(q_emb, axis=0), top_k * 50
            )

            raw_hits = []
            after_price_count = 0

            for score, idx in zip(scores[0], idxs[0]):
                if idx == -1 or idx >= len(self.id_list):
                    continue
                product = self.metadata.get(self.id_list[idx])
                if not product:
                    continue

                # Наличие
                stock = str(product.get("availability", "instock")).lower()
                if stock in ["outofstock", "немає в наявності", "нет в наличии"]:
                    continue

                # Ценовой фильтр
                if price_limit:
                    try:
                        p_val = float(product.get("price", 0))
                        if p_val > price_limit:
                            continue
                    except:
                        continue
                after_price_count += 1

                # Невидимые фильтры
                if not self._apply_invisible_filters(product, props):
                    continue

                raw_hits.append({
                    "product":     product,
                    "score":       float(score),
                    "final_score": 0.0,  # будет заполнен в _semantic_rerank
                    "id":          self.id_list[idx],
                })

            # Реранжирование
            reranked = self._semantic_rerank(
                raw_hits, target, detected_cat, target_vec, query_words
            )

            # Entity filter
            intelligence_filtered = entity_filter(
                [r["product"] for r in reranked],
                ent,
                intent_mapping=self.intent_mapping,
                category_map=getattr(self.ctx, "category_map", {}),
            )

            final_products = []
            for prod in intelligence_filtered:
                for r in reranked:
                    if str(r["product"].get("product_id")) == str(prod.get("product_id")):
                        final_products.append(r)
                        break

            top_results = final_products[:top_k]
            final_status = "SUCCESS" if top_results else "NO_RESULTS"

            scores_all = [r.get("score", 0.0) for r in raw_hits]
            log_retrieval(
                slug=self.slug,
                query_preview=query,
                faiss_candidates=len(raw_hits),
                after_price_filter=after_price_count,
                after_entity_filter=len(intelligence_filtered),
                final_count=len(top_results),
                detected_category=detected_cat or "",
                score_min=min(scores_all, default=0.0),
                score_max=max(scores_all, default=0.0),
            )

            return {
                "status":           final_status,
                "products":         top_results,
                "all_products":     reranked[:top_k * 3],  # для ASK_CLARIFICATION
                "detected_category": detected_cat,
                "target_detected":  target,
                "negative_match":   neg_status,
                "applied_filters":  list(props.keys()),
            }

        except Exception as e:
            logger.error(f"[{self.slug}] Search Pipeline Crash: {e}", exc_info=True)
            return {"status": "ERROR", "products": [], "all_products": []}

    def _detect_category(self, query: str) -> Optional[str]:
        for cat, pattern in self.category_patterns.items():
            if pattern.search(query):
                return cat
        return None

    def _load_negative_patches(self):
        patch_path = Path(self.ctx.base_path) / "fsm_soft_patch.json"
        if patch_path.exists():
            try:
                with open(patch_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.negative_patterns = (
                        data.get("fsm_errors", []) or
                        data.get("troll_patterns", [])
                    )
            except:
                pass

    def _load_store_profile(self):
        """Загрузка аналитики брендов и карты соответствия полей (Intent Mapping)."""
        if hasattr(self.ctx, 'profile') and self.ctx.profile:
            profile = self.ctx.profile
            self.top_brands = [
                b.lower()
                for b in profile.get("brand_matrix", {}).get("top_brands", [])
            ]
            self.intent_mapping = profile.get("intent_mapping", {
                "brand":    ["brand"],
                "category": ["category"],
            })
            logger.debug(f"[{self.slug}] Intent Mapping loaded: {list(self.intent_mapping.keys())}")

    def close(self):
        self.index = None
        self.id_list.clear()
        self.metadata.clear()
        logger.info(f"🛑 [{self.slug}] Retrieval Engine closed.")