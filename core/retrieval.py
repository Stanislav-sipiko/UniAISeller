import os
import json
import faiss
import numpy as np
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from core.logger import logger, log_event, log_retrieval
from core.intelligence import entity_filter, get_stem

MIN_CONFIDENCE_THRESHOLD = 0.55
STRICT_MATCH_BONUS = 0.15

class RetrievalEngine:
    """
    RetrievalEngine v7.8.6 — Vector Search & Metadata Normalization.
    
    Changelog v7.8.6:
        - VALIDATION: Added strict normalization of self.metadata to dict type.
        - COMPATIBILITY: Handles both JSON-list and JSON-object metadata sources.
        - CI/CD: Prevents RuntimeErrors during metadata indexing.
        - FIX: Updated log_retrieval signature to include after_price_filter.
    """
    def __init__(self, ctx: Any, shared_model: Any = None):
        self.ctx = ctx
        self.slug = getattr(ctx, 'slug', 'unknown')
        self.model = shared_model
        
        self.index = None
        self.id_list = []
        self.metadata = {}
        self.category_patterns = {}
        self.negative_patterns = []
        
        self.top_brands = []
        self.intent_mapping = {}
        self.schema_keys = []
        self.synonyms_map = {} 

        self._load_assets()
        self._load_store_profile()
        self._load_synonyms()
        logger.info(f"🔍 [{self.slug}] RetrievalEngine v7.8.6 initialized. Model: {id(self.model)}")

    def _load_assets(self):
        """Loads FAISS index and product metadata with strict type normalization."""
        try:
            base_path = Path(self.ctx.base_path)
            
            index_variants = ["faiss.index", "faiss_index.bin", "vector.index"]
            index_file = next((base_path / v for v in index_variants if (base_path / v).exists()), None)
            
            meta_variants = ["normalized_products_final.json", "products_final.json", "metadata.json"]
            meta_file = next((base_path / v for v in meta_variants if (base_path / v).exists()), None)

            id_map_path = getattr(self.ctx, 'id_map_path', None)
            if not id_map_path:
                id_map_variants = ["id_map.json", "idmap.json"]
                id_map_file = next((base_path / v for v in id_map_variants if (base_path / v).exists()), None)
                if id_map_file: id_map_path = str(id_map_file)

            if not index_file or not meta_file:
                logger.error(f"❌ [{self.slug}] Missing assets. Index: {index_file}, Meta: {meta_file}")
                return

            self.index = faiss.read_index(str(index_file))
            
            with open(meta_file, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
                
                if isinstance(meta_data, list):
                    self.metadata = self._normalize_to_dict(meta_data)
                    self.schema_keys = []
                elif isinstance(meta_data, dict):
                    raw_products = meta_data.get("products", meta_data)
                    self.schema_keys = meta_data.get("schema_keys", [])
                    self.metadata = self._normalize_to_dict(raw_products)
                else:
                    logger.error(f"❌ [{self.slug}] Unknown meta_data type: {type(meta_data)}")
                    self.metadata = {}

            if id_map_path and os.path.exists(id_map_path):
                with open(id_map_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.id_list = data.get("ids", data) if isinstance(data, dict) else data
                    logger.debug(f"[{self.slug}] Loaded {len(self.id_list)} IDs mapping.")

            categories = set()
            for p in self.metadata.values():
                cat = p.get("category")
                if cat: categories.add(str(cat))
            
            for cat in categories:
                clean_cat = re.escape(cat.lower())
                self.category_patterns[cat] = re.compile(rf"\b{clean_cat}\b", re.I)

            self._load_negative_patches()
            
        except Exception as e:
            logger.error(f"❌ [{self.slug}] Asset Load Error: {e}", exc_info=True)

    def _normalize_to_dict(self, data: Any) -> Dict[str, Any]:
        """Ensures metadata is always a dictionary keyed by product ID."""
        normalized = {}
        if isinstance(data, list):
            for idx, p in enumerate(data):
                if not isinstance(p, dict): continue
                p_id = str(p.get("product_id") or p.get("id") or f"idx_{idx}")
                normalized[p_id] = p
        elif isinstance(data, dict):
            if "products" in data and isinstance(data["products"], list):
                return self._normalize_to_dict(data["products"])
            normalized = data
        else:
            logger.error(f"[{self.slug}] Unknown metadata format: {type(data)}")
        return normalized

    def _load_synonyms(self):
        syn_path = Path(self.ctx.base_path) / "search_synonyms.json"
        if syn_path.exists():
            try:
                with open(syn_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    raw_syns = data.get("synonyms", {})
                    for main_word, variants in raw_syns.items():
                        for var in variants:
                            self.synonyms_map[var.lower()] = main_word.lower()
            except Exception as e:
                logger.error(f"[{self.slug}] Synonyms error: {e}")

    def _apply_synonyms(self, query: str) -> str:
        if not self.synonyms_map: return query
        normalized = query.lower()
        words = re.findall(r'\w+', normalized)
        for word in words:
            if word in self.synonyms_map:
                normalized = re.sub(rf"\b{word}\b", self.synonyms_map[word], normalized)
        return normalized

    def _load_store_profile(self):
        profile = getattr(self.ctx, 'profile', {})
        if profile:
            brand_data = profile.get("brand_matrix", {})
            if isinstance(brand_data, dict):
                self.top_brands = [b.lower() for b in brand_data.get("top_brands", [])]
            
            self.intent_mapping = profile.get("intent_mapping", {
                "brand": ["brand", "manufacturer"],
                "category": ["category"],
                "animal": ["animal", "pet_type"]
            })

    def _load_negative_patches(self):
        patch_path = Path(self.ctx.base_path) / "fsm_soft_patch.json"
        if patch_path.exists():
            try:
                with open(patch_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.negative_patterns = data.get("fsm_errors", []) or data.get("troll_patterns", [])
            except: pass

    def _semantic_rerank(self, hits: List[Dict], target: str, category: str, query_words: List[str]) -> List[Dict]:
        if not hits: return []
        query_stems = [get_stem(w) for w in query_words]
        
        for hit in hits:
            p = hit["product"]
            score = hit["score"]
            norm_sim = 1.0 / (1.0 + score)
            
            cat_bonus = 0.0
            p_cat = str(p.get("category", "")).lower()
            if category and p_cat == category.lower():
                cat_bonus = 0.20
            
            title_lower = str(p.get("title", p.get("name", ""))).lower()
            title_match_ratio = 0.0
            if query_stems:
                matches = sum(1 for stem in query_stems if stem in title_lower)
                title_match_ratio = (matches / len(query_stems))
            
            final_score = (norm_sim * 0.8) + cat_bonus + (title_match_ratio * 0.5)
            
            p_brand = str(p.get("brand", "")).lower()
            if p_brand and any(b in p_brand or b in title_lower for b in self.top_brands):
                final_score += 0.15

            hit["final_score"] = round(final_score, 4)

        return sorted(hits, key=lambda x: x["final_score"], reverse=True)

    def _detect_category(self, query: str) -> Optional[str]:
        for cat, pattern in self.category_patterns.items():
            if pattern.search(query): return cat
        return None

    async def search(self, query: Union[str, Dict[str, Any]], entities: Dict[str, Any] = None, top_k: int = 5) -> Dict:
        if not self.index or not self.metadata:
            return {"status": "NO_ASSETS", "products": [], "confidence": 0.0, "is_empty": True}

        try:
            if isinstance(query, dict):
                entities_data = query.get("entities", {}) if isinstance(query.get("entities"), dict) else {}
                query_text = str(
                    query.get("query") or 
                    entities_data.get("target") or 
                    entities_data.get("category") or 
                    query
                )
            else:
                query_text = str(query)

            normalized_query = self._apply_synonyms(query_text)
            q_low = normalized_query.lower()
            
            for pattern in self.negative_patterns:
                if str(pattern).lower() in q_low:
                    return {"status": "STRICT_REJECT", "products": [], "confidence": 0.0, "is_empty": True}

            ent_root = entities or {}
            if not ent_root and isinstance(query, dict):
                ent_root = query

            inner_ents = ent_root.get("entities", ent_root)
            if not isinstance(inner_ents, dict): inner_ents = {}
            
            target = str(inner_ents.get("target") or inner_ents.get("resolved_product") or "").lower()
            detected_cat = inner_ents.get("category")
            if not detected_cat: detected_cat = self._detect_category(normalized_query)

            search_text = target if len(target) > 3 else normalized_query
            q_emb = self.model.encode(f"query: {search_text}", normalize_embeddings=True).astype('float32')

            STOP = {"купити", "хочу", "знайти", "ціна", "є", "маєте", "собаки", "кота", "для", "який", "яка"}
            query_words = [w for w in normalized_query.split() if len(w) > 2 and w not in STOP]

            scores, idxs = self.index.search(np.expand_dims(q_emb, axis=0), 50)

            raw_hits = []
            for score, idx in zip(scores[0], idxs[0]):
                if idx == -1: continue
                
                p_id = str(self.id_list[idx]) if (self.id_list and idx < len(self.id_list)) else str(idx)
                product = self.metadata.get(p_id)
                
                if not product: continue

                avail = str(product.get("availability", "")).lower()
                if any(x in avail for x in ["out", "немає", "нет", "відсутній"]):
                    continue
                
                raw_hits.append({"product": product, "score": float(score)})

            reranked = self._semantic_rerank(raw_hits, target, detected_cat, query_words)

            intelligence_filtered = entity_filter(
                reranked,
                {"entities": inner_ents}, 
                intent_mapping=self.intent_mapping,
                category_map=getattr(self.ctx, "category_map", {}),
            )

            top_score = intelligence_filtered[0]["final_score"] if intelligence_filtered else 0.0
            
            if top_score < MIN_CONFIDENCE_THRESHOLD:
                status = "LOW_CONFIDENCE"
                final_results = []
            else:
                status = "SUCCESS" if intelligence_filtered else "NOT_FOUND"
                final_results = [h["product"] for h in intelligence_filtered[:top_k]]

            log_retrieval(
                slug=self.slug, 
                query_preview=query_text[:50],
                faiss_candidates=len(raw_hits),
                after_entity_filter=len(intelligence_filtered),
                after_price_filter=len(intelligence_filtered),
                final_count=len(final_results),
                detected_category=detected_cat or "none"
            )

            return {
                "status": status,
                "products": final_results,
                "confidence": top_score,
                "detected_category": detected_cat,
                "is_empty": len(final_results) == 0
            }

        except Exception as e:
            logger.error(f"[{self.slug}] Critical Search Error: {e}", exc_info=True)
            return {"status": "ERROR", "products": [], "confidence": 0.0, "is_empty": True}

    def close(self):
        self.index = None
        self.metadata = {}
        logger.info(f"[{self.slug}] Retrieval Engine resources released.")