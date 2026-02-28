import os
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging
from core.translator import TextTranslator

logger = logging.getLogger("UkrSell_Retrieval")

class RetrievalEngine:
    """
    Universal SaaS Retrieval Engine (v4.6).
    Handles any product types (electronics, medicine, etc.) by indexing 
    dynamic 'attributes' and auto-detecting categories.
    """
    def __init__(self, ctx, shared_model, threshold: float = 0.5):
        self.ctx = ctx
        self.model = shared_model  # Shared L12-v2 model
        self.threshold = threshold
        self.translator = TextTranslator()
        
        self.products: List[Dict] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self.available_categories: set = set()
        
        self._load_data()
        self._build_index()

    def _load_data(self):
        """Dynamic load with auto-discovery of categories."""
        products_file = Path(self.ctx.path) / "products.json"
        if not products_file.exists():
            return

        with open(products_file, "r", encoding="utf-8") as f:
            self.products = json.load(f)
        
        # Автоматический сбор всех категорий, присутствующих в базе
        self.available_categories = {
            p.get("category", "").lower() for p in self.products if p.get("category")
        }
        logger.info(f"[{self.ctx.slug}] Loaded {len(self.products)} products. Categories: {self.available_categories}")

    def _text_for_embedding(self, p: Dict) -> str:
        """
        Universal indexing: combines name, category and all dynamic attributes.
        """
        parts = [str(p.get("name", "")), str(p.get("category", ""))]
        
        # Добавляем все значения из гибких атрибутов в индекс
        attributes = p.get("attributes", {})
        if isinstance(attributes, dict):
            parts.extend([str(v) for v in attributes.values()])
        
        return " ".join(parts).lower()

    def _detect_category(self, query: str) -> Optional[str]:
        """Detects if query mentions any of the available categories in this specific store."""
        query_l = query.lower()
        for cat in self.available_categories:
            if cat in query_l:
                return cat
        return None

    def _build_index(self):
        """Lazy FAISS build with L12-v2."""
        if not self.products:
            return
            
        texts = [self._text_for_embedding(p) for p in self.products]
        # Используем Singleton модель из Kernel
        embeddings = np.array(self.model.encode(texts, show_progress_bar=False), dtype="float32")
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings) # Для косинусного сходства
        self.index.add(embeddings)

    async def search(self, query: str) -> Dict:
        """
        Unified search flow:
        1. Translate -> 2. Detect Category -> 3. Vector Search -> 4. Meta Filter
        """
        # 1. Перевод (для мультиязычного поиска)
        translated_query = self.translator.translate(query)
        
        # 2. Определение категории (Safe Gate)
        detected_cat = self._detect_category(translated_query)
        
        # 3. Поиск вектора
        if not self.index:
            return {"status": "NO_RESULTS", "products": []}

        q_emb = np.array(self.model.encode([translated_query]), dtype="float32")
        faiss.normalize_L2(q_emb)
        scores, idxs = self.index.search(q_emb, 5)

        # 4. Фильтрация результатов
        valid_results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1 or score < self.threshold:
                continue
                
            product = self.products[idx]
            prod_cat = product.get("category", "").lower()

            # Жесткий фильтр: если категория в запросе явно определена, 
            # но товар из другой категории — пропускаем.
            if detected_cat and prod_cat != detected_cat:
                continue
                
            valid_results.append({
                "product": product,
                "score": float(score)
            })

        # 5. Обработка ABSENT_CATEGORY
        # Если в запросе была категория, которой нет в этом магазине
        # (Например, запрос 'аспирин' в магазине электроники)
        # Мы можем это понять, если в запросе есть слово, похожее на категорию, 
        # но detected_cat остался пустым. (Здесь можно добавить LLM-валидатор)

        status = "SUCCESS" if valid_results else "NO_RESULTS"
        return {
            "status": status,
            "products": valid_results,
            "detected_category": detected_cat,
            "available_categories": list(self.available_categories)
        }