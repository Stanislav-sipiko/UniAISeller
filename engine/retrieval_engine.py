# engine/retrieval_engine.py
import meilisearch
import json
import os
import re

def clean_id(pid):
    """
    Global utility to sanitize product IDs for Meilisearch compatibility.
    Replaces any character not in [a-zA-Z0-9\-_] with a hyphen.
    """
    return re.sub(r'[^a-zA-Z0-9\-_]', '-', str(pid))

class RetrievalEngine:
    def __init__(self):
        self.client = meilisearch.Client('http://127.0.0.1:7700')
        self.index = self.client.index('products')
        # Path to detailed data
        self.normalized_path = '/root/ukrsell_project_v3/stores/lucky_dog/products_normalized.json'
        self.normalized_data = self._load_normalized()

    def _load_normalized(self):
        if os.path.exists(self.normalized_path):
            with open(self.normalized_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Use the global clean_id function
                return {clean_id(p['product_id']): p for p in data}
        return {}

    def query(self, user_query: str, limit: int = 5):
        try:
            # Flow Search: text-only search via Meilisearch
            search_res = self.index.search(user_query, {'limit': limit})
            hits = search_res.get('hits', [])
            
            results = []
            distances = [] 

            for hit in hits:
                # Use the global clean_id function
                p_id = clean_id(hit.get('product_id'))
                if p_id in self.normalized_data:
                    results.append(self.normalized_data[p_id])
                    # Fixed distance for PolicyController (high confidence)
                    distances.append(0.1) 
            return results, distances
        except Exception:
            return [], []