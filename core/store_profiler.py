# /root/ukrsell_v4/core/store_profiler.py v7.2.0
import json
import os
import hashlib
import logging
import time
import re
import asyncio
from collections import Counter
from typing import Dict, Any, Optional, List


logger = logging.getLogger("UkrSell_Profiler")

class StoreProfiler:
    """
    StoreProfiler Layer v7.2.0.
    - Optimized for normalized_products_final.json.
    - Flat Attribute Discovery: Extracts schema keys from attributes object.
    - Zero Omission: Full script.
    - Strict Obedience: No id_map.json dependency.
    """

    def __init__(self, store_path: str, selector: Any = None):
        self.store_path = store_path
        self.data_path = os.path.join(store_path, "normalized_products_final.json")
        self.config_path = os.path.join(store_path, "config.json")
        self.profile_path = os.path.join(store_path, "store_profile.json")
        self.profile: Optional[Dict[str, Any]] = None
        # Shared selector от ядра. Создаём свой только если не передан —
        # например при запуске скрипта generate_metadata вне платформы.
        self.selector = selector

    async def load_or_build(self) -> Dict[str, Any]:
        """Entry point: builds or loads the profile."""
        if not os.path.exists(self.data_path):
            logger.error(f"Cannot build profile: {self.data_path} not found.")
            return self._get_empty_profile()

        current_hash = self._compute_hash(self.data_path)
        
        if self._needs_rebuild(current_hash):
            logger.info(f"[{os.path.basename(self.store_path)}] Rebuilding store profile from normalized data...")
            if self.selector is None:
                logger.error(f"[{os.path.basename(self.store_path)}] Cannot rebuild profile: selector not provided.")
                return self._get_empty_profile()
            await self.selector.ensure_ready()
            self.profile = await self._build_full_profile_async(current_hash)
            self._save_profile(self.profile)
        else:
            full_data = self._load_full_data_from_disk()
            self.profile = full_data.get("profile", self._get_empty_profile())
            logger.info(f"[{os.path.basename(self.store_path)}] Profile loaded from cache.")
            
        return self.profile

    async def _build_full_profile_async(self, source_hash: str) -> Dict[str, Any]:
        """Выполняет стат-анализ и генерирует приветствие через LLM."""
        stats = self._perform_statistical_analysis(source_hash)
        
        # Загружаем старые данные, чтобы сохранить приветствие, если оно уже было отредактировано вручную
        old_full_data = self._load_full_data_from_disk()
        existing_welcome = old_full_data.get("profile", {}).get("ai_welcome_message")
        
        if existing_welcome and len(existing_welcome) > 10:
            stats["ai_welcome_message"] = existing_welcome
        else:
            logger.info(f"[{os.path.basename(self.store_path)}] Generating new AI Persona welcome message...")
            generated_text = await self._generate_ai_persona(stats)
            stats["ai_welcome_message"] = generated_text
            
        return stats

    async def _generate_ai_persona(self, stats: Dict[str, Any]) -> str:
        """Генерирует описание личности бота на основе статистики товаров."""
        try:
            real_name = "Наш магазин"
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    real_name = config.get("real_name", real_name)

            categories = stats.get("expertise_fields", [])
            cat_context = ", ".join(categories[:3]) if categories else "товари"
            schema_keys = ", ".join(stats.get("schema_keys", [])[:5])
            
            prompt = (
                f"Напиши привітання для Telegram-бота магазину '{real_name}'. "
                f"Ми спеціалізуємось на: {cat_context}. "
                f"Ти — професійний консультант. Ти можеш шукати товари за такими параметрами як {schema_keys}. "
                f"Скажи, що можеш підібрати ідеальний варіант за характеристиками. "
                f"Стиль: експертний, привітний. Мова: Ukrainian. 3 речення."
            )

            client, model = await self.selector.get_heavy()
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM Persona generation failed: {e}")
            return "Вітаємо! Я ваш персональний консультант. Чим можу допомогти?"

    def _perform_statistical_analysis(self, source_hash: str) -> Dict[str, Any]:
        """Анализирует нормализованные данные для выявления структуры ассортимента."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                items = json.load(f)

            if not items or not isinstance(items, list): 
                return self._get_empty_profile()

            categories, brands, prices = [], [], []
            attribute_keys = []
            
            for item in items:
                cat = str(item.get("category", "")).lower().strip()
                p_val = self._parse_price(item.get("price"))

                if cat: categories.append(cat)
                if p_val is not None: prices.append(p_val)

                # Бренд: normalized хранит в attributes["brand"] (dict),
                # либо fallback на "Виробник" в attributes (list для сырых данных)
                attrs = item.get("attributes", {})
                brand = ""
                if isinstance(attrs, dict):
                    brand = str(attrs.get("brand", "") or "").strip()
                    attribute_keys.extend(attrs.keys())
                elif isinstance(attrs, list):
                    for a in attrs:
                        if isinstance(a, dict):
                            if a.get("key") in ("Виробник", "brand", "Brand"):
                                brand = str(a.get("value", "")).strip()
                            attribute_keys.append(a.get("key", ""))

                # Fallback: парсим бренд из search_blob (первый токен часто бренд)
                if not brand:
                    blob = str(item.get("search_blob") or "")
                    # Бренды в blob обычно capitalised отдельным словом
                    import re as _re
                    m = _re.search(r'\b([A-Z][a-zA-Z]{2,})\b', blob)
                    if m:
                        brand = m.group(1)

                if brand:
                    brands.append(brand.lower())

            total_sku = len(items)
            cat_counts = Counter(categories)
            brand_counts = Counter(brands)
            attr_counts = Counter(attribute_keys)
            prices.sort()

            # Ключи, встречающиеся более чем в 5% товаров
            significant_keys = [k for k, v in attr_counts.items() if v > (total_sku * 0.05)]

            main_cat_tuple = cat_counts.most_common(1)[0] if cat_counts else ("none", 0)
            main_share = main_cat_tuple[1] / total_sku if total_sku > 0 else 0

            return {
                "source_hash": source_hash,
                "total_sku": total_sku,
                "store_type": "mono_vertical" if main_share > 0.8 else "multi_category",
                "main_category": main_cat_tuple[0],
                "main_category_share": round(main_share, 2),
                "category_distribution": {k: round(v/total_sku, 3) for k, v in cat_counts.items()},
                "brand_matrix": {
                    "total_brands": len(brand_counts),
                    "top_brands": [b[0].capitalize() for b in brand_counts.most_common(10)]
                },
                "price_analytics": {
                    "min": prices[0] if prices else 0,
                    "max": prices[-1] if prices else 0,
                    "avg": round(sum(prices) / len(prices), 2) if prices else 0
                },
                "expertise_fields": sorted(list(cat_counts.keys())),
                "schema_keys": sorted(significant_keys)
            }
        except Exception as e:
            logger.error(f"Stat analysis error in StoreProfiler: {e}")
            return self._get_empty_profile()

    def _compute_hash(self, file_path: str) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def _needs_rebuild(self, current_hash: str) -> bool:
        if not os.path.exists(self.profile_path): return True
        try:
            with open(self.profile_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("hash") != current_hash
        except: return True

    def _load_full_data_from_disk(self) -> Dict[str, Any]:
        if not os.path.exists(self.profile_path): return {}
        try:
            with open(self.profile_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except: return {}

    def _save_profile(self, profile: Dict[str, Any]):
        output = {
            "hash": profile.get("source_hash"), 
            "generated_at": int(time.time()), 
            "profile": profile
        }
        with open(self.profile_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    def _parse_price(self, price: Any) -> Optional[float]:
        if isinstance(price, (int, float)): return float(price)
        if isinstance(price, str):
            clean = re.sub(r'[^\d.,]', '', price.replace('\xa0', ''))
            if ',' in clean and '.' in clean: clean = clean.replace(',', '')
            elif ',' in clean: clean = clean.replace(',', '.')
            try: return float(clean)
            except: return None
        return None

    def _get_empty_profile(self) -> Dict[str, Any]:
        return {
            "total_sku": 0, 
            "expertise_fields": [], 
            "ai_welcome_message": "", 
            "schema_keys": [],
            "brand_matrix": {"total_brands": 0, "top_brands": []},
            "price_analytics": {"min": 0, "max": 0, "avg": 0}
        }