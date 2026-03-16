# -*- coding: utf-8 -*-
# core/recommendation_brain.py v1.1.0

class RecommendationBrain:
    """
    Recommendation Brain v1.1.0
    Выбирает лучший товар (Champion), когда пользователь просит прямой совет.
    
    Changelog v1.1.0:
    - FIX: Синхронизация с Retrieval v9.1.0 (поддержка ключа 'product').
    - FIX: Улучшен парсинг цен для скоринга (поддержка разных форматов валют).
    """

    ADVICE_PATTERNS = [
        "что лучше", "какую лучше", "какую посоветуете", "посоветуй",
        "кращий", "який краще", "порадь", "найкращий", "найзручніший",
        "best", "recommend", "лучшая", "самая удобная"
    ]

    def detect_advice(self, query: str) -> bool:
        """Определяет, нужен ли пользователю конкретный совет."""
        q = query.lower()
        return any(p in q for p in self.ADVICE_PATTERNS)

    def _get_p(self, item: dict) -> dict:
        """Универсальное извлечение данных товара (Data Contract Patch)."""
        if not isinstance(item, dict):
            return {}
        return item.get("product") or item.get("data") or item

    def pick_best(self, products: list) -> list:
        """
        Выбирает один лучший товар на основе скоринга.
        Принимает уже отфильтрованный и ранжированный список.
        """
        if not products:
            return []

        scored = []
        for item in products:
            # ИСПРАВЛЕНО: Универсальное извлечение для решения проблемы "Матрешки"
            p = self._get_p(item)
            
            score = 0
            # 1. Наличие — критический фактор (+5)
            # Расширенная проверка форматов наличия
            avail = str(p.get("availability") or p.get("in_stock") or "").lower()
            if avail in ["in stock", "instock", "true", "в наявності", "есть в наличии", "1"] or p.get("in_stock") is True:
                score += 5
            
            # 2. Рейтинг (+ значение рейтинга)
            try:
                score += float(p.get("rating") or 0)
            except:
                pass
            
            # 3. Популярность или наличие фото (+1)
            if p.get("image_url") or p.get("image"):
                score += 1
                
            # 4. Цена (бонус за средний сегмент)
            price = 0
            try:
                # Улучшенный парсинг цены: убираем пробелы, запятые на точки
                import re
                raw_price = str(p.get("price", "0")).replace(',', '.').replace('\xa0', '').strip()
                # Извлекаем только число (первое совпадение)
                match = re.search(r"[-+]?\d*\.\d+|\d+", raw_price)
                price = float(match.group()) if match else 0
            except: 
                pass
            
            # Сохраняем логику "золотой середины", расширенную под разные категории
            if 300 < price < 5000: 
                score += 2

            scored.append((score, item))

        # Сортируем по весу и берем первого чемпиона
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [scored[0][1]] if scored else []