# core/smart_ranking.py (PATCHED)

class SmartRanking:
    def _get_p(self, item):
        """Универсальное извлечение данных товара"""
        return item.get("product") or item.get("data") or item

    def rank(self, products, entities, memory):
        if not products:
            return []

        ranked = list(products)
        # --- EXCLUDE LAST ---
        if entities.get("exclude_last"):
            last = memory.get("last_products", [])
            last_ids = {str(p.get("id")) for p in last}
            ranked = [p for p in ranked if str(self._get_p(p).get("id") or p.get("id")) not in last_ids]

        # --- PRICE MODIFIER ---
        price_mod = entities.get("price_modifier")
        if price_mod == "down":
            ranked.sort(key=lambda x: float(str(self._get_p(x).get("price", 999999)).replace(',', '.').split()[0] or 999999))
        elif price_mod == "up":
            ranked.sort(key=lambda x: float(str(self._get_p(x).get("price", 0)).replace(',', '.').split()[0] or 0), reverse=True)

        # --- AVAILABILITY BOOST ---
        ranked.sort(
            key=lambda x: (
                str(self._get_p(x).get("availability") or self._get_p(x).get("in_stock")).lower() in ['instock', 'true', 'в наявності'],
                -(float(self._get_p(x).get("rating") or 0))
            ),
            reverse=True
        )
        return ranked[:6]