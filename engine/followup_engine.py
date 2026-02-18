import re

class FollowUpEngine:
    def __init__(self):
        pass

    def process(self, query, last_products, retrieval):
        """
        Analyzes if the query is a follow-up to previous products.
        Returns raw hits if found, else None.
        """
        # Simple heuristic: if query is very short and we have last products
        if last_products and len(query.split()) <= 3:
            # Check for pointers like "this one", "more", "price of that"
            pointers = ["цена", "ціна", "ще", "еще", "купити", "купить", "фото"]
            if any(p in query.lower() for p in pointers):
                return last_products
        return None

    def parse_price(self, price_str):
        """Extracts numeric value from price string."""
        if not price_str:
            return 0
        try:
            # Remove all non-numeric characters except dots
            nums = re.findall(r'\d+', str(price_str))
            return int(nums[0]) if nums else 0
        except Exception:
            return 0