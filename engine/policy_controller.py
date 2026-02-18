# engine/policy_controller.py
import json
import os

class PolicyController:
    def __init__(self):
        # Путь к политике в рамках проекта v3
        self.policy_path = "/root/ukrsell_project_v3/engine/current_policy.json"
        self.default_policy = {
            "thresholds": {
                "high_confidence": 0.3,
                "medium_confidence": 0.6
            },
            "price_segments": {
                "budget": 300,
                "premium": 800
            }
        }
        self.config = self._load_config()

    def _load_config(self):
        if os.path.exists(self.policy_path):
            with open(self.policy_path, "r") as f:
                return json.load(f)
        return self.default_policy

    def get_cta_type(self, distance: float, price: float = 500.0) -> str:
        """
        Определяет стратегию на основе точности поиска и цены товара.
        """
        thresholds = self.config["thresholds"]
        segments = self.config["price_segments"]

        # 1. Если поиск очень неточный
        if distance > thresholds["medium_confidence"]:
            return "SOFT_SUGGESTION" # "Уточните, что именно вы ищете?"

        # 2. Если товар дорогой (Премиум)
        if price >= segments["premium"]:
            return "CONSULTATIVE_SALE" # "Помочь подобрать размер для вашего дога?"

        # 3. Если товар бюджетный (Быстрая покупка)
        if price <= segments["budget"]:
            return "URGENT_SALE" # "Популярный товар, забирайте сейчас!"

        # 4. Средний сегмент
        return "DIRECT_SALE" # "В наличии, готовы отправить."