# engine/objection_handler.py
import json
import os
from typing import List, Dict, Optional

class ObjectionHandler:
    """
    Обрабатывает возражения пользователя без участия LLM.
    Работает по заранее подготовленным примерам из objection_examples.json.
    """

    def __init__(self):
        self.objection_path = "/root/ukrsell_project_v3/stores/lucky_dog/objection_examples.json"
        self.examples = self._load_examples()

    def _load_examples(self) -> List[Dict]:
        """
        Загружает примеры возражений из JSON.
        Возвращает пустой список при ошибке.
        """
        if os.path.exists(self.objection_path):
            try:
                with open(self.objection_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def handle(self, query: str, hits: List[Dict]) -> Optional[str]:
        """
        Возвращает текст ответа на возражение, если оно обнаружено.
        Если возражение не найдено — возвращает None.
        """
        if not query:
            return None

        q_lower = query.lower()

        # Определяем категорию возражения
        category = None

        if any(word in q_lower for word in ["дорого", "цін", "дешевше"]):
            category = "price"

        elif any(word in q_lower for word in ["доставка", "відправка"]):
            category = "delivery"

        elif any(word in q_lower for word in ["довіря", "гарант", "надійн"]):
            category = "trust"

        if not category:
            return None

        # Ищем подходящий ответ
        for example in self.examples:
            if example.get("category") == category:
                return example.get("response")

        return None
