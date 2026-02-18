# engine/planner_llm.py
import re
from llm_selector import LLMSelector

class PlannerLLM:
    """
    PlannerLLM отвечает за генерацию уточнённых поисковых запросов.
    Он использует тот же LLMSelector, что и main.py, чтобы всегда работать
    с живой, доступной fast-моделью (Cerebras или Groq).
    """

    def __init__(self):
        # Инициализируем общий селектор моделей
        self.selector = LLMSelector()

        # Получаем fast-клиент и fast-модель
        self.client, self.model = self.selector.get_fast()

        # Если fast-модель недоступна — пробуем heavy
        if self.client is None or self.model is None:
            self.client, self.model = self.selector.get_heavy()

        # Если вообще нет моделей — аварийный режим
        if self.client is None or self.model is None:
            raise RuntimeError("PlannerLLM: No available LLM models found.")

    def extract_search_queries(self, user_query: str) -> list:
        """
        Разбивает пользовательский запрос на отдельные поисковые подзапросы.
        Использует fast LLM (или heavy, если fast недоступна).
        Возвращает список строк.
        """

        prompt = (
            "Розбий запит користувача на окремі назви товарів для пошуку в базі.\n"
            "Приклад: \"лежак для собаки та миска\" -> лежак для собаки, миска\n"
            f"Запит: \"{user_query}\"\n"
            "Поверни ТІЛЬКИ список через кому, без пояснень."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a search query optimizer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            content = response.choices[0].message.content

            # Разбиваем по запятым
            parts = [q.strip() for q in content.split(",")]

            # Фильтруем пустые строки
            parts = [p for p in parts if len(p) > 1]

            # Если LLM вернул мусор — fallback
            if not parts:
                return [user_query]

            return parts

        except Exception:
            # Fallback: возвращаем исходный запрос
            return [user_query]
