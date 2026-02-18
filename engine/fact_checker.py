# engine/fact_checker.py
from llm_selector import LLMSelector

class FactChecker:
    """
    FactChecker формирует финальный HTML-ответ, совместимый с Telegram.
    Разрешены только теги: <b>, <i>, <u>, <s>, <a>, <br>, <code>, <pre>, <blockquote>.
    """

    def __init__(self, brand_prompt: str = None):
        self.selector = LLMSelector()

        self.client, self.model = self.selector.get_heavy()
        if self.client is None or self.model is None:
            self.client, self.model = self.selector.get_fast()

        if self.client is None or self.model is None:
            raise RuntimeError("FactChecker: No available LLM models found.")

        self.brand_prompt = brand_prompt if brand_prompt else (
            "Ти — ввічливий та конкретний консультант магазину зоотоварів Lucky Dog. "
            "Відповідай коротко, по суті, у форматі HTML, який підтримує Telegram."
        )

    def _format_attributes(self, attrs):
        if not attrs or not isinstance(attrs, list):
            return "Не вказано"
        formatted = []
        for a in attrs[:4]:
            key = a.get("key", "")
            value = a.get("value", "")
            if key or value:
                formatted.append(f"{key}: {value}")
        return ", ".join(formatted) if formatted else "Не вказано"

    def _build_final_prompt(self, query, results_map, cta_type, intent, prompt_override=None):
        """
        Формує системний промпт для LLM (строго HTML, без списків).
        """
        context_parts = []
        all_found_hits = []

        for search_term, hits in results_map.items():
            if hits:
                all_found_hits.extend(hits)
                product_info = ""
                for h in hits[:3]:
                    title = h.get("title", "")
                    price = h.get("price_current", "")
                    attrs = self._format_attributes(h.get("attributes"))
                    desc = h.get("description_clean", "")[:200]
                    url = h.get("url", "")

                    product_info += (
                        f"Назва: {title}\n"
                        f"Ціна: {price} UAH\n"
                        f"Характеристики: {attrs}\n"
                        f"Опис: {desc}\n"
                        f"Посилання: {url}\n\n"
                    )

                context_parts.append(
                    f"ЗНАЙДЕНО ЗА ЗАПИТОМ '{search_term}':\n{product_info}"
                )
            else:
                context_parts.append(f"ЗА ЗАПИТОМ '{search_term}': НІЧОГО НЕ ЗНАЙДЕНО")

        base_instruction = prompt_override if prompt_override else self.brand_prompt

        prompt = (
            f"{base_instruction}\n\n"
            "ФОРМАТ ВІДПОВІДІ (ОБОВʼЯЗКОВО):\n"
            "- Використовуй ТІЛЬКИ теги: <b>, <i>, <u>, <s>, <a>, <br>, <code>, <pre>, <blockquote>.\n"
            "- НЕ використовуй <ul>, <li>, <ol>, <p>, <div>, <span>, <table>, <img>.\n"
            "- НЕ використовуй Markdown.\n"
            "- НЕ використовуй списки. Замість цього роби нумерацію вручну: '1.', '2.', '3.'\n"
            "- Кожен товар подавай у форматі:\n"
            "  <b>1. Назва товару</b><br>\n"
            "  Ціна: ... грн<br>\n"
            "  Характеристики: ...<br>\n"
            "  <a href=\"URL\">Переглянути товар</a><br><br>\n\n"
            "КОНТЕКСТ ТОВАРІВ:\n"
            f"{chr(10).join(context_parts)}\n\n"
            f"CTA: {cta_type}\n"
            f"INTENT: {intent}\n"
        )

        first = all_found_hits[0] if all_found_hits else {"price_current": 0, "title": ""}
        return prompt, str(first.get("price_current")), first.get("title")

    def generate_answer(self, query: str, results_map: dict, cta_type: str,
                        intent: str = "product_query", prompt_override: str = None) -> str:

        system_instruction, real_price, real_title = self._build_final_prompt(
            query, results_map, cta_type, intent, prompt_override
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": query}
                ],
                temperature=0.1
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty LLM response")

            # Safety: remove forbidden tags if LLM inserted them
            forbidden = ["<ul", "<li", "<ol", "<p", "<div", "<span", "<table", "<img"]
            for tag in forbidden:
                if tag in content.lower():
                    content = content.replace(tag, "")

            return content

        except Exception:
            return (
                "<b>Сталася помилка при формуванні відповіді.</b><br>"
                "Спробуйте, будь ласка, ще раз."
            )
