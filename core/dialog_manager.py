import json
import logging
import os
import asyncio

logger = logging.getLogger("DialogManager")

class DialogManager:
    def __init__(self, ctx, llm_selector):
        self.ctx = ctx
        self.selector = llm_selector
        # Динамический путь к патчу внутри папки конкретного магазина
        self.patch_path = os.path.join(self.ctx.base_path, "fsm_soft_patch.json")

    def _build_dynamic_prompt(self, negative_examples_list: list) -> str:
        negative_examples = ""
        if negative_examples_list:
            negative_examples = "\nНЕГАТИВНЫЕ ПРИМЕРЫ (ТРОЛЛИНГ):\n" + "\n".join(f"- {ex}" for ex in negative_examples_list[:10])

        base_prompt = self.ctx.prompts.get("product_consultant", "Ты помощник.")
        
        return (
            f"{base_prompt}\n\n"
            "ПРАВИЛА ТЕХНИЧЕСКОЙ НОРМАЛИЗАЦИИ:\n"
            "1. ACTION 'TROLL': если запрос — абсурд, дичь или оффтоп.\n"
            "2. ACTION 'SEARCH': если запрос понятен и по делу.\n"
            f"{negative_examples}\n\n"
            "ОТВЕЧАЙ СТРОГО В JSON ФОРМАТЕ: {'action': '...', 'response_text': '...'}"
        )

    def record_troll_pattern(self, user_text: str):
        """Записывает абсурдный запрос для самообучения."""
        try:
            data = {"troll_patterns": [], "fsm_errors": []}
            if os.path.exists(self.patch_path):
                with open(self.patch_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            clean_text = user_text.lower().strip()
            if clean_text not in data.get("troll_patterns", []):
                data.setdefault("troll_patterns", []).append(clean_text)
                data.setdefault("fsm_errors", []).append({"request": user_text, "type": "negative_example"})

                with open(self.patch_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"САМООБУЧЕНИЕ: Паттерн троллинга записан для {self.ctx.slug}")
        except Exception as e:
            logger.error(f"Ошибка записи паттерна: {e}")

    def get_negative_examples(self) -> list:
        if os.path.exists(self.patch_path):
            try:
                with open(self.patch_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("troll_patterns", [])
            except: pass
        return []

    async def analyze_intent(self, user_text: str) -> dict:
        """Решает, нужно ли обслуживать запрос или это троллинг."""
        # Простая проверка буфера (быстрая экономия токенов)
        patterns = self.get_negative_examples()
        t = user_text.lower().strip()
        if any(p in t for p in patterns):
            return {"action": "TROLL", "response_text": "Опять за старое? Давайте лучше по делу. 😎"}

        # Если в буфере нет, идем к LLM (Heavy)
        client, model = self.selector.get_heavy()
        system_prompt = self._build_dynamic_prompt(patterns)

        try:
            # Используем asyncio.to_thread для синхронного вызова клиента
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            
            if result.get("action") == "TROLL":
                self.record_troll_pattern(user_text)
            
            return result
        except Exception as e:
            logger.error(f"DialogManager Error: {e}")
            return {"action": "SEARCH"}