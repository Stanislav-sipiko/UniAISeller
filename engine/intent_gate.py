# engine/intent_gate.py
import json
from openai import OpenAI
from config import CEREBRAS_API_KEY
from engine.auto_fixer import AutoFixer

# Настройка под быструю модель
MODEL_FSM = "llama3.1-8b"
client = OpenAI(base_url="https://api.cerebras.ai/v1", api_key=CEREBRAS_API_KEY)
auto_fixer = AutoFixer()

class IntentGate:
    def detect_intent(self, user_query: str) -> dict:
        # Получаем динамические примеры троллинга/ошибок
        neg_ex = auto_fixer.get_negative_examples()
        
        system_prompt = f"""
        Ви — класифікатор інтентів для магазину Lucky Dog. 
        Поверніть ТІЛЬКИ JSON: {{"intent": "...", "query": "..."}}
        
        КАТЕГОРІЇ:
        1. "product_query" - пошук товарів, ціни, наявність.
        2. "objection" - скарги на ціну, доставку або сумніви.
        3. "offtopic" - тролінг, політика, програмування, особисті питання.

        {neg_ex}
        """
        try:
            r = client.chat.completions.create(
                model=MODEL_FSM,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            return json.loads(r.choices[0].message.content)
        except Exception:
            return {"intent": "product_query", "query": user_query}