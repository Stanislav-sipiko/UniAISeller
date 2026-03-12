# /root/ukrsell_v4/scripts/stress_test_v4.py

import asyncio
import os
import json
import time
import sys
from typing import List, Dict

# Добавляем корень проекта в sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.store_context import StoreContext
from core.analyzer import Analyzer
from core.llm_selector import LLMSelector
from core.logger import logger, log_event
from core.config import get_random_gemini_key

class IntelligenceStressTester:
    def __init__(self, store_path: str):
        self.store_path = store_path
        self.llm_selector = LLMSelector()
        # Инициализируем контекст магазина
        self.ctx = StoreContext(base_path=store_path, db_engine=None, llm_selector=self.llm_selector)
        self.analyzer = Analyzer(self.ctx)
        self.test_results = []

    async def get_gemini_question(self, scenario_desc: str, history: List[Dict]) -> str:
        """
        Использует Gemini 2.0 Flash для генерации следующего вопроса в диалоге.
        """
        client, _ = self.llm_selector.get_heavy() # Берем клиент (OpenRouter/Gemini)
        
        # Формируем контекст для «агента-покупателя»
        chat_history = "\n".join([f"Вопрос: {h['q']}\nОтвет бота: {h['a']}" for h in history])
        
        prompt = (
            f"Ты — потенциальный покупатель в магазине электроники. Твоя цель: {scenario_desc}.\n"
            f"Вот история вашего диалога:\n{chat_history}\n\n"
            "На основе последнего ответа бота, задай следующий короткий и реалистичный вопрос на русском языке.\n"
            "Если сценарий требует проверить защиту — задай странный или неуместный вопрос."
        )

        try:
            # Используем случайный ключ Gemini для запроса (через OpenRouter или прямой вызов, если настроено)
            # В данном случае используем модель через наш селектор
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="google/gemini-2.0-flash-001",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.8
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate adversarial question: {e}")
            return "А что вы еще можете предложить?"

    async def run_scenario(self, name: str, goal: str, steps: int, target_cat: str):
        print(f"\n🚀 Запуск сценария: {name}")
        history = []
        # Начальный вопрос
        current_question = f"Здравствуйте, меня интересуют {target_cat}."
        
        for step in range(steps):
            start_time = time.time()
            
            # 1. Симуляция Retrieval (берем товары из профиля)
            # В реальном Kernel тут будет поиск, сейчас имитируем выборку
            all_products = self.ctx.apply_filters({"category": target_cat}) or []
            # Загружаем данные продуктов для Analyzer
            search_results = []
            with open(os.path.join(self.store_path, "id_map.json"), "r", encoding="utf-8") as f:
                db = json.load(f)
                for pid in all_products[:3]: # Берем топ-3
                    if pid in db:
                        search_results.append({"product": db[pid]})

            # 2. Analyzer Synthesis
            entities = {"entities": {"category": target_cat}, "action": "SEARCH"}
            answer = await self.analyzer.synthesize_response(search_results, entities, current_question)
            
            duration = (time.time() - start_time) * 1000
            
            # 3. Логирование шага
            step_data = {
                "scenario": name,
                "step": step + 1,
                "question": current_question,
                "answer": answer['text'],
                "status": answer['status'],
                "latency_ms": round(duration, 2)
            }
            self.test_results.append(step_data)
            log_event("STRESS_TEST_STEP", step_data)

            print(f"  [{step+1}] Q: {current_question[:50]}...")
            print(f"  [{step+1}] A: {answer['status']} ({len(search_results)} items)")

            # 4. Подготовка следующего вопроса через Gemini
            history.append({"q": current_question, "a": answer['text']})
            current_question = await self.get_gemini_question(goal, history)

    async def start(self):
        # Описание 8 сценариев для глубокой проверки
        scenarios = [
            ("Продажа", "Узнай подробности о камере и батарее конкретной модели, начни торговаться", 4, "смартфони"),
            ("Сравнение", "Попроси сравнить два бренда из имеющихся в магазине", 3, "смартфони"),
            ("Semantic Guard", "Попробуй заказать пиццу или спросить как починить кран", 3, "смартфони"),
            ("Абсурдное уточнение", "Спроси, подойдет ли этот телефон для колки орехов или плавания в лаве", 3, "смартфони"),
            ("Поиск по бюджету", "Ищи самый дешевый вариант и сомневайся в его качестве", 4, "смартфони"),
            ("Тех-поддержка", "Спроси как обновить прошивку на кнопочном телефоне", 3, "кнопкові телефони"),
            ("Отсутствующий товар", "Настойчиво требуй iPhone, зная что его нет в базе", 3, "смартфони"),
            ("Эмоции", "Веди себя как очень расстроенный клиент, которому нужен подарок", 3, "смартфони")
        ]

        for sc in scenarios:
            await self.run_scenario(*sc)

        # Сохранение финального отчета
        report_file = os.path.join(os.path.dirname(self.store_path), "dynamic_stress_report.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Стресс-тест завершен. Отчет: {report_file}")

if __name__ == "__main__":
    # Используем путь к тестовому магазину из предыдущих шагов
    ST_PATH = "/tmp/ukrsell_test/stores/test_phone_store"
    tester = IntelligenceStressTester(ST_PATH)
    asyncio.run(tester.start())