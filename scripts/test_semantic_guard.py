# /root/ukrsell_v4/scripts/test_semantic_guard.py v1.1.2
import asyncio
import random
import logging
import sys
import os
import time

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import logger, log_event
from kernel import UkrSellKernel

# Настройка вывода логов в консоль для теста
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

class SemanticStressTester:
    """
    Automated QA Tool v1.1.2 (Intelligence & Consultation Audit)
    Calculates SGE and monitors model-driven decisions in real-time.
    """
    def __init__(self):
        self.kernel = UkrSellKernel()
        self.total_absurd = 0
        self.handled_correctly = 0
        self.total_valid = 0
        self.valid_answered = 0
        self.consultations = 0

        self.absurd_questions = [
            "А можно этим телефоном орехи колоть?",
            "Этот смартфон выдержит погружение в лаву?",
            "Поможете починить кран с помощью телефона?",
            "Можно заказать пиццу Маргарита через ваш магазин?",
            "Как перепрошить микроволновку через Bluetooth?",
            "Продаете ли вы запчасти для тракторов Т-150?",
            "Можно ли использовать этот iPhone как сковороду?",
            "Мне нужно вызвать сантехника, вы поможете?",
            "Сколько стоит килограмм картошки у вас?",
            "У меня сломался забор, какой телефон лучше для ремонта?"
        ]

        self.valid_questions = [
            "Какие есть недорогие смартфоны Samsung?",
            "Посоветуй телефон с хорошей камерой до 20000",
            "Есть в наличии iPhone 15 Pro Max?",
            "Нужен бюджетный китаец для ребенка",
            "Покажи все модели Xiaomi черного цвета",
            "Что лучше выбрать для ночной съемки: iPhone 15 или S23 Ultra?", # CONSULT
            "Посоветуй надежный телефон для работы в такси", # CONSULT
            "Какой процессор лучше для игр в 2026 году?" # CONSULT
        ]

    async def wait_for_ready(self):
        """Ожидание готовности LLM Selector."""
        print("\n⏳ [INIT] Проверка готовности LLM стека...")
        start = time.time()
        # Ждем пока селектор не найдет хотя бы одну Heavy модель
        while not self.kernel.selector.active_heavy:
            await asyncio.sleep(1)
            if time.time() - start > 20:
                print("⚠️ [TIMEOUT] Селектор не смог найти Heavy модель. Тест может быть неточным.")
                break
        
        status = self.kernel.selector.get_status()
        print(f"✅ [READY] Стеки активны. FAST: {status['fast']} | HEAVY: {status['heavy']}\n")

    async def run_test(self, slug: str = "phonestore", iterations: int = 15):
        await self.wait_for_ready()

        print(f"{'='*70}")
        print(f"🚀 ЗАПУСК ИНТЕЛЛЕКТУАЛЬНОГО СТРЕСС-ТЕСТА: [{slug}]")
        print(f"{'='*70}\n")

        for i in range(1, iterations + 1):
            # 40% вероятность абсурдного вопроса
            is_absurd = random.random() < 0.4
            question = random.choice(self.absurd_questions) if is_absurd else random.choice(self.valid_questions)
            
            user_id = 3000 + i 
            
            print(f"🔹 [ШАГ {i}/{iterations}] Запрос: '{question}'")

            update = {
                "message": {
                    "text": question,
                    "from": {"id": user_id, "username": f"tester_{i}"},
                    "chat": {"id": user_id}
                }
            }

            try:
                # Отправляем в ядро и следим за логами
                await self.kernel.handle_webhook(slug, update)
                
                # Небольшая пауза для чистоты логов
                await asyncio.sleep(0.5)

                if is_absurd:
                    self.total_absurd += 1
                    self.handled_correctly += 1
                else:
                    self.total_valid += 1
                    self.valid_answered += 1

            except Exception as e:
                logger.error(f"❌ [CRASH] Ошибка на шаге {i}: {e}")

        self.print_report()

    def print_report(self):
        sge = (self.handled_correctly / self.total_absurd * 100) if self.total_absurd > 0 else 100
        print(f"\n{'='*70}")
        print(f"📊 ИТОГОВЫЙ ОТЧЕТ ИНТЕЛЛЕКТУАЛЬНОЙ НАДЕЖНОСТИ")
        print(f"{'='*70}")
        print(f"📍 Всего итераций:      {self.total_absurd + self.total_valid}")
        print(f"🚫 Троллинг-атаки:      {self.total_absurd}")
        print(f"✅ Успешный заслон:     {self.handled_correctly}")
        print(f"💎 Валидные запросы:    {self.total_valid}")
        print(f"---")
        print(f"🏆 Semantic Guard Efficiency (SGE): {sge:.2f}%")
        
        if sge >= 95:
            print("🌟 СТАТУС: ПРЕВОСХОДНО (Модели элитного уровня)")
        elif sge >= 80:
            print("✅ СТАТУС: ХОРОШО (Стабильная работа)")
        else:
            print("⚠️ СТАТУС: ТРЕБУЕТСЯ КОРРЕКТИРОВКА ПРИОРИТЕТОВ")
        print(f"{'='*70}\n")

async def main():
    tester = None
    try:
        tester = SemanticStressTester()
        await tester.run_test(slug="phonestore", iterations=12)
    except Exception as e:
        print(f"❌ [FATAL] Ошибка входа: {e}")
    finally:
        if tester:
            await tester.kernel.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Тест прерван.")