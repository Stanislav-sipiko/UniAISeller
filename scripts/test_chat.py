# /root/ukrsell_v4/scripts/test_chat.py v8.3.5
import asyncio
import sys
import os
import time as _time
import traceback
from concurrent.futures import ThreadPoolExecutor

# Настройка путей проекта: корректный выход на корень из папки scripts/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Теперь импорты core и kernel сработают корректно. 
# В v8.3.4 основной класс в файле kernel.py называется UkrSellKernel.
# Мы полностью исключили core.conversation_repair, так как он не входит в v8.3.4 Industrial.
try:
    from kernel import UkrSellKernel
    from core.logger import logger
except ImportError as e:
    print(f"❌ Critical Import Error: {e}")
    print(f"DEBUG: sys.path is {sys.path}")
    print(f"HINT: Check if '/root/ukrsell_v4/kernel.py' exists and the class is named 'UkrSellKernel'")
    sys.exit(1)

def _print_header():
    """Вывод заголовка консольного интерфейса."""
    print("\033[95m" + "=" * 55 + "\033[0m")
    print("🚀 \033[1mUkrSell V4 — AI Sales Consultant\033[0m")
    print("CLI Test Mode [Analyzer v8.3.4 Hybrid Industrial]")
    print("Введите 'exit', 'quit' или 'выход' для завершения")
    print("\033[95m" + "=" * 55 + "\033[0m\n")

def _print_trace(response: dict, external_latency: float):
    """
    Синхронизированный вывод отладочной информации.
    Извлекает данные напрямую из структуры ответа Analyzer v8.3.4.
    """
    trace = response.get("trace", {})
    
    print("\033[90m" + "─" * 55)
    print(f"DEBUG INFO:")
    # Поля из корня ответа Analyzer/Kernel
    print(f"  Intent Applied:  \033[33m{response.get('intent_applied', 'N/A')}\033[0m")
    print(f"  Model Used:      {response.get('model', 'N/A')}")
    print(f"  Status:          {response.get('status', 'N/A')}")
    
    # Внутреннее время обработки LLM (из Analyzer)
    internal_ms = response.get("ms")
    if internal_ms:
        print(f"  Core Latency:    {internal_ms} ms")
    
    # Общее время (Round Trip Time)
    print(f"  Total RTT:       {round(external_latency * 1000, 1)} ms")
    
    # Дополнительные данные из trace
    if "entities" in trace:
        print(f"  Detected Entities: {trace.get('entities')}")
    if "count" in trace:
        print(f"  Products Found:    {trace.get('count')}")
    if "slug" in trace:
        print(f"  Active Store:      {trace.get('slug')}")
    
    print("─" * 55 + "\033[0m\n")

async def ainput(prompt: str) -> str:
    """
    Неблокирующий ввод для asyncio. 
    Предотвращает замирание event loop, пока пользователь пишет сообщение.
    """
    with ThreadPoolExecutor(1, "AsyncInput") as executor:
        return await asyncio.get_event_loop().run_in_executor(executor, input, prompt)

async def _initialize_kernel():
    """Безопасная инициализация ядра."""
    print("⚙️  Инициализация UkrSellKernel...")
    kernel = UkrSellKernel()

    if hasattr(kernel, "initialize"):
        try:
            if asyncio.iscoroutinefunction(kernel.initialize):
                await kernel.initialize()
            else:
                kernel.initialize()
            print("✅ Kernel initialized.")
        except Exception as e:
            print(f"❌ Ошибка в Kernel.initialize: {e}")
            traceback.print_exc()
            sys.exit(1)

    return kernel

async def _chat_loop(kernel: UkrSellKernel):
    """Основной цикл взаимодействия."""
    chat_id = "test_user_v8_industrial"
    
    # Проверка загруженных магазинов
    slugs = kernel.get_all_active_slugs()
    if slugs:
        print(f"📦 Доступные магазины: {', '.join(slugs)}")
    else:
        print("⚠️  Предупреждение: Магазины не найдены в Registry. Проверьте папку /root/ukrsell_v4/stores/")

    print("✅ Система готова. Ожидание ввода...\n")

    while True:
        try:
            user_input = await ainput("\033[94m\033[1mUSER:\033[0m ")
            user_input = user_input.strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "выход", "stop"]:
                print("\n👋 Сессия завершена. Хорошего дня!")
                break

            start_time = _time.perf_counter()

            # Вызов логики ядра (v8.3.4 Industrial)
            response = await kernel.handle_message(user_input, chat_id)

            latency = _time.perf_counter() - start_time

            if not isinstance(response, dict):
                print(f"❌ Error: Kernel returned {type(response)} instead of dict")
                continue

            bot_text = response.get("text", "[Пустой ответ]")
            print(f"\n\033[92m\033[1mBOT:\033[0m\n{bot_text}\n")

            _print_trace(response, latency)

        except KeyboardInterrupt:
            print("\n⏹ Прервано пользователем.")
            break
        except Exception as e:
            print(f"\n❌ Ошибка обработки: {e}")
            traceback.print_exc()

async def main():
    """Точка входа."""
    _print_header()
    try:
        kernel = await _initialize_kernel()
        await _chat_loop(kernel)
    except Exception as e:
        print(f"❌ Критическая ошибка приложения: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass