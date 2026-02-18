#!/usr/bin/env python3
# main.py v11.0.0 - Lucky Dog SaaS Orchestrator (Zero Omission)

import os
import json
import asyncio
import logging
import re
import time
from typing import Dict, Any, List, Tuple

from aiohttp import web

from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

from config import BOT_TOKEN

from llm_selector import LLMSelector
from engine.session_memory import SessionMemory
from engine.followup_engine import FollowUpEngine
from engine.retrieval_engine import RetrievalEngine
from engine.planner_llm import PlannerLLM
from engine.intent_gate import IntentGate
from engine.fact_checker import FactChecker
from engine.objection_handler import ObjectionHandler
from engine.auto_fixer import AutoFixer
from engine.policy_controller import PolicyController

# ==============================================
# КОНФИГУРАЦИЯ И ПУТИ
# ==============================================

PROJECT_ROOT = "/root/ukrsell_project_v3"

DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "stores",
    "lucky_dog",
    "products_search_optimized.json"
)

NORMALIZED_PATH = os.path.join(
    PROJECT_ROOT,
    "stores",
    "lucky_dog",
    "products_normalized.json"
)

CHROMA_DB_PATH = os.path.join(
    PROJECT_ROOT,
    "stores",
    "lucky_dog",
    "chroma_db",
    "chroma.sqlite3"
)

PATCH_FILE = os.path.join(
    PROJECT_ROOT,
    "fsm_soft_patch.json"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LuckyDog_v11_0_0")

NO_RESULTS_ANSWERS = [
    "К сожалению, по вашему запросу ничего не найдено. Попробуем поискать что-то другое?",
    "Я проверил весь ассортимент, но именно этого товара сейчас нет. Загляните к нам попозже!",
    "Гав! Наши хвостики разобрали этот товар слишком быстро. Найдем замену?"
]

OFFTOPIC_ANSWERS = [
    "Я консультирую только по товарам для животных в Lucky Dog. Чем могу помочь вашему любимцу?",
    "Извините, но я специализируюсь на зоотоварах. Вас интересует что-то для вашего хвостика?"
]

# ==============================================
# FSM И КЭШ
# ==============================================

class ShopFSM(StatesGroup):
    waiting_for_query = State()


user_log_cache: Dict[str, List[Dict[str, Any]]] = {}

# ==============================================
# УТИЛИТЫ
# ==============================================

def md_escape(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'([_*[\]()~`>#+=|{}.!-])', r'\\\1', str(text))


def url_escape(url: str) -> str:
    if not url:
        return "#"
    return url.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def parse_price(val) -> float:
    try:
        if isinstance(val, (int, float)):
            return float(val)
        clean = re.sub(r"[^\d.]", "", str(val).replace(",", "."))
        return float(clean) if clean else 0.0
    except Exception:
        return 0.0


def extract_price_limit(text: str):
    text_low = text.lower()
    m = re.search(r"(?:до|меньше|за|бюджет|цена|вартість|<=|≤|~)\s*(\d+)", text_low)
    if m:
        return float(m.group(1))
    m_alt = re.search(r"(\d+)\s*(?:грн|uah|гривень)", text_low)
    if m_alt:
        return float(m_alt.group(1))
    return None


def is_followup_navigation(text: str) -> bool:
    t = text.lower()
    phrases = [
        "еще",
        "ещё",
        "еще варианты",
        "ещё варианты",
        "покажи еще",
        "покажи ещё",
        "что еще есть",
        "что ещё есть",
        "и все",
        "и всё"
    ]
    return any(p in t for p in phrases)


# ==============================================
# ЛЛМ И ДВИЖКИ
# ==============================================

llm_selector = LLMSelector()
fast_client, fast_model = llm_selector.get_fast()
heavy_client, heavy_model = llm_selector.get_heavy()

retrieval_engine = RetrievalEngine()
followup_engine = FollowUpEngine()
planner_llm = PlannerLLM()
intent_gate = IntentGate()
fact_checker = FactChecker()
objection_handler = ObjectionHandler()
auto_fixer = AutoFixer()
policy_controller = PolicyController()


class LegacySearchEngine:
    """
    Обёртка над старым products_search_optimized.json,
    чтобы сохранить поведение v7.9.1 как fallback.
    """

    def __init__(self):
        self.index: List[Dict[str, Any]] = []
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.index = raw
                logger.info(f"LegacySearchEngine: загружено {len(self.index)} товаров из products_search_optimized.json")
            except Exception as e:
                logger.error(f"LegacySearchEngine load error: {e}")
                self.index = []
        else:
            logger.warning(f"LegacySearchEngine: файл {DATA_FILE} не найден")

    def search(self, query: str, animal: str = None, price_limit: float = None) -> List[Dict[str, Any]]:
        q_words = re.findall(r"\w+", query.lower())
        if not q_words and not price_limit:
            return []

        scored: List[Tuple[float, float, Dict[str, Any]]] = []
        a_tokens: List[str] = []

        if animal == "собаки":
            a_tokens = ["собак", "пес", "цуцен", "dog", "puppy", "щенок", "щеня"]
        elif animal == "кошки":
            a_tokens = ["кот", "кош", "кіт", "cat", "kitten", "кошеня", "котя"]

        for p in self.index:
            title = p.get("title", "")[:120].lower()
            blob = p.get("search_blob", "").lower()
            price = parse_price(p.get("price_current"))

            if price_limit and price > price_limit:
                continue

            score = 0.0
            if query.lower() in title:
                score += 25.0

            for w in q_words:
                if w in title:
                    score += 5.0
                elif w in blob:
                    score += 2.0

            if not q_words and price_limit:
                score = 1.0

            if score > 0:
                if a_tokens and any(at in blob for at in a_tokens):
                    score += 15.0
                scored.append((score, price, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[2] for x in scored[:5]]


legacy_search = LegacySearchEngine()

# ==============================================
# ОСНОВНОЙ ПАЙПЛАЙН ОБРАБОТКИ ЗАПРОСА
# ==============================================

async def process_query(user_id: int, text: str, state: FSMContext) -> str:
    """
    Главный оркестратор:
    - IntentGate: определяем тип запроса
    - SessionMemory: поднимаем историю и last_products
    - FollowUpEngine: пытаемся обработать уточнение
    - PlannerLLM: генерируем уточнённые поисковые запросы
    - RetrievalEngine + LegacySearchEngine: ищем товары
    - PolicyController: выбираем CTA
    - FactChecker: формируем финальный ответ
    - ObjectionHandler: перехватываем возражения
    """

    uid_str = str(user_id)
    session = SessionMemory(uid_str)

    # 1. Intent detection
    intent_result = intent_gate.detect_intent(text)
    intent = intent_result.get("intent", "product_query")
    safe_query = intent_result.get("query", text)

    # 2. Offtopic — сразу короткий ответ
    if intent == "offtopic":
        return random.choice(OFFTOPIC_ANSWERS)

    # 3. Поднимаем контекст
    last_products = session.get_last_products()
    history = session.get_last_messages(limit=20)
    profile = session.get_profile()
    user_type = profile.get("type", "casual")

    # 4. Follow-up попытка
    raw_hits = followup_engine.process(safe_query, last_products, retrieval_engine)
    results_map: Dict[str, List[Dict[str, Any]]] = {}

    if raw_hits:
        # Follow-up сработал: работаем только с last_products
        results_map[safe_query] = raw_hits
    else:
        # 5. PlannerLLM: генерируем поисковые подзапросы
        search_queries = planner_llm.extract_search_queries(safe_query)

        for q in search_queries:
            # Основной поиск через RetrievalEngine (Meilisearch + normalized.json)
            hits, distances = retrieval_engine.query(q, limit=5)

            # Если RetrievalEngine ничего не дал — fallback в LegacySearchEngine
            if not hits:
                legacy_hits = legacy_search.search(q)
                hits = legacy_hits
                distances = [0.5 for _ in hits]

            results_map[q] = hits

    # 6. Если вообще ничего не нашли
    any_hits = any(results_map.values())
    if not any_hits:
        # Запишем как "абсурд" для AutoFixer, если это реально странный запрос
        auto_fixer.record_fsm_error(safe_query, "no_results")
        return random.choice(NO_RESULTS_ANSWERS)

    # 7. Обработка возражений (если intent == objection)
    if intent == "objection":
        # Берём все хиты в один список
        flat_hits: List[Dict[str, Any]] = []
        for v in results_map.values():
            flat_hits.extend(v)
        objection_answer = objection_handler.handle(safe_query, flat_hits)
        if objection_answer:
            session.update(user_msg=text, bot_res=objection_answer, intent=intent, products=flat_hits[:3])
            return objection_answer

    # 8. Выбор CTA на основе первого результата
    first_query = next(iter(results_map.keys()))
    first_hits = results_map[first_query]
    first_hit = first_hits[0] if first_hits else {}
    first_price = parse_price(first_hit.get("price_current"))
    # Для PolicyController distance мы сейчас задаём как 0.1 (в RetrievalEngine так и делается)
    cta_type = policy_controller.get_cta_type(distance=0.1, price=first_price)

    # 9. Генерация ответа через FactChecker
    answer_text = fact_checker.generate_answer(
        query=safe_query,
        results_map=results_map,
        cta_type=cta_type,
        intent=intent,
        prompt_override=None
    )

    # 10. Обновляем SessionMemory
    # Берём до 10 товаров из всех запросов
    all_products: List[Dict[str, Any]] = []
    for v in results_map.values():
        all_products.extend(v)
    session.update(
        user_msg=text,
        bot_res=answer_text,
        intent=intent,
        products=all_products[:10]
    )

    return answer_text


# ==============================================
# ХЕНДЛЕРЫ БОТА
# ==============================================

async def cmd_start(message: types.Message, state: FSMContext):
    await state.clear()
    await message.answer("Привет! 🐾 Я помощник Lucky Dog. Что ищем?")
    await state.set_state(ShopFSM.waiting_for_query)


async def handle_main_logic(message: types.Message, state: FSMContext):
    if not message.text:
        return

    text = message.text.strip()
    uid = str(message.from_user.id)

    if uid not in user_log_cache:
        user_log_cache[uid] = []

    data = await state.get_data()
    bot_action = ""

    # Meta-talk (как в v7.9.1)
    meta_triggers = ["разве", "понимаешь", "тупой", "вопрос", "fsm", "это не", "почему", "глупый"]
    if any(w in text.lower() for w in meta_triggers) and data.get("last_query"):
        bot_action = "Meta-Talk: Clarification sent"
        ans = (
            "Я ищу товары строго по словам. Если я ошибся, попробуйте написать название товара точнее "
            "(например: 'плед для собаки')."
        )
        await message.answer(ans)
        user_log_cache[uid].append({"u": text, "b": bot_action, "ts": time.time()})
        return

    # Навигация по результатам (ещё, дешевле, дороже) — пока оставим только "ещё"
    if is_followup_navigation(text) and data.get("last_query"):
        # В v11 навигацию по страницам мы делаем через повторный вызов process_query
        # с тем же last_query, но это можно расширить до offset/limit.
        last_query = data.get("last_query")
        answer = await process_query(message.from_user.id, last_query, state)
        bot_action = "FollowUp Navigation"
        await message.answer(answer, parse_mode=ParseMode.MARKDOWN)
        user_log_cache[uid].append({"u": text, "b": bot_action, "ts": time.time()})
        return

    # Абьюз / токсик
    if any(w in text.lower() for w in ["клоун", "дурак", "бесполезный", "тупой", "идиот"]):
        bot_action = "Blocked: Abuse"
        auto_fixer.record_fsm_error(text, "abuse")
        await message.answer("Пожалуйста, будьте вежливы.")
        user_log_cache[uid].append({"u": text, "b": bot_action, "ts": time.time()})
        return

    # Явный оффтоп (быстрый фильтр, до IntentGate)
    if any(w in text.lower() for w in ["погода", "кто ты", "политика"]):
        bot_action = "Blocked: Offtopic (fast)"
        await message.answer(random.choice(OFFTOPIC_ANSWERS))
        user_log_cache[uid].append({"u": text, "b": bot_action, "ts": time.time()})
        return

    # Основной пайплайн v11.0.0
    answer = await process_query(message.from_user.id, text, state)

    # Логируем
    bot_action = "v11_answer"
    user_log_cache[uid].append({"u": text, "b": bot_action, "ts": time.time()})

    # Отправляем ответ
    # FactChecker уже вернул Markdown-совместимый текст
    await message.answer(answer, parse_mode=ParseMode.MARKDOWN)


# ==============================================
# АДМИНКА И КАТАЛОГ
# ==============================================

async def admin_handler(request: web.Request) -> web.Response:
    style = """
    <style>
        body { font-family: sans-serif; padding: 20px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; vertical-align: top;}
        th { background-color: #f4f4f4; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .ts { color: #888; font-size: 0.85em; width: 80px; }
        .u-col { width: 30%; font-weight: bold; color: #333; }
        .b-col { color: #0056b3; }
    </style>
    """
    html = f"<html><head><meta charset='utf-8'>{style}</head><body>"
    html += "<h1>📊 Live Audit: v11.0.0</h1>"

    for uid, logs in user_log_cache.items():
        html += f"<h3>👤 User: {uid}</h3>"
        html += "<table><tr><th>Time</th><th>User Input</th><th>Bot Action</th></tr>"
        for l in logs:
            t_str = time.strftime("%H:%M:%S", time.localtime(l["ts"]))
            html += (
                f"<tr><td class='ts'>{t_str}</td>"
                f"<td class='u-col'>{l['u']}</td>"
                f"<td class='b-col'>{l['b']}</td></tr>"
            )
        html += "</table><hr>"

    html += "</body></html>"
    return web.Response(text=html, content_type="text/html")


async def catalog_handler(request: web.Request) -> web.Response:
    if os.path.exists(NORMALIZED_PATH):
        try:
            with open(NORMALIZED_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return web.json_response(data[:30])
        except Exception:
            return web.json_response({"error": "read error"}, status=500)
    return web.json_response({"error": "file not found"}, status=404)


# ==============================================
# ЗАПУСК СЕРВЕРА
# ==============================================

async def main():
    if not BOT_TOKEN:
        print("CRITICAL: BOT_TOKEN is missing in config.py")
        return

    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher(storage=MemoryStorage())

    dp.message.register(cmd_start, F.text == "/start")
    dp.message.register(handle_main_logic, F.text)

    app = web.Application()
    SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path="/")
    app.router.add_get("/adminbot/", admin_handler)
    app.router.add_get("/catalog", catalog_handler)

    setup_application(app, dp, bot=bot)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8084)
    await site.start()

    logger.info("Bot v11.0.0 Online on port 8084")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
