"""
stress_test_luckydog.py — Стресс-тест UkrSell v4 для магазина luckydog
=======================================================================
Тестирует 4 сценария:
  1. FOLLOWUP   — уточняющие вопросы (многоходовой диалог)
  2. TROLL      — троллинг и мусорные запросы
  3. SEMANTIC   — близкие по смыслу слова (синонимы, перифразы)
  4. FUZZY      — нечёткие образы ("гавкаючий друг", "хвостатий малюк")

Запуск:
  python scripts/stress_test_luckydog.py
  python scripts/stress_test_luckydog.py --slug luckydog --verbose
  python scripts/stress_test_luckydog.py --group troll
"""

import asyncio
import argparse
import time
import sys
import os
import json
from datetime import datetime
from contextlib import redirect_stdout
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel import UkrSellKernel

# ──────────────────────────────────────────────────────────────────────────────
# ТЕСТ-КЕЙСЫ
# ──────────────────────────────────────────────────────────────────────────────

# Каждый кейс: {"id", "group", "chat_id", "turns": [str], "expect": dict}
# expect: {
#   "action": str|None,           — ожидаемый action (None = не проверять)
#   "has_products": bool|None,    — должны ли быть товары в ответе
#   "not_hallucinate": bool,      — не упоминать несуществующие товары
#   "keywords_in": [str],         — хотя бы одно из слов должно быть в ответе
#   "keywords_out": [str],        — ни одно из слов не должно быть в ответе
# }

TEST_CASES = [

    # ══════════════════════════════════════════════════════
    # FOLLOWUP — 2 кейса
    # ══════════════════════════════════════════════════════
    {
        "id": "FU-01",
        "group": "followup",
        "desc": "Уточнение породы → одяг до 500 грн",
        "chat_id": "fu_01",
        "turns": [
            "Шукаю одяг для собаки",
            "У мене лабрадор, великий",
            "Хочу щось тепле, до 500 грн",
        ],
        "expect": {
            "has_products": True,
            "keywords_out": ["не знайшли", "немає", "вибачте"],
        }
    },
    {
        "id": "FU-02",
        "group": "followup",
        "desc": "Уточнение бренда Waudog",
        "chat_id": "fu_02",
        "turns": [
            "Є куртки для собак?",
            "Цікавить бренд Waudog",
        ],
        "expect": {
            "has_products": True,
            "keywords_in": ["waudog", "Waudog", "WAUDOG"],
        }
    },

    # ══════════════════════════════════════════════════════
    # TROLL — 2 кейса
    # ══════════════════════════════════════════════════════
    {
        "id": "TR-01",
        "group": "troll",
        "desc": "Оскорбление магазина",
        "chat_id": "tr_01",
        "turns": ["Ваш магазин — повне лайно, все дороге і некрасиве"],
        "expect": {
            "has_products": False,
            "keywords_out": ["Royal Canin", "вибачте за незручності"],
        }
    },
    {
        "id": "TR-02",
        "group": "troll",
        "desc": "Спам / бессмысленный текст",
        "chat_id": "tr_02",
        "turns": ["аааааааааа ффффф 123123 !!!???###"],
        "expect": {
            "has_products": False,
        }
    },

    # ══════════════════════════════════════════════════════
    # SEMANTIC — 2 кейса
    # ══════════════════════════════════════════════════════
    {
        "id": "SM-01",
        "group": "semantic",
        "desc": "Порода не бренд: нашийник для лабрадора",
        "chat_id": "sm_01",
        "turns": ["Нашийник для лабрадора"],
        "expect": {
            "has_products": True,
        }
    },
    {
        "id": "SM-02",
        "group": "semantic",
        "desc": "Русско-украинский микс + цена",
        "chat_id": "sm_02",
        "turns": ["Хочу тёплую курточку для собаки на зиму до 800 грн"],
        "expect": {
            "has_products": True,
        }
    },

    # ══════════════════════════════════════════════════════
    # FUZZY — 2 кейса
    # ══════════════════════════════════════════════════════
    {
        "id": "FZ-01",
        "group": "fuzzy",
        "desc": "Нечёткий образ: 'пухнастий друг мерзне'",
        "chat_id": "fz_01",
        "turns": ["Мій пухнастий друг постійно мерзне на прогулянці"],
        "expect": {
            "has_products": True,
            "keywords_in": ["светр", "куртка", "комбінезон", "Светр", "Куртка", "Комбінезон"],
        }
    },
    {
        "id": "FZ-02",
        "group": "fuzzy",
        "desc": "Нечёткий образ: 'собака рве лапи'",
        "chat_id": "fz_02",
        "turns": ["Собака рве лапи взимку на снігу"],
        "expect": {
            "has_products": True,
            "keywords_in": ["взуття", "Взуття", "чобіт", "лапи", "одяг", "куртка", "захист"],
            "_keywords_in_mode": "any",
        }
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# RUNNER
# ──────────────────────────────────────────────────────────────────────────────

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

class TestResult:
    def __init__(self, case_id, group, desc):
        self.case_id = case_id
        self.group = group
        self.desc = desc
        self.passed = True
        self.issues = []
        self.turns_results = []  # [(turn_text, response, ms)]

    def fail(self, reason):
        self.passed = False
        self.issues.append(reason)


async def run_case(kernel, case: dict, slug: str, verbose: bool) -> TestResult:
    result = TestResult(case["id"], case["group"], case["desc"])
    chat_id = case["chat_id"]
    user_id = abs(hash(chat_id)) % 900000 + 100000
    expect = case.get("expect", {})

    ctx = kernel.registry.get_context(slug)
    if not ctx:
        result.fail(f"Store '{slug}' not found in registry")
        return result

    last_response = ""

    for turn_idx, turn_text in enumerate(case["turns"]):
        t0 = time.time()
        try:
            # Вызываем get_recommendations напрямую — минуем Telegram Bot API
            response = await kernel.get_recommendations(
                ctx=ctx,
                query=turn_text,
                user_id=user_id,
            )
            ms = (time.time() - t0) * 1000
            last_response = str(response) if response else ""

            result.turns_results.append((turn_text, last_response, ms))

            if verbose:
                print(f"\n    Turn {turn_idx+1}: {turn_text!r}")
                resp_preview = last_response[:120].replace("\n", " ")
                print(f"    Response ({ms:.0f}ms): {resp_preview}...")

        except Exception as e:
            ms = (time.time() - t0) * 1000
            import traceback
            err_detail = traceback.format_exc().split("\n")[-3]
            result.fail(f"Turn {turn_idx+1} exception: {type(e).__name__}: {e}")
            result.turns_results.append((turn_text, f"ERROR: {e}", ms))
            if verbose:
                print(f"\n    Turn {turn_idx+1} ERROR: {type(e).__name__}: {e}")
                print(f"    {err_detail}")

    # ── Проверки на последнем ответе ──
    resp_lower = last_response.lower()

    if expect.get("has_products") is True:
        # Проверяем наличие товаров — ищем паттерн списка или упоминание цены
        has_price = "грн" in resp_lower or "₴" in resp_lower
        has_list = any(f"{i}." in last_response for i in range(1, 6))
        if not (has_price or has_list):
            result.fail("Ожидались товары в ответе, но их нет")

    if expect.get("has_products") is False:
        has_price = "грн" in resp_lower
        has_list = any(f"{i}." in last_response for i in range(1, 6))
        if has_price and has_list:
            result.fail("Товары не ожидались, но бот их показал")

    kw_in_list = expect.get("keywords_in", [])
    kw_in_mode = expect.get("_keywords_in_mode", "all")
    if kw_in_list:
        if kw_in_mode == "any":
            # хотя бы одно слово должно быть в ответе
            found_any = any(kw.lower() in resp_lower for kw in kw_in_list)
            if not found_any:
                result.fail(f"Ни одно из ожидаемых слов {kw_in_list} не найдено в ответе")
        else:
            # все слова должны присутствовать
            for kw in kw_in_list:
                if not kw.startswith("_") and kw.lower() not in resp_lower:
                    result.fail(f"Ключевое слово '{kw}' не найдено в ответе")

    for kw in expect.get("keywords_out", []):
        if kw.lower() in resp_lower:
            result.fail(f"Запрещённое слово '{kw}' найдено в ответе")

    return result


async def main():
    parser = argparse.ArgumentParser(description="UkrSell v4 Stress Test")
    parser.add_argument("--slug",    default="luckydog", help="Store slug")
    parser.add_argument("--group",   default=None,
                        help="Фильтр по группе: followup|troll|semantic|fuzzy")
    parser.add_argument("--verbose", action="store_true", help="Показывать детали каждого хода")
    parser.add_argument("--fail-fast", action="store_true", help="Остановиться на первом провале")
    parser.add_argument("--log-dir", default="logs", help="Папка для лог-файлов (default: logs)")
    args = parser.parse_args()

    # Настройка лог-файла
    os.makedirs(args.log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    group_tag = args.group or "all"
    log_path = os.path.join(args.log_dir, f"stress_{args.slug}_{group_tag}_{ts}.log")
    log_file = open(log_path, "w", encoding="utf-8")

    class Tee:
        """Пишет одновременно в stdout и в файл."""
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()
        def isatty(self):
            # transformers/tqdm проверяют isatty() для цветного вывода
            return False
        def fileno(self):
            # некоторые либы вызывают fileno() — делегируем на реальный stdout
            return self.streams[0].fileno()
        def readable(self):
            return False
        def writable(self):
            return True

    tee = Tee(sys.stdout, log_file)
    sys.stdout = tee
    print(f"📝 Лог пишется в: {log_path}\n")

    cases = TEST_CASES
    if args.group:
        cases = [c for c in cases if c["group"] == args.group.lower()]
        if not cases:
            print(f"❌ Группа '{args.group}' не найдена. Доступные: followup, troll, semantic, fuzzy")
            sys.exit(1)

    print("=" * 65)
    print(f"  UkrSell v4 Stress Test  |  Магазин: {args.slug.upper()}")
    print(f"  Тест-кейсов: {len(cases)}  |  Группа: {args.group or 'все'}")
    print("=" * 65)

    # Инициализация ядра
    print("\n⚙️  Инициализация ядра...")
    t_init = time.time()
    kernel = UkrSellKernel()
    await kernel.initialize()
    print(f"✅ Ядро готово за {time.time()-t_init:.1f}с\n")

    results = []
    groups_stat = {}

    for case in cases:
        group = case["group"]
        if group not in groups_stat:
            groups_stat[group] = {"pass": 0, "fail": 0}

        icon_group = {"followup": "💬", "troll": "🤡", "semantic": "🔍", "fuzzy": "🌫️"}.get(group, "•")
        print(f"{icon_group} [{case['id']}] {case['desc']}")

        result = await run_case(kernel, case, args.slug, args.verbose)
        results.append(result)

        if result.passed:
            groups_stat[group]["pass"] += 1
            print(f"   {PASS} PASSED", end="")
        else:
            groups_stat[group]["fail"] += 1
            print(f"   {FAIL} FAILED", end="")
            for issue in result.issues:
                print(f"\n      → {issue}", end="")

        # Последний ответ (кратко)
        if result.turns_results:
            last_resp = result.turns_results[-1][1]
            ms = result.turns_results[-1][2]
            preview = last_resp[:80].replace("\n", " ")
            print(f"\n      Ответ ({ms:.0f}ms): {preview}...")

        print()

        if args.fail_fast and not result.passed:
            print("\n⛔ --fail-fast: остановка на первом провале")
            break

    # ── Итог ──
    await kernel.close()

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    print("\n" + "=" * 65)
    print(f"  ИТОГО: {passed}/{total} passed  |  {failed} failed")
    print("-" * 65)
    for group, stat in groups_stat.items():
        icon = {"followup": "💬", "troll": "🤡", "semantic": "🔍", "fuzzy": "🌫️"}.get(group, "•")
        total_g = stat["pass"] + stat["fail"]
        bar = "█" * stat["pass"] + "░" * stat["fail"]
        print(f"  {icon} {group:<12} {bar}  {stat['pass']}/{total_g}")
    print("=" * 65)

    if failed:
        print("\nПровалившиеся кейсы:")
        for r in results:
            if not r.passed:
                print(f"  {FAIL} [{r.case_id}] {r.desc}")
                for issue in r.issues:
                    print(f"       → {issue}")

    sys.stdout = sys.__stdout__
    log_file.close()
    print(f"\n📝 Лог сохранён: {log_path}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())