# /root/ukrsell_v4/core/utils.py v1.0.0
"""
Общие утилиты проекта ukrsell_v4.

v1.0.0:
  - detect_language_from_titles() — определение языка по заголовкам товаров.
    Вынесено из StoreProfiler и StoreContext (DRY).
"""

from typing import List


def detect_language_from_titles(titles: List[str]) -> str:
    """
    Определяет язык магазина по набору заголовков товаров.

    Алгоритм: ищет символы уникальные для украинского (іїєґ)
    или русского (ыэёъ) языка в объединённом тексте заголовков.
    При отсутствии характерных символов — возвращает 'Ukrainian' как default.

    Аргументы:
        titles — список заголовков товаров (любой длины, None-значения игнорируются).

    Возвращает:
        'Ukrainian' | 'Russian'
    """
    sample = " ".join(t for t in titles if t).lower()
    if any(c in sample for c in "іїєґ"):
        return "Ukrainian"
    if any(c in sample for c in "ыэёъ"):
        return "Russian"
    return "Ukrainian"