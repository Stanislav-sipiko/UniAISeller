# /root/ukrsell_v4/core/logger.py
import logging
import os
import json
from logging.handlers import RotatingFileHandler
from datetime import datetime
from core.config import BASE_DIR, DEBUG

# Путь к логам (создаем папку, используя BASE_DIR из исправленного config.py)
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Имя файла лога
LOG_FILE = os.path.join(LOG_DIR, "ukrsell_v4.log")

# Основной формат: Дата | Уровень | Модуль | Сообщение
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Настройка базового логгера
logger = logging.getLogger("UkrSell_V4")
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    # 1. Файловый хендлер с ротацией (5MB на файл, храним 5 последних)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. Консольный хендлер для разработки (красивый вывод в терминал)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def log_event(event_type: str, payload: dict | str, level: str = "info"):
    """
    Централизованный метод записи событий. 
    Автоматически сериализует словари в JSON для удобного парсинга в будущем.
    """
    if isinstance(payload, dict):
        # Превращаем в компактный JSON без лишних пробелов
        message = f"{event_type} | JSON: {json.dumps(payload, ensure_ascii=False)}"
    else:
        message = f"{event_type} | {str(payload)}"
    
    # Динамический выбор уровня (info, debug, error, и т.д.)
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message)

# Специальный логгер для трекинга воронки (Pipeline Tracker)
def log_pipeline_step(step_name: str, duration: float, status: str = "OK", extra: dict = None):
    """Логирует время выполнения каждого этапа в Kernel."""
    payload = {
        "step": step_name,
        "ms": round(duration * 1000, 2),
        "status": status,
        "extra": extra or {}
    }
    log_event("PIPELINE_METRIC", payload)


def log_llm_request(slug: str, tier: str, model: str, provider: str,
                    prompt_preview: str = "", max_tokens: int = 0):
    """Логирует исходящий запрос к LLM — перед вызовом API."""
    log_event("LLM_REQUEST", {
        "slug": slug,
        "tier": tier,
        "model": model,
        "provider": provider,
        "max_tokens": max_tokens,
        "prompt_preview": prompt_preview[:120] if prompt_preview else "",
    })


def log_llm_response(slug: str, model: str, duration_ms: float,
                     response_preview: str = "", finish_reason: str = "",
                     tokens_used: int = 0):
    """Логирует ответ LLM — после вызова API."""
    log_event("LLM_RESPONSE", {
        "slug": slug,
        "model": model,
        "ms": round(duration_ms, 1),
        "finish_reason": finish_reason,
        "tokens_used": tokens_used,
        "response_preview": response_preview[:120] if response_preview else "",
    })


def log_llm_error(slug: str, model: str, provider: str,
                  error_type: str, error_msg: str):
    """Логирует ошибку LLM-вызова с контекстом для диагностики."""
    log_event("LLM_ERROR", {
        "slug": slug,
        "model": model,
        "provider": provider,
        "error_type": error_type,
        "error_msg": str(error_msg)[:200],
    }, level="error")


def log_retrieval(slug: str, query_preview: str, faiss_candidates: int,
                  after_price_filter: int, after_entity_filter: int,
                  final_count: int, detected_category: str = "",
                  score_min: float = 0.0, score_max: float = 0.0,
                  duration_ms: float = 0.0):
    """Логирует полную воронку retrieval — для анализа качества поиска."""
    log_event("RETRIEVAL_FUNNEL", {
        "slug": slug,
        "query": query_preview[:60],
        "category_detected": detected_category,
        "faiss_raw": faiss_candidates,
        "after_price": after_price_filter,
        "after_entity": after_entity_filter,
        "final": final_count,
        "score_range": f"{score_min:.3f}–{score_max:.3f}",
        "ms": round(duration_ms, 1),
    })


def log_intent(slug: str, chat_id: str, model: str, action: str,
               entities: dict, duration_ms: float, from_cache: bool = False):
    """Логирует результат intent-анализа от LLM."""
    log_event("INTENT_RESULT", {
        "slug": slug,
        "chat_id": str(chat_id),
        "model": model,
        "action": action,
        "entities": entities,
        "ms": round(duration_ms, 1),
        "from_cache": from_cache,
    })


def log_model_selected(slug: str, tier: str, model: str,
                       provider: str, latency_ms: float = 0.0,
                       fallback: bool = False):
    """Логирует какая модель выбрана селектором и из какого tier."""
    log_event("MODEL_SELECTED", {
        "slug": slug,
        "tier": tier,
        "model": model,
        "provider": provider,
        "latency_ms": round(latency_ms, 1),
        "fallback": fallback,
    })