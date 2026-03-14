# /root/ukrsell_v4/core/cache_manager.py v1.3.0
import json
import os
import time
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from core.logger import logger

class SemanticCache:
    """
    Store-specific Semantic Cache for FAQ and common queries.
    v1.3.0: 
    - THRESHOLD: Reduced default threshold to 0.85 for better synonym handling (Stage 4).
    - LOGGING: Added detailed similarity score logging for HIT/MISS/CLOSE cases.
    - SYNC: Compatible with 'versioned' JSON structure (entries/query/intent).
    - AUTO-REPAIR: Corrupted or incompatible cache files are reset safely.
    """

    def __init__(self, ctx: Any, shared_model: Any):
        """
        Инициализация кэша.
        :param ctx: StoreContext (должен содержать slug и base_path)
        :param shared_model: Общая модель SentenceTransformer из ядра
        """
        self.ctx = ctx
        self.slug = getattr(ctx, 'slug', 'unknown')
        self.model = shared_model
        
        # Путь к файлу кэша в папке конкретного магазина
        self.cache_path = Path(self.ctx.base_path) / "semantic_cache.json"
        
        self.cache_data: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.cache_version = 1
        
        # Порог сходства (Этап 4): снижен до 0.85 по умолчанию.
        # Это позволяет ловить фразы типа "как оплатить" и "способы оплаты".
        self.threshold = 0.85
        if hasattr(self.ctx, 'profile') and isinstance(self.ctx.profile, dict):
            # Если в профиле магазина явно задан порог, используем его, иначе 0.85
            self.threshold = self.ctx.profile.get("cache_threshold", 0.85)

        self._load_cache()

    def _load_cache(self) -> None:
        """
        Загрузка кэша из JSON и генерация эмбеддингов при старте.
        Поддерживает структуру {"version": 1, "entries": [...]}.
        """
        if not self.cache_path.exists():
            logger.info(f"[{self.slug}] No existing semantic cache found at {self.cache_path}")
            return

        # Проверка на пустой файл (0 байт)
        if self.cache_path.stat().st_size == 0:
            logger.warning(f"[{self.slug}] Semantic cache file is empty. Removing: {self.cache_path}")
            try:
                self.cache_path.unlink()
            except Exception as e:
                logger.error(f"[{self.slug}] Could not delete empty cache file: {e}")
            return

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                try:
                    raw_data = json.load(f)
                except json.JSONDecodeError as jde:
                    logger.error(f"[{self.slug}] CRITICAL: Cache JSON corrupted: {jde}. Purging.")
                    f.close()
                    self.cache_path.unlink()
                    return

                # Обработка структуры с entries или плоского списка
                if isinstance(raw_data, dict) and "entries" in raw_data:
                    self.cache_data = raw_data["entries"]
                    self.cache_version = raw_data.get("version", 1)
                elif isinstance(raw_data, list):
                    self.cache_data = raw_data
                else:
                    logger.warning(f"[{self.slug}] Invalid cache format. Resetting.")
                    self.cache_data = []
                    return

            if self.cache_data:
                # Синхронизация имен: ищем 'query' (новый стандарт) или 'question' (старый)
                questions = []
                valid_entries = []
                
                for item in self.cache_data:
                    q_text = item.get('query') or item.get('question')
                    if q_text:
                        questions.append(str(q_text))
                        valid_entries.append(item)
                
                self.cache_data = valid_entries
                
                if questions:
                    # Кодируем все вопросы сразу (батч-процессинг)
                    self.embeddings = self.model.encode(questions, normalize_embeddings=True)
                    logger.info(f"[{self.slug}] Semantic Cache loaded: {len(self.cache_data)} items (Threshold: {self.threshold}).")
                else:
                    logger.warning(f"[{self.slug}] Cache found but no valid queries extracted.")
        
        except Exception as e:
            logger.error(f"[{self.slug}] Unexpected error loading semantic cache: {e}", exc_info=True)
            self.cache_data = []

    def get_answer(self, query: str) -> Optional[Union[str, Dict[str, Any]]]:
        """
        Поиск семантически похожего ответа в кэше.
        Возвращает содержимое поля 'intent' или 'answer'.
        """
        if self.embeddings is None or not self.cache_data or not query:
            return None

        try:
            # Эмбеддинг входящего запроса (нормализованный)
            query_emb = self.model.encode([query], normalize_embeddings=True)
            
            # Считаем косинусное сходство (dot product для нормализованных векторов)
            similarities = np.dot(self.embeddings, query_emb.T).flatten()
            
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])

            # Логика принятия решения
            if best_score >= self.threshold:
                logger.info(f"[{self.slug}] Semantic Cache HIT! Score: {best_score:.4f} (Threshold: {self.threshold})")
                entry = self.cache_data[best_idx]
                # Возвращаем intent (новый формат) или answer (старый формат)
                return entry.get('intent') or entry.get('answer')
            
            # Логирование близких промахов для отладки порога
            if best_score > 0.80:
                logger.info(f"[{self.slug}] Semantic Cache CLOSE MISS. Score: {best_score:.4f} (Required: {self.threshold})")
            else:
                logger.debug(f"[{self.slug}] Semantic Cache MISS. Best score: {best_score:.4f}")
                
        except Exception as e:
            logger.error(f"[{self.slug}] Error during cache lookup: {e}")
            
        return None

    def add(self, query: str, answer: Union[str, Dict[str, Any]]) -> bool:
        """
        Добавление новой пары вопрос-интент в кэш.
        """
        if not query or not answer:
            return False
            
        # Не кэшируем ошибки, технические сообщения или слишком короткие ответы
        if isinstance(answer, str) and (
            "ошибка" in answer.lower() or 
            "вибачте" in answer.lower() or 
            len(answer) < 10
        ):
            return False

        # Проверка на дубликаты (чтобы не забивать кэш одинаковыми эмбеддингами)
        if self.get_answer(query):
            return False

        try:
            new_entry = {
                "query": query.strip(),
                "intent": answer,
                "timestamp": time.time()
            }
            self.cache_data.append(new_entry)

            # Сохраняем в структурированном виде
            output = {
                "version": self.cache_version,
                "entries": self.cache_data
            }

            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            # Инкрементальное обновление эмбеддингов
            new_emb = self.model.encode([query], normalize_embeddings=True)
            if self.embeddings is None:
                self.embeddings = new_emb
            else:
                self.embeddings = np.vstack([self.embeddings, new_emb])

            logger.info(f"[{self.slug}] Cache entry added. Total entries: {len(self.cache_data)}")
            return True

        except Exception as e:
            logger.error(f"[{self.slug}] Failed to update semantic cache: {e}")
            return False

    def clear(self) -> bool:
        """Полная очистка кэша магазина."""
        try:
            if self.cache_path.exists():
                os.remove(self.cache_path)
            self.cache_data = []
            self.embeddings = None
            logger.info(f"[{self.slug}] Semantic Cache cleared.")
            return True
        except Exception as e:
            logger.error(f"[{self.slug}] Failed to clear cache: {e}")
            return False

    def __repr__(self) -> str:
        return f"<SemanticCache(slug={self.slug}, items={len(self.cache_data)}, threshold={self.threshold})>"