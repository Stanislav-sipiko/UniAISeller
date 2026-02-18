# engine/auto_fixer.py
import json
import os
import datetime

class AutoFixer:
    """
    AutoFixer — модуль для динамического обучения FSM.
    Он записывает ошибки классификатора интентов и формирует
    список негативных примеров, которые затем подмешиваются
    в системный промпт IntentGate.
    """

    def __init__(self):
        self.patch_path = "/root/ukrsell_project_v3/fsm_soft_patch.json"
        self.log_path = "/root/ukrsell_project_v3/auto_fixer_activity.log"

    def record_fsm_error(self, user_query: str, error_reason: str):
        """
        Записывает ошибку FSM в лог и добавляет её в список негативных примеров.
        """
        timestamp = datetime.datetime.now().isoformat()

        # 1. Запись в лог
        log_entry = (
            f"[{timestamp}] FSM_ERROR: Query: '{user_query}' | "
            f"Reason: {error_reason}\n"
        )
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Error writing to log: {e}")

        # 2. Обновление патчей
        patches = self._load_patches()

        new_patch = {
            "query": user_query,
            "reason": error_reason,
            "added_at": timestamp,
            "ttl_days": 30
        }

        patches.append(new_patch)
        self._save_patches(patches)

    def _load_patches(self):
        """
        Загружает список патчей.
        Возвращает пустой список при ошибке.
        """
        if os.path.exists(self.patch_path):
            try:
                with open(self.patch_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            except Exception:
                return []
        return []

    def _save_patches(self, patches):
        """
        Сохраняет патчи, предварительно фильтруя по TTL.
        """
        now = datetime.datetime.now()
        active_patches = []

        for p in patches:
            try:
                added_at = datetime.datetime.fromisoformat(p["added_at"])
                ttl_days = p.get("ttl_days", 30)
                if (now - added_at).days < ttl_days:
                    active_patches.append(p)
            except Exception:
                continue

        # Ограничиваем список последними 20 примерами
        active_patches = active_patches[-20:]

        try:
            with open(self.patch_path, "w", encoding="utf-8") as f:
                json.dump(active_patches, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving patches: {e}")

    def get_negative_examples(self) -> str:
        """
        Возвращает строку с последними негативными примерами
        для подмешивания в системный промпт IntentGate.
        """
        patches = self._load_patches()

        if not patches:
            return ""

        last_patches = patches[-5:]

        examples = "\nДИНАМІЧНІ ПРАВИЛА (НЕ ПОВТОРЮВАТИ ЦИХ ПОМИЛОК):\n"
        for p in last_patches:
            q = p.get("query", "N/A")
            r = p.get("reason", "N/A")
            examples += f"- Запит: '{q}' | Помилка: {r}\n"

        return examples
