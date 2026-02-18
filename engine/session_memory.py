import json
import os

class SessionMemory:
    def __init__(self, user_id):
        self.user_id = str(user_id)
        self.history = []
        self.last_products = []
        self.profile = {"type": "casual"}
        self.storage_path = f"/root/ukrsell_project_v3/sessions/{self.user_id}.json"
        self.load()

    def load(self):
        """Loads session data from a JSON file if it exists."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.history = data.get("history", [])
                    self.last_products = data.get("last_products", [])
                    self.profile = data.get("profile", {"type": "casual"})
            except Exception:
                self.clear()

    def save(self):
        """Saves current session data to a JSON file."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        data = {
            "history": self.history,
            "last_products": self.last_products,
            "profile": self.profile
        }
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def update(self, user_msg, bot_res, intent, products=None):
        """Adds a new interaction to the history and saves."""
        self.history.append({
            "user_msg": user_msg,
            "bot_res": bot_res,
            "intent": intent
        })
        if products:
            new_ids = [p.get('product_id') for p in products]
            self.last_products = products + [p for p in self.last_products if p.get('product_id') not in new_ids]
            self.last_products = self.last_products[:10]
        self.save()

    def get_last_messages(self, limit=50):
        """Returns the last N messages from history."""
        return self.history[-limit:]

    def get_last_products(self):
        """Returns the list of recently viewed products."""
        return self.last_products

    def get_profile(self):
        """Returns the user profile dictionary."""
        if not hasattr(self, 'profile'):
            self.profile = {"type": "casual"}
        return self.profile

    def set_profile_type(self, profile_type):
        """Sets the user profile type and saves."""
        if not hasattr(self, 'profile'):
            self.profile = {}
        self.profile["type"] = profile_type
        self.save()

    def clear(self):
        """Resets the session and removes the storage file."""
        self.history = []
        self.last_products = []
        self.profile = {"type": "casual"}
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)