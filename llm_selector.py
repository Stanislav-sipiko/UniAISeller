# llm_selector.py — v11.0.0
import requests
import time
from groq import Groq
import openai
from config import GROQ_API_KEY, CEREBRAS_API_KEY

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

KEYWORDS = {
    "heavy": ["llama-3.3-70b", "qwen-3-235b"],
    "fast":  ["llama3.1-8b", "llama-3.1-8b"]
}

class LLMSelector:
    def __init__(self):
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.cerebras_client = openai.OpenAI(
            api_key=CEREBRAS_API_KEY,
            base_url=CEREBRAS_BASE_URL
        )

        self.groq_models = []
        self.cerebras_models = []

        self.fast_client = None
        self.fast_model = None

        self.heavy_client = None
        self.heavy_model = None

        self.refresh()

    # -----------------------------
    # MODEL LISTING
    # -----------------------------
    def list_groq(self):
        try:
            resp = self.groq_client.models.list()
            return [m.id for m in resp.data]
        except Exception:
            return []

    def list_cerebras(self):
        try:
            resp = self.cerebras_client.models.list()
            return [m.id for m in resp.data]
        except Exception:
            return []

    # -----------------------------
    # MODEL TEST
    # -----------------------------
    def test_model(self, client, model):
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=4
            )
            return True
        except Exception:
            return False

    # -----------------------------
    # FIND MATCH
    # -----------------------------
    def find_match(self, models, keywords):
        for kw in keywords:
            for m in models:
                if kw in m:
                    return m
        return None

    # -----------------------------
    # REFRESH ROUTING
    # -----------------------------
    def refresh(self):
        self.groq_models = self.list_groq()
        self.cerebras_models = self.list_cerebras()

        # HEAVY: prefer GROQ
        heavy_groq = self.find_match(self.groq_models, KEYWORDS["heavy"])
        heavy_cer  = self.find_match(self.cerebras_models, KEYWORDS["heavy"])

        if heavy_groq and self.test_model(self.groq_client, heavy_groq):
            self.heavy_client = self.groq_client
            self.heavy_model = heavy_groq
        elif heavy_cer and self.test_model(self.cerebras_client, heavy_cer):
            self.heavy_client = self.cerebras_client
            self.heavy_model = heavy_cer

        # FAST: prefer CEREBRAS
        fast_cer  = self.find_match(self.cerebras_models, KEYWORDS["fast"])
        fast_groq = self.find_match(self.groq_models, KEYWORDS["fast"])

        if fast_cer and self.test_model(self.cerebras_client, fast_cer):
            self.fast_client = self.cerebras_client
            self.fast_model = fast_cer
        elif fast_groq and self.test_model(self.groq_client, fast_groq):
            self.fast_client = self.groq_client
            self.fast_model = fast_groq

    # -----------------------------
    # PUBLIC API
    # -----------------------------
    def get_fast(self):
        return self.fast_client, self.fast_model

    def get_heavy(self):
        return self.heavy_client, self.heavy_model
