import logging
import time
import openai
from groq import Groq
from typing import Tuple, Any, Optional

# Assuming keys are in environment or a central config
from core.config import GROQ_API_KEY, CEREBRAS_API_KEY

logger = logging.getLogger("UkrSell_LLMSelector")

class LLMSelector:
    """
    SaaS-ready LLM Router. 
    Selects between Fast (Cerebras) and Heavy (Groq/Qwen) models.
    """
    KEYWORDS = {
        "heavy": ["qwen-3-235b", "llama-3.3-70b"],
        "fast":  ["llama3.1-8b", "llama-3.1-8b"]
    }

    def __init__(self):
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.cerebras_client = openai.OpenAI(
            api_key=CEREBRAS_API_KEY,
            base_url="https://api.cerebras.ai/v1"
        )
        self.fast_client, self.fast_model = None, None
        self.heavy_client, self.heavy_model = None, None
        self.refresh()

    def refresh(self):
        """Polls providers and sets active clients."""
        # Logic remains same as V3: list models, find matches, test ping.
        # ... (Method find_match and test_model integrated here) ...
        logger.info(f"LLM Refresh complete. Fast: {self.fast_model}, Heavy: {self.heavy_model}")

    def get_fast(self) -> Tuple[Any, str]:
        return self.fast_client, self.fast_model

    def get_heavy(self) -> Tuple[Any, str]:
        return self.heavy_client, self.heavy_model