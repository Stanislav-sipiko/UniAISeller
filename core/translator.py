import logging
from deep_translator import GoogleTranslator

logger = logging.getLogger("TextTranslator")

class TextTranslator:
    """
    Универсальный переводчик для платформы.
    Обеспечивает понимание мультиязычных запросов.
    """
    def __init__(self, target_lang='en'):
        self.target_lang = target_lang
        self.translator = GoogleTranslator(source='auto', target=target_lang)

    async def translate(self, text: str) -> str:
        try:
            # deep-translator работает синхронно, поэтому используем заглушку для async
            # В будущем можно заменить на полностью асинхронную библиотеку
            if not text or len(text) < 2:
                return text
                
            translated = self.translator.translate(text)
            return translated
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text # Возвращаем оригинал при ошибке