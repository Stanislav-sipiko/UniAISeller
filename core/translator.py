# /root/ukrsell_v4/core/translator.py
import logging
import asyncio
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory

# Ensuring consistent language detection
DetectorFactory.seed = 0

logger = logging.getLogger("UkrSell_Translator")

class TextTranslator:
    """
    Universal SaaS Translator (v5.0.0).
    Provides language detection and multi-target translation capabilities.
    Oriented towards Ukrainian as the primary store database language.
    """
    def __init__(self, default_target='uk'):
        self.default_target = default_target
        # GoogleTranslator will be instantiated per-call to handle different targets
        logger.info(f"TextTranslator initialized. Default target: {default_target}")

    async def detect_language(self, text: str) -> str:
        """
        Detects the language of the input text using langdetect.
        Returns ISO language code (e.g., 'ru', 'uk', 'en').
        """
        try:
            if not text or len(text.strip()) < 3:
                return "unknown"
            
            # Running synchronous detection in a thread to keep the event loop free
            detected = await asyncio.to_thread(detect, text)
            logger.debug(f"Language detection: '{text[:20]}...' -> {detected}")
            return detected
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return "unknown"

    async def translate(self, text: str, target_lang: str = None) -> str:
        """
        Asynchronous wrapper for GoogleTranslator.
        Automatically detects source and translates to target_lang.
        """
        target = target_lang or self.default_target
        try:
            if not text or len(text.strip()) < 2:
                logger.debug(f"Skip translation: text too short ('{text}')")
                return text
            
            if text.replace(" ", "").isdigit():
                logger.debug(f"Skip translation: text is numeric ('{text}')")
                return text

            # Creating a fresh translator instance for this specific target
            translator_obj = GoogleTranslator(source='auto', target=target)
            
            logger.info(f"Translating to {target}: '{text[:50]}'")
            
            # Perform translation in a separate thread
            try:
                translated = await asyncio.to_thread(translator_obj.translate, text)
            except Exception as thread_e:
                logger.error(f"Worker thread crash during translation: {thread_e}")
                return text
            
            if translated:
                if translated.strip().lower() == text.strip().lower():
                    logger.info("Translation result is identical to source.")
                else:
                    logger.info(f"Translation successful: '{text[:30]}...' -> '{translated[:30]}...'")
                return translated
            
            logger.warning(f"Translator returned empty result for: '{text[:30]}'")
            return text
            
        except Exception as e:
            logger.error(f"Critical translation error: {e}", exc_info=True)
            return text

    def __repr__(self):
        return f"<TextTranslator default_target={self.default_target}>"