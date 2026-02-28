import logging
import asyncio
from aiohttp import web
from sentence_transformers import SentenceTransformer

from core.registry import StoreRegistry
from core.llm_selector import LLMSelector
from core.translator import TextTranslator
from engine.base import StoreEngine

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("UkrSell_Kernel")

class PlatformKernel:
    """
    SaaS Kernel v4.6.
    Управляет жизненным циклом магазинов и общими ресурсами (L12-v2, LLM Selector).
    """
    def __init__(self, stores_dir: str = "stores"):
        # 1. Реестр и хранилище магазинов
        self.registry = StoreRegistry(stores_dir)
        self.app = web.Application()
        
        # 2. Инициализация общих Shared Services (RAM optimization)
        logger.info("Initializing Shared Platform Services...")
        
        # Загружаем мультиязычную модель один раз для всех магазинов
        self.shared_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Инициализируем селектор моделей (Groq/Cerebras)
        self.llm_selector = LLMSelector()
        
        # Глобальный переводчик
        self.translator = TextTranslator()
        
        # 3. Настройка роутинга
        self._setup_routes()
        
        # Хуки жизненного цикла aiohttp
        self.app.on_startup.append(self._on_startup)
        self.app.on_cleanup.append(self._on_cleanup)

    def _setup_routes(self):
        """Регистрация всех эндпоинтов: от вебхуков до админки."""
        # Главный вход для Telegram
        self.app.router.add_post("/webhook/{slug}/", self.webhook_handler)
        
        # Мониторинг и Health Check
        self.app.router.add_get("/health", self.health_check)
        
        # Администрирование (Reload и Статистика)
        self.app.router.add_post("/reload/{slug}/", self.reload_store_handler)
        self.app.router.add_get("/admin/{slug}/stats", self.admin_stats_handler)

    def _engine_factory(self, ctx) -> StoreEngine:
        """Фабрика для создания движков с внедрением зависимостей."""
        return StoreEngine(
            ctx=ctx, 
            shared_model=self.shared_model,
            llm_selector=self.llm_selector,
            translator=self.translator
        )

    async def _on_startup(self, app):
        """Первоначальная загрузка всех магазинов."""
        logger.info("Kernel: Starting up and loading stores...")
        self.registry.load_all(engine_factory=self._engine_factory)
        logger.info(f"Loaded slugs: {self.registry.get_all_slugs()}")

    async def _on_cleanup(self, app):
        """Грациозное завершение работы всех сессий."""
        logger.info("Kernel: Shutting down and closing sessions...")
        for slug in self.registry.get_all_slugs():
            engine = self.registry.get_engine(slug)
            if engine:
                await engine.close()

    # --- Хендлеры (Handlers) ---

    async def webhook_handler(self, request: web.Request):
        """Маршрутизация входящих обновлений Telegram."""
        slug = request.match_info.get("slug")
        engine = self.registry.get_engine(slug)

        if not engine:
            logger.warning(f"Store not found for slug: {slug}")
            return web.Response(status=404, text="Store not found")

        try:
            update = await request.json()
            # Передаем управление в StoreEngine
            await engine.handle_update(update)
            return web.Response(status=200, text="OK")
        except Exception as e:
            logger.error(f"Error in webhook [{slug}]: {e}")
            return web.Response(status=500)

    async def health_check(self, request: web.Request):
        """Статус всей системы."""
        return web.json_response({
            "status": "healthy",
            "active_stores": len(self.registry.stores),
            "loaded_slugs": self.registry.get_all_slugs(),
            "llm_fast": self.llm_selector.get_fast()[1]
        })

    async def reload_store_handler(self, request: web.Request):
        """Горячая перезагрузка конкретного магазина без остановки ядра."""
        slug = request.match_info.get("slug")
        success = self.registry.reload_store(slug, engine_factory=self._engine_factory)
        
        if not success:
            return web.Response(status=404, text="Reload failed")
        
        return web.json_response({"status": "reloaded", "slug": slug})

    async def admin_stats_handler(self, request: web.Request):
        """Диагностическая информация по магазину."""
        slug = request.match_info.get("slug")
        engine = self.registry.get_engine(slug)
        
        if not engine:
            return web.Response(status=404)
            
        return web.json_response({
            "slug": slug,
            "products_count": len(engine.retrieval.products),
            "categories": list(engine.retrieval.available_categories),
            "index_ready": engine.retrieval.index is not None
        })

    def run(self, host: str = "127.0.0.1", port: int = 8090):
        """Запуск сервера платформы."""
        logger.info(f"UkrSell Platform running on http://{host}:{port}")
        web.run_app(self.app, host=host, port=port)

if __name__ == "__main__":
    kernel = PlatformKernel()
    kernel.run()