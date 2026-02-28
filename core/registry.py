import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from core.store_context import StoreContext

logger = logging.getLogger("UkrSell_Registry")

class StoreRegistry:
    """
    The Platform Gatekeeper (v4.3). 
    Enforces slug uniqueness, URL safety, and manages Lifecycle of StoreEngines.
    """
    SLUG_REGEX = re.compile(r"^[a-z0-9_-]+$")

    def __init__(self, stores_root: str = "stores"):
        self.root_path = Path(stores_root).resolve()
        self.stores: Dict[str, StoreContext] = {}
        self.engines: Dict[str, Any] = {}  # Holds StoreEngine instances

    def _is_valid_slug(self, slug: str) -> bool:
        """Strict validation for URL-safe slugs."""
        return bool(self.SLUG_REGEX.match(slug))

    def load_all(self, engine_factory: Callable[[StoreContext], Any]):
        """
        Full system scan. Initializes both StoreContext and StoreEngine.
        Can be called multiple times for a full platform refresh.
        """
        logger.info(f"Initializing platform store scan at: {self.root_path}")
        
        if not self.root_path.exists():
            self.root_path.mkdir(parents=True)
            logger.warning(f"Created missing stores root at {self.root_path}")
            return

        new_stores: Dict[str, StoreContext] = {}
        new_engines: Dict[str, Any] = {}

        for folder in self.root_path.iterdir():
            if not folder.is_dir() or folder.name.startswith("."):
                continue

            slug = folder.name.lower()

            # 1. Validation
            if not self._is_valid_slug(slug):
                logger.error(f"INVALID SLUG: '{slug}' (Use only a-z, 0-9, _, -)")
                continue

            # 2. Collision Check
            if slug in new_stores:
                logger.error(f"COLLISION: Slug '{slug}' already registered. Skipping {folder}")
                continue

            # 3. Initialization of Context and Engine
            try:
                ctx = StoreContext(folder)
                # Create the engine using the provided factory (DI)
                engine = engine_factory(ctx)
                
                new_stores[slug] = ctx
                new_engines[slug] = engine
                
                logger.info(f"REGISTERED: {ctx.summary()}")
            except Exception as e:
                logger.error(f"FAILED TO LOAD [Store: {slug}]: {e}")

        self.stores = new_stores
        self.engines = new_engines
        logger.info(f"Registry load complete. Active stores: {len(self.stores)}")

    def reload_store(self, slug: str, engine_factory: Callable[[StoreContext], Any]) -> bool:
        """
        Hot-reload a single store data and its engine without restarting the platform.
        """
        slug = slug.lower()
        ctx = self.get_context(slug)
        if ctx:
            try:
                ctx.reload()  # Reloads JSONs and resets internal state
                # Re-initialize engine with fresh data
                self.engines[slug] = engine_factory(ctx)
                logger.info(f"SUCCESSFULLY RELOADED: {slug}")
                return True
            except Exception as e:
                logger.error(f"Reload failed for {slug}: {e}")
        return False

    def get_context(self, slug: str) -> Optional[StoreContext]:
        """Retrieve context by slug. Case-insensitive."""
        return self.stores.get(slug.lower())

    def get_engine(self, slug: str) -> Optional[Any]:
        """Retrieve engine by slug. Case-insensitive."""
        return self.engines.get(slug.lower())

    def get_all_slugs(self) -> List[str]:
        """Returns a list of all successfully loaded store slugs."""
        return list(self.stores.keys())

    def __repr__(self):
        return f"<StoreRegistry(active_stores={len(self.stores)})>"