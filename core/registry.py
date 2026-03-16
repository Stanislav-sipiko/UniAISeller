# /root/ukrsell_v4/core/registry.py v8.3.4
import re
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from core.store_context import StoreContext
from core.logger import logger

class StoreRegistry:
    """
    The Platform Gatekeeper (v8.3.4). 
    Responsible for scanning the filesystem, initializing StoreContexts, 
    and maintaining a mapping of active StoreEngines.
    
    Changelog v8.3.4:
    - Sync: Removed redundant data_ready initialization (now handled by Kernel factory).
    - Cleanup: Deprecated manual component back-injection to favor Kernel's engine_factory.
    - Reliability: Atomic updates for stores and engines mapping.
    """
    SLUG_REGEX = re.compile(r"^[a-z0-9_-]+$")

    def __init__(self, stores_root: str = "stores", kernel: Any = None):
        """
        Registry initialization. 
        :param stores_root: Path to the directory with all stores.
        :param kernel: Reference to UkrSellKernel for injection into contexts.
        """
        self.root_path = Path(stores_root).resolve()
        self.kernel = kernel  
        self.stores: Dict[str, StoreContext] = {} 
        self.engines: Dict[str, Any] = {} # Stores StoreEngine objects

    def _is_valid_slug(self, slug: str) -> bool:
        """Check the validity of the store folder name (slug)."""
        return bool(self.SLUG_REGEX.match(slug))

    async def load_all(self, engine_factory: Callable[['StoreContext'], Any], llm_selector: Any, only_slug: Optional[str] = None):
        """
        Scans root_path, creates StoreContext for each folder, 
        and initializes StoreEngine via the provided factory.
        
        :param engine_factory: Factory function from Kernel to build the engine.
        :param llm_selector: LLMSelector instance for model routing.
        :param only_slug: If provided, only this store will be loaded.
        """
        logger.info(f"🚀 Starting store scan in: {self.root_path}")
        if only_slug:
            logger.info(f"🎯 Isolation mode active: Only loading [{only_slug}]")
        
        if not self.root_path.exists():
            logger.warning(f"Directory {self.root_path} not found. Creating...")
            self.root_path.mkdir(parents=True)
            return

        new_stores = {}
        new_engines = {}

        # Iterate through folders in the stores directory
        for folder in self.root_path.iterdir():
            if not folder.is_dir() or folder.name.startswith("."):
                continue

            slug = folder.name.lower()
            
            # --- Store Filtering ---
            if only_slug and slug != only_slug.lower():
                continue

            if not self._is_valid_slug(slug):
                logger.error(f"INVALID SLUG: '{slug}'. Skipping...")
                continue

            try:
                # 1. Create store context
                ctx = StoreContext(
                    base_path=str(folder), 
                    db_engine=None, 
                    llm_selector=llm_selector,
                    kernel=self.kernel
                )
                ctx.slug = slug 
                
                # Note: data_ready Event is now managed by engine_factory in Kernel v8.3.4
                
                # 2. Async Initialize StoreContext (Loading configs, prompts, etc.)
                if hasattr(ctx, 'initialize'):
                    success = await ctx.initialize()
                    if not success:
                        logger.error(f"❌ Failed to initialize StoreContext for [{slug}]. Skipping...")
                        continue
                
                # 3. Create Engine via factory (Kernel handles Component Injection)
                engine = engine_factory(ctx)
                
                # Verify minimum engine viability
                if not hasattr(engine, 'handle_update'):
                    logger.error(f"❌ Engine for [{slug}] lacks handle_update() method. Skipping...")
                    continue

                # 4. Finalizing Context Links
                # In v8.3.4, we ensure references are synchronized
                if hasattr(engine, 'db') and not getattr(ctx, 'db', None):
                    ctx.db = engine.db

                new_stores[slug] = ctx
                new_engines[slug] = engine
                
                logger.info(f"✅ Store [{slug}] registered successfully. Path: {folder}")
            except Exception as e:
                logger.error(f"❌ Error loading store [{slug}]: {str(e)}", exc_info=True)

        # Atomic registry update
        self.stores = new_stores
        self.engines = new_engines
        logger.info(f"Registry updated. Active stores: {len(self.stores)}")

    def get_engine(self, slug: str) -> Optional[Any]:
        """Get StoreEngine object by its identifier."""
        return self.engines.get(slug.lower())

    def get_context(self, slug: str) -> Optional[StoreContext]:
        """Get StoreContext object by its identifier."""
        return self.stores.get(slug.lower())

    def get_all_slugs(self) -> List[str]:
        """Returns a list of identifiers for all active stores."""
        return list(self.engines.keys())

    async def unload_store(self, slug: str):
        """
        Correctly unloads store from memory and closes its resources.
        """
        slug = slug.lower()
        if slug in self.engines:
            engine = self.engines[slug]
            if hasattr(engine, 'close'):
                try:
                    if asyncio.iscoroutinefunction(engine.close):
                        await engine.close()
                    else:
                        engine.close()
                    logger.info(f"Engine resources for [{slug}] released.")
                except Exception as e:
                    logger.error(f"Error closing engine [{slug}]: {e}")
            
            del self.engines[slug]
            if slug in self.stores:
                # Cleanup references to prevent memory leaks
                self.stores[slug].db = None
                self.stores[slug].kernel = None
                if hasattr(self.stores[slug], 'dialog_manager'):
                    self.stores[slug].dialog_manager = None
                if hasattr(self.stores[slug], 'analyzer'):
                    self.stores[slug].analyzer = None
                del self.stores[slug]
            logger.info(f"Store [{slug}] fully unloaded from registry.")

    def __repr__(self):
        return f"<StoreRegistry stores={len(self.engines)} root='{self.root_path}'>"