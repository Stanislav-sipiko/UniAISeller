import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("UkrSell_StoreContext")

class StoreContext:
    """
    Secure Resource Manager for a single store (v4.2).
    Handles caching, path isolation, and lazy loading of heavy objects.
    """
    CONFIG_VERSION = 1
    REQUIRED_KEYS = ["bot_token", "store_name", "indexing_fields", "filters"]
    SLUG_REGEX = re.compile(r"^[a-z0-9_-]+$")

    def __init__(self, store_path: Path):
        self.path = store_path.resolve()  # Absolute root path
        self.slug = self.path.name.lower()
        
        # 1. Immediate Slug Validation
        if not self.SLUG_REGEX.match(self.slug):
            raise ValueError(f"Invalid store slug '{self.slug}'. Use only a-z, 0-9, _, -")
        
        # In-memory cache for light, read-only data
        self._config: Dict[str, Any] = {}
        self._products: List[Dict] = []
        self._prompts: Dict[str, Any] = {}
        
        # Heavy objects placeholder (Lazy Loading)
        self._faiss_index: Any = None
        
        # Initialize and validate
        self._load_and_validate_all()

    def _get_safe_path(self, sub_path: str) -> Path:
        """
        Prevents Path Traversal attacks by verifying that 
        the resolved path stays within store boundaries.
        """
        safe_path = (self.path / sub_path).resolve()
        if not str(safe_path).startswith(str(self.path)):
            raise PermissionError(f"Security violation: path {sub_path} is outside store bounds.")
        return safe_path

    def _load_and_validate_all(self):
        """Initial load of critical config and data into memory cache."""
        # 1. Load Config
        self._config = self._read_json_file("config.json")
        
        # 2. Check Version & Required Keys
        if self._config.get("config_version", 0) < self.CONFIG_VERSION:
            logger.warning(f"Store '{self.slug}' uses outdated config (v{self._config.get('config_version')}).")
            
        for key in self.REQUIRED_KEYS:
            if key not in self._config:
                raise KeyError(f"Missing required key '{key}' in {self.slug}/config.json")

        # 3. Load & Validate Products Schema
        raw_products = self._read_json_file("products.json")
        if not isinstance(raw_products, list):
            raise TypeError(f"products.json in {self.slug} must be a LIST of objects.")
        
        # Basic record validation (all items must be dicts)
        if raw_products and not all(isinstance(p, dict) for p in raw_products):
             raise TypeError(f"Invalid record format in {self.slug}/products.json. Expected objects.")
             
        self._products = raw_products

        # 4. Load Prompts
        self._prompts = self._read_json_file("prompts.json")
        logger.debug(f"StoreContext[{self.slug}] initialized successfully.")

    def _read_json_file(self, filename: str) -> Any:
        """Internal helper to read JSON through the security gate."""
        path = self._get_safe_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Required file {filename} not found in {self.slug}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def reload(self):
        """Hot-reloads cached data and resets heavy objects."""
        logger.info(f"Hot-reloading store context: {self.slug}")
        self._load_and_validate_all()
        self._faiss_index = None  # Reset lazy-loaded index

    # --- Property Accessors (Read-only Cache) ---
    @property
    def config(self) -> Dict: 
        return self._config

    @property
    def products(self) -> List[Dict]: 
        return self._products

    @property
    def prompts(self) -> Dict: 
        return self._prompts

    # --- Lazy Loading Methods ---
    def get_faiss_index(self):
        """Loads FAISS index into memory only when first requested."""
        if self._faiss_index is None:
            index_path = "data/index.faiss"
            if self.exists(index_path):
                # NOTE: Actual faiss.read_index logic will be injected here
                logger.info(f"Lazy loaded FAISS index for {self.slug}")
                self._faiss_index = "READY"  # Simulated for now
            else:
                logger.warning(f"FAISS index requested but not found for {self.slug}")
        return self._faiss_index

    # --- File Ops API ---
    def exists(self, sub_path: str) -> bool:
        """Safe check for file existence."""
        try:
            return self._get_safe_path(sub_path).exists()
        except PermissionError:
            return False

    def save_data(self, filename: str, data: Any):
        """Allows saving to the data/ directory specifically (patches, etc)."""
        if not filename.startswith("data/"):
            filename = f"data/{filename}"
        path = self._get_safe_path(filename)
        
        # Ensure subdirectory exists (e.g., data/)
        path.parent.mkdir(exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def summary(self) -> str:
        """Returns a diagnostic overview of the store context."""
        return (f"Store[{self.slug}]: Name='{self._config['store_name']}', "
                f"Products={len(self._products)}, Filters={len(self._config['filters'])}")

    def __repr__(self):
        return f"<StoreContext slug={self.slug} products={len(self._products)}>"