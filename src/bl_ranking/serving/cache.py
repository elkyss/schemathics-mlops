"""Short-lived response cache for retries/refreshes/double-submits. Keyed by request + model version, so a new
@champion invalidates it automatically. Values are rankings only (no personal data)."""
import hashlib
import json
import threading

from cachetools import TTLCache


class ResponseCache:
    def __init__(self, ttl_s: float, max_items: int):
        self.enabled = max_items > 0  # CACHE_MAX_ITEMS=0 turns caching off (e.g. for honest load tests)
        self._cache: TTLCache = TTLCache(maxsize=max(max_items, 1), ttl=ttl_s)
        self._lock = threading.Lock()

    @staticmethod
    def key(payload: dict, model_version: str) -> str:
        canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(f"{model_version}|{canonical}".encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        with self._lock:
            return self._cache.get(key)

    def put(self, key: str, value: dict) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._cache[key] = value
