"""Short-lived response cache for retries/refreshes/double-submits. Keyed by request + model version, so a new
@champion invalidates it automatically. Values are rankings only.

The key is an HMAC with a random per-process secret: it identifies identical requests (cache, inference log) but,
unlike a plain hash of a payload that contains name and phone, cannot be matched by hashing guessed inputs."""
import hashlib
import hmac
import json
import secrets
import threading

from cachetools import TTLCache


class ResponseCache:
    def __init__(self, ttl_s: float, max_items: int, secret: bytes | None = None):
        self._secret = secret or secrets.token_bytes(32)
        self.enabled = max_items > 0  # CACHE_MAX_ITEMS=0 turns caching off (e.g. for honest load tests)
        self._cache: TTLCache = TTLCache(maxsize=max(max_items, 1), ttl=ttl_s)
        self._lock = threading.Lock()

    def key(self, payload: dict, model_version: str) -> str:
        canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        return hmac.new(self._secret, f"{model_version}|{canonical}".encode(), hashlib.sha256).hexdigest()

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
