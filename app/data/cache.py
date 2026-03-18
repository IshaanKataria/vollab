"""In-memory cache with TTL."""

import time
from typing import Any


class TTLCache:
    def __init__(self):
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        if key in self._store:
            expires, value = self._store[key]
            if time.monotonic() < expires:
                return value
            del self._store[key]
        return None

    def set(self, key: str, value: Any, ttl_seconds: float):
        self._store[key] = (time.monotonic() + ttl_seconds, value)

    def clear(self):
        self._store.clear()
