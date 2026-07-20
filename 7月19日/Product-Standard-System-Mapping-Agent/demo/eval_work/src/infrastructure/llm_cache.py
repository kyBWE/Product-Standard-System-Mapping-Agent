from __future__ import annotations
from collections import OrderedDict
import hashlib
import logging


logger = logging.getLogger("LLMCache")


class LLMResponseCache:
    """基于 (method, input_hash) 的 LRU 缓存。"""

    def __init__(self, maxsize: int = 512):
        self._maxsize = max(1, maxsize)
        self._store: OrderedDict[tuple[str, str], str] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(method: str, *parts: str) -> tuple[str, str]:
        payload = method + "\x1f" + "\x1f".join(parts)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return method, digest

    def get(self, method: str, *parts: str) -> str | None:
        key = self.make_key(method, *parts)
        if key not in self._store:
            self.misses += 1
            return None
        self.hits += 1
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, method: str, value: str, *parts: str) -> None:
        key = self.make_key(method, *parts)
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0
