"""
services/fake_redis.py
-----------------------
A minimal async in-memory key-value store that mirrors the redis.asyncio API
used by redis_rules.py and news_sentinel.py.

Used automatically when Redis is unavailable (no Docker).
Only implements the methods actually called by this codebase:
  mget, mset, set, get, delete, ping
"""
from __future__ import annotations
import asyncio
import time


class FakeRedis:
    """Thread-safe asyncio-compatible in-memory key/value store."""

    def __init__(self):
        self._store: dict[str, tuple[str, float | None]] = {}  # key -> (value, expires_at)

    def _is_expired(self, key: str) -> bool:
        if key not in self._store:
            return True
        _, exp = self._store[key]
        if exp is not None and time.monotonic() > exp:
            del self._store[key]
            return True
        return False

    async def ping(self) -> bool:
        return True

    async def get(self, key: str) -> str | None:
        if self._is_expired(key):
            return None
        return self._store[key][0]

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        expires_at = (time.monotonic() + ex) if ex else None
        self._store[key] = (str(value), expires_at)

    async def mget(self, *keys: str) -> list[str | None]:
        return [None if self._is_expired(k) else self._store[k][0] for k in keys]

    async def mset(self, mapping: dict) -> None:
        for k, v in mapping.items():
            self._store[str(k)] = (str(v), None)

    async def delete(self, *keys: str) -> None:
        for k in keys:
            self._store.pop(k, None)

    async def aclose(self) -> None:
        pass
