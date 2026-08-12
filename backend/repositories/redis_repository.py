import json
from typing import cast

import redis.asyncio as aioredis

from backend.config import settings


class RedisRepository:
    def __init__(self) -> None:
        self._client: aioredis.Redis | None = None

    @property
    def client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(
                settings.REDIS_URL, decode_responses=True, socket_connect_timeout=5
            )
        return self._client

    async def ping(self) -> bool:
        try:
            return await self.client.ping()
        except Exception:
            return False

    async def get(self, key: str) -> str | None:
        return cast("str | None", await self.client.get(key))

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        await self.client.set(key, value, ex=ttl)

    async def cache_embedding(self, text_hash: str, vector: list[float], ttl: int = 3600) -> None:
        await self.set(f"emb:{text_hash}", json.dumps(vector), ttl=ttl)

    async def get_cached_embedding(self, text_hash: str) -> list[float] | None:
        raw = await self.get(f"emb:{text_hash}")
        if raw is not None:
            return json.loads(raw)
        return None
