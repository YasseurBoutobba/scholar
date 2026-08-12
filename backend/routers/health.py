import time

from fastapi import APIRouter
from sqlalchemy import text

from backend.models.schemas import HealthResponse, ServiceHealth
from backend.repositories.qdrant_repository import QdrantRepository
from backend.repositories.redis_repository import RedisRepository

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health():
    services: dict[str, ServiceHealth] = {}

    t0 = time.monotonic()
    try:
        repo = QdrantRepository()
        ok = repo.health_check()
        services["qdrant"] = ServiceHealth(
            status="healthy" if ok else "unhealthy",
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        )
    except Exception as e:
        services["qdrant"] = ServiceHealth(
            status="unhealthy",
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
            error=str(e),
        )

    t0 = time.monotonic()
    try:
        redis = RedisRepository()
        ok = await redis.ping()
        services["redis"] = ServiceHealth(
            status="healthy" if ok else "unhealthy",
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        )
    except Exception as e:
        services["redis"] = ServiceHealth(
            status="unhealthy",
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
            error=str(e),
        )

    t0 = time.monotonic()
    try:
        from backend.models.base import engine

        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        services["postgres"] = ServiceHealth(
            status="healthy",
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        )
    except Exception as e:
        services["postgres"] = ServiceHealth(
            status="unhealthy",
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
            error=str(e),
        )

    overall = "healthy" if all(s.status == "healthy" for s in services.values()) else "degraded"
    return HealthResponse(status=overall, services=services)
