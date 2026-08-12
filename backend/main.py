import asyncio
import logging
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ExceptionHandler

from backend.config import settings
from backend.err_handlers import (
    app_error_handler,
    http_exception_handler,
    unhandled_exception_handler,
    validation_error_handler,
)
from backend.exceptions import AppError
from backend.models.base import Base, engine
from backend.repositories.qdrant_repository import QdrantRepository
from backend.routers import health, papers, query, research, runs
from backend.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    setup_logging(settings.LOG_LEVEL)

    try:
        qdrant = QdrantRepository()
        await asyncio.to_thread(qdrant.ensure_collection)
    except Exception as e:
        logger.warning(f"Qdrant not available at startup: {e}")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield

    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_exception_handler(Exception, unhandled_exception_handler)
    app.add_exception_handler(
        StarletteHTTPException, cast(ExceptionHandler, http_exception_handler)
    )
    app.add_exception_handler(
        RequestValidationError, cast(ExceptionHandler, validation_error_handler)
    )
    app.add_exception_handler(AppError, cast(ExceptionHandler, app_error_handler))

    app.include_router(health.router)
    app.include_router(papers.router, prefix="/api/v1")
    app.include_router(query.router, prefix="/api/v1")
    app.include_router(research.router, prefix="/api/v1")
    app.include_router(runs.router, prefix="/api/v1")

    return app


app = create_app()
