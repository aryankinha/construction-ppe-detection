from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import configure_root_logger, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_root_logger(settings.LOG_LEVEL)
    logger.info("Starting PPE Detection API (env=%s)", settings.APP_ENV)

    from app.core.detector import PPEDetector
    app.state.detector = PPEDetector(settings.MODEL_PATH, settings.DETECTION_CONFIDENCE)
    logger.info("YOLO model loaded from %s", settings.MODEL_PATH)

    yield

    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="PPE Detection",
        description="Webcam + image PPE detection",
        version="3.0.0",
        lifespan=lifespan,
    )

    origins_raw = settings.CORS_ORIGINS.strip()
    if origins_raw == "*" or not origins_raw:
        allow_origins = ["*"]
    else:
        allow_origins = [o.strip() for o in origins_raw.split(",") if o.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from app.api.routes.health import router as health_router
    from app.api.routes.detect import router as detect_router

    app.include_router(health_router, prefix="/api/v1")
    app.include_router(detect_router, prefix="/api/v1")

    @app.get("/")
    async def root():
        return {"service": "ppe-detection", "docs": "/docs"}

    return app


app = create_app()
