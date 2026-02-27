"""FastAPI application factory for the BitBat REST API."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bitbat.api.routes import analytics, health, metrics, predictions, system


def create_app() -> FastAPI:
    """Build and return the FastAPI application with all routers mounted."""
    app = FastAPI(
        title="BitBat API",
        description="Bitcoin price prediction REST API — serves predictions, "
        "performance metrics, and system health information.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(predictions.router)
    app.include_router(analytics.router)
    app.include_router(metrics.router)
    app.include_router(system.router)

    return app


# Module-level app instance for ``uvicorn bitbat.api.app:app``
app = create_app()
