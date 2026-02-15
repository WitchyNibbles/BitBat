"""FastAPI application factory for the BitBat REST API."""

from __future__ import annotations

from fastapi import FastAPI

from bitbat.api.routes import analytics, health, metrics, predictions


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

    app.include_router(health.router)
    app.include_router(predictions.router)
    app.include_router(analytics.router)
    app.include_router(metrics.router)

    return app


# Module-level app instance for ``uvicorn bitbat.api.app:app``
app = create_app()
