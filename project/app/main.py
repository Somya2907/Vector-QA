"""
main.py

FastAPI application entry point.
Initialises settings, configures logging, wires up all pipeline components,
and mounts the API router.
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from src.utils.config import get_settings
from src.utils.logger import configure_logging, get_logger
from app.routes import router

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application.

    Wire up component dependencies here (embedder, vector store, retriever,
    generator, pipeline) and attach them to app.state so routes can access them.

    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Permission-Aware Vector QA API",
        version="0.1.0",
        description="Retrieval-augmented QA with fine-grained access control.",
    )

    # TODO: Initialise and attach pipeline components to app.state
    # app.state.pipeline = build_pipeline(settings)

    app.include_router(router, prefix="/api/v1")

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness probe — returns OK when the server is up."""
        return {"status": "ok"}

    logger.info("app_created", host=settings.api_host, port=settings.api_port)
    return app


app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, reload=True)
