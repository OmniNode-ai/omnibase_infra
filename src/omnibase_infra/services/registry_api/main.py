# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry API FastAPI Application.

Creates and configures the FastAPI application for the Registry API.
Provides factory function for flexible instantiation with different
backend configurations.

Usage:
    # Create app with default settings (no backends)
    app = create_app()

    # Create app with full backends
    app = create_app(
        projection_reader=reader,
        consul_handler=handler,
    )

    # Run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

Related Tickets:
    - OMN-1278: Contract-Driven Dashboard - Registry Discovery
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from omnibase_infra.services.registry_api.routes import router
from omnibase_infra.services.registry_api.service import ServiceRegistryDiscovery

if TYPE_CHECKING:
    from omnibase_infra.handlers.service_discovery import HandlerServiceDiscoveryConsul
    from omnibase_infra.projectors import ProjectionReaderRegistration

logger = logging.getLogger(__name__)

# API metadata
API_TITLE = "ONEX Registry API"
API_DESCRIPTION = """
Registry Discovery API for ONEX Dashboard Integration.

This API provides access to node registrations and live service instances
for dashboard consumption. It combines data from:

- **PostgreSQL Projections**: Node registration state, capabilities, and metadata
- **Consul Service Discovery**: Live service instances with health status

## Key Features

- **Full Dashboard Payload**: Single endpoint for all dashboard data
- **Partial Success**: Returns data even when one backend fails
- **Widget Mapping**: Configuration for capability-to-widget rendering
- **Health Monitoring**: Component-level health status

## Related Tickets

- OMN-1278: Contract-Driven Dashboard - Registry Discovery
"""
API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler for startup/shutdown.

    Initializes backend connections on startup and cleans up on shutdown.

    Args:
        app: FastAPI application instance.

    Yields:
        None (context manager pattern).
    """
    logger.info("Registry API starting up")

    # Log configuration
    service = getattr(app.state, "registry_service", None)
    if service is not None:
        logger.info(
            "Registry service configured",
            extra={
                "has_projection_reader": service._projection_reader is not None,
                "has_consul_handler": service._consul_handler is not None,
            },
        )
    else:
        logger.warning("Registry service not configured - API will return limited data")

    yield

    logger.info("Registry API shutting down")

    # Cleanup Consul handler if we own it
    if service is not None and service._consul_handler is not None:
        try:
            await service._consul_handler.shutdown()
            logger.info("Consul handler shutdown complete")
        except Exception:
            logger.exception("Error during Consul handler shutdown")


def create_app(
    projection_reader: ProjectionReaderRegistration | None = None,
    consul_handler: HandlerServiceDiscoveryConsul | None = None,
    widget_mapping_path: Path | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create and configure the Registry API FastAPI application.

    Factory function that creates a FastAPI app with the specified
    backend configurations. All backends are optional - the API will
    return partial data with warnings when backends are unavailable.

    Args:
        projection_reader: Optional projection reader for node registrations.
        consul_handler: Optional Consul handler for live instances.
        widget_mapping_path: Optional path to widget mapping YAML.
        cors_origins: Optional list of allowed CORS origins.
            Defaults to ["*"] for development, should be restricted in production.

    Returns:
        Configured FastAPI application.

    Example:
        >>> from omnibase_infra.services.registry_api import create_app
        >>> app = create_app()
        >>> # Run with: uvicorn module:app --host 0.0.0.0 --port 8000
    """
    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Configure CORS
    origins = cors_origins or os.environ.get("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Create and attach service
    service = ServiceRegistryDiscovery(
        projection_reader=projection_reader,
        consul_handler=consul_handler,
        widget_mapping_path=widget_mapping_path,
    )
    app.state.registry_service = service

    # Include routes
    app.include_router(router)

    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        """Root endpoint with API info."""
        return {
            "service": API_TITLE,
            "version": API_VERSION,
            "docs": "/docs",
            "health": "/registry/health",
        }

    logger.info(
        "Registry API created",
        extra={
            "version": API_VERSION,
            "cors_origins": origins,
        },
    )

    return app


# Default app instance for direct uvicorn usage
# Example: uvicorn omnibase_infra.services.registry_api.main:app --host 0.0.0.0 --port 8000
app = create_app()


__all__ = ["app", "create_app"]
