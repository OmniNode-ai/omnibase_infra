# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry API FastAPI Application.

Creates and configures the FastAPI application for the Registry API.
Provides factory function for flexible instantiation with different
backend configurations.

Usage:
    # Create app with container (required)
    from omnibase_core.container import ModelONEXContainer

    container = ModelONEXContainer()
    app = create_app(container=container, cors_origins=["http://localhost:3000"])

    # Create app with full backends
    app = create_app(
        container=container,
        projection_reader=reader,
        consul_handler=handler,
        cors_origins=["http://localhost:3000"],
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

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
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
    service: ServiceRegistryDiscovery | None = getattr(
        app.state, "registry_service", None
    )
    if service is not None:
        logger.info(
            "Registry service configured",
            extra={
                "has_projection_reader": service.has_projection_reader,
                "has_consul_handler": service.has_consul_handler,
            },
        )
    else:
        logger.warning("Registry service not configured - API will return limited data")

    yield

    logger.info("Registry API shutting down")

    # Cleanup Consul handler if we own it
    if service is not None and service.consul_handler is not None:
        try:
            await service.consul_handler.shutdown()
            logger.info("Consul handler shutdown complete")
        except Exception as e:
            logger.exception(
                "Error during Consul handler shutdown",
                extra={"error_type": type(e).__name__},
            )


def create_app(
    container: ModelONEXContainer,
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
        container: ONEX container for dependency injection. Required for
            ONEX DI pattern compliance.
        projection_reader: Optional projection reader for node registrations.
        consul_handler: Optional Consul handler for live instances.
        widget_mapping_path: Optional path to widget mapping YAML.
        cors_origins: Optional list of allowed CORS origins.
            If not provided, reads from CORS_ORIGINS environment variable.
            Raises ProtocolConfigurationError if neither is configured (fail-fast).

    Returns:
        Configured FastAPI application.

    Raises:
        ProtocolConfigurationError: If CORS origins not configured via parameter
            or CORS_ORIGINS environment variable.

    Example:
        >>> from omnibase_infra.services.registry_api import create_app
        >>> from omnibase_core.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> app = create_app(container=container)
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

    # Configure CORS - fail-fast if not configured
    if cors_origins is not None:
        origins = cors_origins
    else:
        env_origins = os.environ.get("CORS_ORIGINS")
        if env_origins is None:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation="configure_cors",
            )
            raise ProtocolConfigurationError(
                "CORS_ORIGINS must be configured. "
                "Set the CORS_ORIGINS environment variable (comma-separated list of allowed origins) "
                "or pass cors_origins parameter to create_app(). "
                "Example: CORS_ORIGINS=http://localhost:3000,https://dashboard.example.com",
                context=context,
            )
        origins = env_origins.split(",")

    # Warn about wildcard CORS - only when explicitly configured
    if "*" in origins:
        logger.warning(
            "CORS explicitly configured with wildcard origin '*'. "
            "This is acceptable for development but should be restricted in production.",
            extra={"origins": origins},
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Create and attach service
    service = ServiceRegistryDiscovery(
        container=container,
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


# Module-level app instance is not supported since container is required.
# For production usage, use create_app() with proper configuration:
#
# Example:
#     from omnibase_core.container import ModelONEXContainer
#     from omnibase_infra.services.registry_api import create_app
#
#     container = ModelONEXContainer()
#     app = create_app(
#         container=container,
#         projection_reader=reader,
#         consul_handler=handler,
#         cors_origins=["http://localhost:3000"],
#     )
#     uvicorn.run(app, host="0.0.0.0", port=8000)
#
# For uvicorn CLI usage, create a launcher module that instantiates the container.
app: FastAPI | None = None


__all__ = ["app", "create_app"]
