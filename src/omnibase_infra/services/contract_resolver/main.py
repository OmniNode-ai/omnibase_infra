# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Contract Resolver Bridge FastAPI Application.

Creates and configures the FastAPI application for the Contract Resolver Bridge.
Provides a factory function for flexible instantiation in tests and production.

Usage:
    # Create app with container (required)
    from omnibase_core.container import ModelONEXContainer
    container = ModelONEXContainer()
    app = create_app(container=container, cors_origins=["http://localhost:3000"])

    # Run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8091)

Related Tickets:
    - OMN-2756: Phase 2 — Transitional node-shaped HTTP bridge for contract.resolve
    - OMN-2754: Phase 1 — contract.resolve compute node (NodeContractResolveCompute)
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.services.contract_resolver.routes import router

logger = logging.getLogger(__name__)

# API metadata
API_TITLE = "ONEX Contract Resolver Bridge"
API_DESCRIPTION = """
Contract Resolver Bridge — HTTP surface for NodeContractResolveCompute.

Provides synchronous HTTP access to the ONEX contract.resolve compute node
so the dashboard (Node.js/Express) can resolve overlaid contracts without a
Kafka round-trip.

## Routes

- **POST /api/nodes/contract.resolve** — Resolve a contract with ordered patches
- **GET /health** — Liveness check

## Design

This is a **transitional** bridge (OMN-2756). It wraps
``NodeContractResolveCompute.resolve()`` exactly — no bespoke API surface.
When the real ONEX node runner supports HTTP invocation, only the execution
path changes; the Express proxy code and contract registry entry are unchanged.

## Related Tickets

- OMN-2756: HTTP bridge implementation
- OMN-2754: NodeContractResolveCompute (compute node)
- OMN-2358: Dashboard Contract Builder epic
"""
API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler for startup/shutdown.

    Args:
        app: FastAPI application instance.

    Yields:
        None (context manager pattern).
    """
    logger.info(
        "Contract Resolver Bridge starting up",
        extra={"port": 8091, "version": API_VERSION},
    )
    yield
    logger.info("Contract Resolver Bridge shutting down")


def create_app(
    container: ModelONEXContainer,
    cors_origins: list[str] | None = None,
    event_bus: object | None = None,
) -> FastAPI:
    """Create and configure the Contract Resolver Bridge FastAPI application.

    Factory function that creates a FastAPI app. The event bus is optional —
    Kafka emission is fire-and-forget and degrades gracefully without it.

    Args:
        container: ONEX container for dependency injection. Required for
            ONEX DI pattern compliance.
        cors_origins: Optional list of allowed CORS origins. If not provided,
            reads from ``CORS_ORIGINS`` environment variable.
            Raises ``ProtocolConfigurationError`` if neither is configured.
        event_bus: Optional event bus for fire-and-forget Kafka emission.
            When present it must expose a ``publish(event)`` method.

    Returns:
        Configured FastAPI application.

    Raises:
        ProtocolConfigurationError: If CORS origins are not configured via
            parameter or ``CORS_ORIGINS`` environment variable.

    Example:
        >>> from omnibase_infra.services.contract_resolver import create_app
        >>> from omnibase_core.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> app = create_app(container=container, cors_origins=["http://localhost:3000"])
        >>> # Run with: uvicorn module:app --host 0.0.0.0 --port 8091
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

    # Configure CORS — fail-fast if not configured
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
                "Set the CORS_ORIGINS environment variable (comma-separated list of "
                "allowed origins) or pass cors_origins parameter to create_app(). "
                "Example: CORS_ORIGINS=http://localhost:3000,https://dashboard.example.com",
                context=context,
            )
        origins = [o.strip() for o in env_origins.split(",") if o.strip()]

    if "*" in origins:
        logger.warning(
            "CORS explicitly configured with wildcard origin '*'. "
            "Acceptable for development — restrict in production.",
            extra={"origins": origins},
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Attach optional event bus for fire-and-forget Kafka emission
    app.state.event_bus = event_bus
    app.state.container = container

    # Include routes
    app.include_router(router)

    # Root redirect to docs info
    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        """Root endpoint with service info."""
        return {
            "service": API_TITLE,
            "version": API_VERSION,
            "docs": "/docs",
            "health": "/health",
            "resolve": "/api/nodes/contract.resolve",
        }

    logger.info(
        "Contract Resolver Bridge created",
        extra={
            "version": API_VERSION,
            "cors_origins": origins,
            "event_bus_configured": event_bus is not None,
        },
    )

    return app


# Module-level app instance is not supported since container is required.
# For production usage, use create_app() with proper configuration:
#
# Example:
#     from omnibase_core.container import ModelONEXContainer
#     from omnibase_infra.services.contract_resolver import create_app
#
#     container = ModelONEXContainer()
#     app = create_app(
#         container=container,
#         cors_origins=["http://localhost:3000"],
#     )
#     uvicorn.run(app, host="0.0.0.0", port=8091)
app: FastAPI | None = None


__all__ = ["API_TITLE", "API_VERSION", "app", "create_app"]
