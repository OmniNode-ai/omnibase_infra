# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""HTTP Health Server for ONEX Runtime.

This module provides a minimal HTTP server for exposing health check endpoints.
It is designed to run alongside the ONEX runtime kernel to satisfy Docker/K8s
health check requirements.

The server exposes:
    - GET /health: Returns runtime health status as JSON
    - GET /ready: Returns readiness status as JSON (alias for /health)

Configuration:
    ONEX_HTTP_PORT: Port to listen on (default: 8085)

Example:
    >>> from omnibase_infra.runtime.health_server import HealthServer
    >>> from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
    >>>
    >>> async def main():
    ...     runtime = RuntimeHostProcess()
    ...     server = HealthServer(runtime=runtime, port=8085)
    ...     await server.start()
    ...     # Server is now running
    ...     await server.stop()

Note:
    This server uses aiohttp for async HTTP handling, which is already a
    dependency of omnibase_infra for other infrastructure operations.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING
from uuid import uuid4

from aiohttp import web

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError

if TYPE_CHECKING:
    from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_HTTP_PORT = 8085
DEFAULT_HTTP_HOST = "0.0.0.0"  # noqa: S104 - Required for container networking


class HealthServer:
    """Minimal HTTP server for health check endpoints.

    This server provides health check endpoints for Docker and Kubernetes
    liveness/readiness probes. It delegates health status to the RuntimeHostProcess.

    Attributes:
        runtime: The RuntimeHostProcess instance to query for health status
        port: Port to listen on
        host: Host to bind to
        version: Runtime version string to include in health response

    Example:
        >>> server = HealthServer(runtime=runtime, port=8085)
        >>> await server.start()
        >>> # curl http://localhost:8085/health
        >>> await server.stop()
    """

    def __init__(
        self,
        runtime: RuntimeHostProcess,
        port: int = DEFAULT_HTTP_PORT,
        host: str = DEFAULT_HTTP_HOST,
        version: str = "unknown",
    ) -> None:
        """Initialize the health server.

        Args:
            runtime: RuntimeHostProcess instance to delegate health checks to.
            port: Port to listen on (default: 8085).
            host: Host to bind to (default: 0.0.0.0 for container networking).
            version: Runtime version string for health response.
        """
        self._runtime: RuntimeHostProcess = runtime
        self._port: int = port
        self._host: str = host
        self._version: str = version

        # Server state
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._is_running: bool = False

        logger.debug(
            "HealthServer initialized",
            extra={
                "port": self._port,
                "host": self._host,
                "version": self._version,
            },
        )

    @property
    def is_running(self) -> bool:
        """Return True if the health server is running.

        Returns:
            Boolean indicating whether the server is running.
        """
        return self._is_running

    @property
    def port(self) -> int:
        """Return the configured port.

        Returns:
            The port number the server listens on.
        """
        return self._port

    async def start(self) -> None:
        """Start the HTTP health server.

        Creates an aiohttp web application with health endpoints and starts
        listening on the configured host and port.

        This method is idempotent - calling start() on an already started
        server is safe and has no effect.

        Raises:
            RuntimeHostError: If server fails to start due to port binding
                or other network errors.
        """
        if self._is_running:
            logger.debug("HealthServer already started, skipping")
            return

        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.HTTP,
            operation="start_health_server",
            target_name=f"{self._host}:{self._port}",
            correlation_id=correlation_id,
        )

        try:
            # Create aiohttp application
            self._app = web.Application()

            # Register routes
            self._app.router.add_get("/health", self._handle_health)
            self._app.router.add_get("/ready", self._handle_health)  # Alias

            # Create and start runner
            self._runner = web.AppRunner(self._app)
            await self._runner.setup()

            # Create site and start listening
            self._site = web.TCPSite(
                self._runner,
                self._host,
                self._port,
            )
            await self._site.start()

            self._is_running = True

            logger.info(
                "HealthServer started",
                extra={
                    "host": self._host,
                    "port": self._port,
                    "endpoints": ["/health", "/ready"],
                },
            )

        except OSError as e:
            # Port binding failure (e.g., address already in use)
            raise RuntimeHostError(
                f"Failed to start health server on {self._host}:{self._port}: {e}",
                context=context,
            ) from e

        except Exception as e:
            # Unexpected error during server startup
            raise RuntimeHostError(
                f"Unexpected error starting health server: {e}",
                context=context,
            ) from e

    async def stop(self) -> None:
        """Stop the HTTP health server.

        Gracefully shuts down the aiohttp web server and releases resources.

        This method is idempotent - calling stop() on an already stopped
        server is safe and has no effect.
        """
        if not self._is_running:
            logger.debug("HealthServer already stopped, skipping")
            return

        logger.info("Stopping HealthServer")

        # Cleanup in reverse order of creation
        if self._site is not None:
            await self._site.stop()
            self._site = None

        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

        self._app = None
        self._is_running = False

        logger.info("HealthServer stopped successfully")

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle GET /health and GET /ready requests.

        Delegates to RuntimeHostProcess.health_check() for actual health status
        and returns a JSON response with status information.

        Args:
            request: The incoming HTTP request.

        Returns:
            JSON response with health status. HTTP 200 if healthy,
            HTTP 503 if unhealthy or degraded.

        Response Format:
            {
                "status": "healthy" | "degraded" | "unhealthy",
                "version": "x.y.z",
                "details": { ... }  // Full health check details
            }
        """
        # Suppress unused argument warning - aiohttp handler signature requires request
        _ = request

        try:
            # Get health status from runtime
            health_details = await self._runtime.health_check()

            # Determine overall status
            is_healthy = bool(health_details.get("healthy", False))
            is_degraded = bool(health_details.get("degraded", False))

            if is_healthy:
                status = "healthy"
                http_status = 200
            elif is_degraded:
                # Degraded means core is running but some handlers failed.
                # Return 200 so Docker/K8s considers container healthy.
                # The "degraded" status in response body indicates partial functionality.
                status = "degraded"
                http_status = 200
            else:
                status = "unhealthy"
                http_status = 503

            response_body = {
                "status": status,
                "version": self._version,
                "details": health_details,
            }

            return web.Response(
                text=json.dumps(response_body),
                status=http_status,
                content_type="application/json",
            )

        except Exception as e:
            # Health check itself failed
            correlation_id = uuid4()
            logger.exception(
                "Health check failed with exception",
                extra={
                    "error": str(e),
                    "correlation_id": str(correlation_id),
                },
            )

            error_response = {
                "status": "unhealthy",
                "version": self._version,
                "error": str(e),
                "correlation_id": str(correlation_id),
            }

            return web.Response(
                text=json.dumps(error_response),
                status=503,
                content_type="application/json",
            )


__all__: list[str] = ["HealthServer", "DEFAULT_HTTP_PORT", "DEFAULT_HTTP_HOST"]
