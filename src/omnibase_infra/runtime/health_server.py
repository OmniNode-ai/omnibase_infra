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
from typing import TYPE_CHECKING, Optional

from aiohttp import web

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
from omnibase_infra.utils.correlation import generate_correlation_id

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
        """Start the HTTP health server for Docker/Kubernetes probes.

        Creates an aiohttp web application with health check endpoints and starts
        listening on the configured host and port. The server exposes standardized
        health check endpoints that integrate with container orchestration platforms.

        Startup Process:
            1. Check if server is already running (idempotent safety check)
            2. Create aiohttp Application instance
            3. Register health check routes (/health, /ready)
            4. Initialize AppRunner and perform async setup
            5. Create TCPSite bound to configured host and port
            6. Start listening for incoming health check requests
            7. Mark server as running and log startup with correlation tracking

        Health Endpoints:
            - GET /health: Primary health check endpoint
            - GET /ready: Readiness probe (alias for /health)

        Both endpoints return JSON with:
            - status: "healthy" | "degraded" | "unhealthy"
            - version: Runtime kernel version
            - details: Full health check details from RuntimeHostProcess

        HTTP Status Codes:
            - 200: Healthy or degraded (container operational)
            - 503: Unhealthy (container should be restarted)

        This method is idempotent - calling start() on an already running
        server is safe and has no effect. This prevents double-start errors
        during rapid restart scenarios.

        Raises:
            RuntimeHostError: If server fails to start. Common causes include:
                - Port already in use (OSError with EADDRINUSE)
                - Permission denied on privileged port (OSError with EACCES)
                - Network interface unavailable
                - Unexpected aiohttp initialization errors

            All errors include:
                - correlation_id: UUID for distributed tracing
                - context: ModelInfraErrorContext with transport type, operation
                - Original exception chaining: via "from e" for root cause analysis

        Example:
            >>> server = HealthServer(runtime=runtime, port=8085)
            >>> await server.start()
            >>> # Server now listening at http://0.0.0.0:8085/health
            >>> # Docker can probe: curl http://localhost:8085/health

        Example Error (Port In Use):
            RuntimeHostError: Failed to start health server on 0.0.0.0:8085: [Errno 48] Address already in use
            (correlation_id: 123e4567-e89b-12d3-a456-426614174000)

        Docker Integration:
            HEALTHCHECK --interval=30s --timeout=3s \\
                CMD curl -f http://localhost:8085/health || exit 1
        """
        if self._is_running:
            logger.debug("HealthServer already started, skipping")
            return

        correlation_id = generate_correlation_id()
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
                "HealthServer started (correlation_id=%s)",
                correlation_id,
                extra={
                    "host": self._host,
                    "port": self._port,
                    "endpoints": ["/health", "/ready"],
                    "version": self._version,
                },
            )

        except OSError as e:
            # Port binding failure (e.g., address already in use, permission denied)
            error_msg = (
                f"Failed to start health server on {self._host}:{self._port}: {e}"
            )
            logger.exception(
                "%s (correlation_id=%s)",
                error_msg,
                correlation_id,
                extra={
                    "error_type": type(e).__name__,
                    "errno": e.errno if hasattr(e, "errno") else None,
                },
            )
            raise RuntimeHostError(
                error_msg,
                context=context,
            ) from e

        except Exception as e:
            # Unexpected error during server startup
            error_msg = f"Unexpected error starting health server: {e}"
            logger.exception(
                "%s (correlation_id=%s)",
                error_msg,
                correlation_id,
                extra={
                    "error_type": type(e).__name__,
                },
            )
            raise RuntimeHostError(
                error_msg,
                context=context,
            ) from e

    async def stop(self) -> None:
        """Stop the HTTP health server gracefully.

        Gracefully shuts down the aiohttp web server and releases all resources.
        The shutdown process ensures proper cleanup of network resources, active
        connections, and internal state.

        Shutdown Process:
            1. Check if server is already stopped (idempotent safety check)
            2. Stop TCPSite to reject new connections
            3. Clean up AppRunner to release resources
            4. Clear Application reference
            5. Mark server as not running
            6. Log successful shutdown with correlation tracking

        Resource Cleanup Order:
            The cleanup follows reverse initialization order to ensure proper
            resource release and prevent resource leaks:
            - TCPSite (network binding)
            - AppRunner (request handlers)
            - Application (route definitions)

        This method is idempotent - calling stop() on an already stopped
        server is safe and has no effect. This prevents double-stop errors
        during graceful shutdown scenarios.

        Cleanup Guarantees:
            - All network sockets are closed
            - Active HTTP connections are terminated gracefully
            - Event loop resources are released
            - Server state is reset for potential restart

        Example:
            >>> server = HealthServer(runtime=runtime, port=8085)
            >>> await server.start()
            >>> # ... runtime operation ...
            >>> await server.stop()
            >>> # Server no longer listening, resources released

        Exception Handling:
            This method does not raise exceptions. Any errors during cleanup
            are logged but do not prevent the shutdown sequence from completing.
            This ensures that stop() always succeeds and the server state is
            consistently marked as stopped.
        """
        if not self._is_running:
            logger.debug("HealthServer already stopped, skipping")
            return

        correlation_id = generate_correlation_id()
        logger.info(
            "Stopping HealthServer (correlation_id=%s)",
            correlation_id,
        )

        # Cleanup in reverse order of creation
        # Stop TCPSite first to reject new connections
        if self._site is not None:
            try:
                await self._site.stop()
            except Exception as e:
                logger.warning(
                    "Error stopping TCPSite during shutdown (correlation_id=%s)",
                    correlation_id,
                    extra={
                        "error_type": type(e).__name__,
                        "error": str(e),
                    },
                )
            self._site = None

        # Clean up AppRunner to release resources
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception as e:
                logger.warning(
                    "Error cleaning up AppRunner during shutdown (correlation_id=%s)",
                    correlation_id,
                    extra={
                        "error_type": type(e).__name__,
                        "error": str(e),
                    },
                )
            self._runner = None

        # Clear application reference
        self._app = None
        self._is_running = False

        logger.info(
            "HealthServer stopped successfully (correlation_id=%s)",
            correlation_id,
        )

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle GET /health and GET /ready requests.

        This is the main health check endpoint handler for Docker/Kubernetes
        health probes. It delegates to RuntimeHostProcess.health_check() for
        actual health status determination and returns a standardized JSON
        response with status information and diagnostics.

        Health Status Logic:
            1. Query RuntimeHostProcess for current health state
            2. Analyze health details to determine overall status
            3. Map status to appropriate HTTP status code
            4. Construct JSON response with version and diagnostics
            5. Return response to health probe client

        Status Determination:
            - healthy: All components operational, return HTTP 200
            - degraded: Core running but some handlers failed, return HTTP 200
            - unhealthy: Critical failure, return HTTP 503

        Degraded State HTTP 200 Design Decision:
            Degraded containers intentionally return HTTP 200 to keep them in service
            rotation. This is a deliberate design choice that prioritizes investigation
            over automatic restarts.

            Rationale:
                1. Automatic restarts may mask recurring issues that need investigation
                2. Reduced functionality is often preferable to no functionality
                3. Cascading failures can occur if multiple containers restart simultaneously
                4. Operators can monitor degraded status via metrics/alerts and investigate

            Alternative Considered:
                Returning HTTP 503 would remove degraded containers from load balancer
                rotation while keeping liveness probes passing. This was rejected because
                it reduces capacity during partial outages when some functionality may
                still be valuable to users.

            Customization:
                If your deployment requires removing degraded containers from rotation,
                you can override this behavior by subclassing HealthServer and modifying
                the _handle_health method, or configure your load balancer to inspect
                the response body "status" field instead of relying solely on HTTP codes.

        Args:
            request: The incoming aiohttp HTTP request. This parameter is required
                by the aiohttp handler signature but is intentionally unused in this
                implementation as health checks do not require request data.

        Returns:
            JSON response with health status information. The HTTP status code
            indicates container health to orchestration platforms:
                - HTTP 200: Container is healthy or degraded (operational)
                - HTTP 503: Container is unhealthy (restart recommended)

        Response Format (Success):
            {
                "status": "healthy" | "degraded" | "unhealthy",
                "version": "x.y.z",
                "details": {
                    "healthy": bool,
                    "degraded": bool,
                    "runtime_active": bool,
                    "handlers": {...},
                    // Additional health check details
                }
            }

        Response Format (Error):
            {
                "status": "unhealthy",
                "version": "x.y.z",
                "error": "Exception message",
                "correlation_id": "uuid-for-tracing"
            }

        Docker Integration Example:
            HEALTHCHECK --interval=30s --timeout=3s --retries=3 \\
                CMD curl -f http://localhost:8085/health || exit 1

        Kubernetes Integration Example:
            livenessProbe:
              httpGet:
                path: /health
                port: 8085
              initialDelaySeconds: 30
              periodSeconds: 10

        Exception Handling:
            If health_check() raises an exception, the handler:
            1. Logs the full exception with correlation_id for tracing
            2. Returns HTTP 503 with error details
            3. Includes correlation_id in response for debugging
            This ensures health probes always receive a response even during
            runtime failures, preventing indefinite probe hangs.
        """
        # Suppress unused argument warning - aiohttp handler signature requires request
        _ = request

        try:
            # Get health status from runtime
            health_details = await self._runtime.health_check()

            # Determine overall status based on health check results
            is_healthy = bool(health_details.get("healthy", False))
            is_degraded = bool(health_details.get("degraded", False))

            if is_healthy:
                status = "healthy"
                http_status = 200
            elif is_degraded:
                # DESIGN DECISION: Degraded status returns HTTP 200 (not 503)
                #
                # Rationale: Degraded containers remain in service rotation to allow
                # operators to investigate issues without triggering automatic restarts.
                # The "degraded" status in the response body indicates reduced functionality
                # while keeping the container operational for Docker/Kubernetes probes.
                #
                # Why HTTP 200 instead of 503:
                #   1. Prevents cascading failures if multiple containers degrade together
                #   2. Reduced functionality is often better than no functionality
                #   3. Automatic restarts may mask recurring issues needing investigation
                #   4. Operators can monitor "degraded" status via metrics/alerts
                #
                # Alternative considered: HTTP 503 would remove degraded containers from
                # load balancer rotation while keeping liveness probes passing. Rejected
                # because it reduces capacity during partial outages when degraded
                # containers may still serve valuable traffic.
                #
                # Customization: To remove degraded containers from rotation, either:
                #   - Subclass HealthServer and override _handle_health()
                #   - Configure load balancer to inspect response body "status" field
                #   - Change http_status below to 503 if restart-on-degrade is preferred
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
            # Health check itself failed - generate correlation_id for tracing
            correlation_id = generate_correlation_id()
            logger.exception(
                "Health check failed with exception (correlation_id=%s)",
                correlation_id,
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

            error_response = {
                "status": "unhealthy",
                "version": self._version,
                "error": str(e),
                "error_type": type(e).__name__,
                "correlation_id": str(correlation_id),
            }

            return web.Response(
                text=json.dumps(error_response),
                status=503,
                content_type="application/json",
            )


__all__: list[str] = ["HealthServer", "DEFAULT_HTTP_PORT", "DEFAULT_HTTP_HOST"]
