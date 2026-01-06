# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""HTTP Health Service for ONEX Runtime.

This module provides a minimal HTTP service for exposing health check endpoints.
It is designed to run alongside the ONEX runtime kernel to satisfy Docker/K8s
health check requirements.

The service exposes:
    - GET /health: Returns runtime health status as JSON
    - GET /ready: Returns readiness status as JSON (alias for /health)

Configuration:
    ONEX_HTTP_PORT: Port to listen on (default: 8085)

Example:
    >>> from omnibase_infra.runtime.service_health import ServiceHealth
    >>> from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
    >>>
    >>> async def main():
    ...     runtime = RuntimeHostProcess()
    ...     server = ServiceHealth(runtime=runtime, port=8085)
    ...     await server.start()
    ...     # Server is now running
    ...     await server.stop()

Note:
    This service uses aiohttp for async HTTP handling, which is already a
    dependency of omnibase_infra for other infrastructure operations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from aiohttp import web

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
from omnibase_infra.runtime.models.model_health_check_response import (
    ModelHealthCheckResponse,
)
from omnibase_infra.utils.correlation import generate_correlation_id

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

    from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess

logger = logging.getLogger(__name__)

# Default configuration - hardcoded to avoid import-time crashes from invalid env vars
# Environment variable override is handled safely in ServiceHealth.__init__
DEFAULT_HTTP_PORT: int = 8085
DEFAULT_HTTP_HOST = "0.0.0.0"  # noqa: S104 - Required for container networking


def _get_port_from_env(default: int) -> int:
    """Safely parse ONEX_HTTP_PORT from environment with fallback to default.

    This function handles invalid environment variable values gracefully by
    logging a warning and returning the default value, rather than raising
    an exception. This prevents import-time crashes and allows the application
    to start even with misconfigured environment variables.

    Args:
        default: The fallback port value if env var is unset or invalid.

    Returns:
        Parsed port value if valid and within range (1-65535), otherwise default.
    """
    from omnibase_infra.errors import ProtocolConfigurationError
    from omnibase_infra.utils.util_env_parsing import parse_env_int

    try:
        return parse_env_int(
            "ONEX_HTTP_PORT",
            default,
            min_value=1,
            max_value=65535,
            transport_type=EnumInfraTransportType.HTTP,
            service_name="health_server",
        )
    except ProtocolConfigurationError as e:
        logger.warning(
            "Invalid ONEX_HTTP_PORT environment variable, using default %d: %s",
            default,
            e,
        )
        return default


class ServiceHealth:
    """Minimal HTTP server for health check endpoints.

    This server provides health check endpoints for Docker and Kubernetes
    liveness/readiness probes. It delegates health status to the RuntimeHostProcess.

    Attributes:
        runtime: The RuntimeHostProcess instance to query for health status
        port: Port to listen on
        host: Host to bind to
        version: Runtime version string to include in health response
        container: Optional ONEX dependency injection container

    Container Integration (OMN-529):
        ServiceHealth supports two initialization modes:

        1. Direct runtime injection (original pattern):
            >>> server = ServiceHealth(runtime=runtime, port=8085)

        2. Container-based injection (ONEX-compliant):
            >>> server = ServiceHealth(container=container, runtime=runtime)
            >>> # Or use the async factory method:
            >>> server = await ServiceHealth.create_from_container(container)

    Example:
        >>> server = ServiceHealth(runtime=runtime, port=8085)
        >>> await server.start()
        >>> # curl http://localhost:8085/health
        >>> await server.stop()
    """

    def __init__(
        self,
        container: ModelONEXContainer | None = None,
        runtime: RuntimeHostProcess | None = None,
        port: int | None = None,
        host: str = DEFAULT_HTTP_HOST,
        version: str = "unknown",
    ) -> None:
        """Initialize the health server.

        Args:
            container: Optional ONEX dependency injection container.
            runtime: RuntimeHostProcess instance to delegate health checks to.
                     Required if container is not provided.
            port: Port to listen on. If None, uses ONEX_HTTP_PORT env var or 8085.
            host: Host to bind to (default: 0.0.0.0 for container networking).
            version: Runtime version string for health response.

        Raises:
            ValueError: If neither container nor runtime is provided.
        """
        # Store container for ONEX compliance (OMN-529)
        self._container: ModelONEXContainer | None = container

        # Validate that at least one dependency source is provided
        if container is None and runtime is None:
            raise ValueError(
                "ServiceHealth requires either 'container' or 'runtime' to be provided. "
                "Use ServiceHealth(runtime=runtime) or ServiceHealth(container=container)."
            )

        self._runtime: RuntimeHostProcess | None = runtime
        # If port is explicitly provided, use it; otherwise parse from env var safely
        self._port: int = (
            port if port is not None else _get_port_from_env(DEFAULT_HTTP_PORT)
        )
        self._host: str = host
        self._version: str = version

        # Server state
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._is_running: bool = False

        logger.debug(
            "ServiceHealth initialized",
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

    @property
    def container(self) -> ModelONEXContainer | None:
        """Return the optional ONEX dependency injection container.

        Returns:
            The stored ModelONEXContainer instance, or None if not provided.
        """
        return self._container

    @property
    def runtime(self) -> RuntimeHostProcess:
        """Return the RuntimeHostProcess instance.

        Returns:
            The RuntimeHostProcess used for health checks.

        Raises:
            RuntimeError: If runtime was not provided during initialization
                and has not been resolved from the container.
        """
        if self._runtime is None:
            raise RuntimeError(
                "RuntimeHostProcess not available. "
                "Either provide runtime during __init__ or use create_from_container()."
            )
        return self._runtime

    @classmethod
    async def create_from_container(
        cls,
        container: ModelONEXContainer,
        port: int | None = None,
        host: str = DEFAULT_HTTP_HOST,
        version: str = "unknown",
    ) -> ServiceHealth:
        """Create a ServiceHealth by resolving RuntimeHostProcess from container.

        This is the preferred ONEX-compliant way to create a ServiceHealth when
        using container-based dependency injection.

        Args:
            container: ONEX dependency injection container.
            port: Port to listen on. If None, uses ONEX_HTTP_PORT env var or 8085.
            host: Host to bind to (default: 0.0.0.0 for container networking).
            version: Runtime version string for health response.

        Returns:
            Initialized ServiceHealth with runtime resolved from container.

        Example:
            >>> container = ModelONEXContainer()
            >>> await wire_infrastructure_services(container)
            >>> server = await ServiceHealth.create_from_container(container)
            >>> await server.start()
        """
        from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess

        runtime = await container.service_registry.resolve_service(RuntimeHostProcess)
        return cls(
            container=container,
            runtime=runtime,
            port=port,
            host=host,
            version=version,
        )

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
            >>> server = ServiceHealth(runtime=runtime, port=8085)
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
            logger.debug("ServiceHealth already started, skipping")
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
                "ServiceHealth started (correlation_id=%s)",
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
            >>> server = ServiceHealth(runtime=runtime, port=8085)
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
            logger.debug("ServiceHealth already stopped, skipping")
            return

        correlation_id = generate_correlation_id()
        logger.info(
            "Stopping ServiceHealth (correlation_id=%s)",
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
            "ServiceHealth stopped successfully (correlation_id=%s)",
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
                you can override this behavior by subclassing ServiceHealth and modifying
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
                    "is_running": bool,
                    "is_draining": bool,  // True during graceful shutdown drain
                    "pending_message_count": int,  // In-flight messages
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
            health_details = await self.runtime.health_check()

            # Runtime type validation: health_check() returns dict per contract
            # This helps static analysis and provides runtime validation
            # NOTE: Use explicit if/raise instead of assert - assertions can be
            # disabled with Python's -O flag, which would skip this safety check
            if not isinstance(health_details, dict):
                raise TypeError(
                    f"health_check() must return dict, got {type(health_details).__name__}"
                )

            # Determine overall status based on health check results
            is_healthy = bool(health_details.get("healthy", False))
            is_degraded = bool(health_details.get("degraded", False))

            if is_healthy:
                status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
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
                #   - Subclass ServiceHealth and override _handle_health()
                #   - Configure load balancer to inspect response body "status" field
                #   - Change http_status below to 503 if restart-on-degrade is preferred
                status = "degraded"
                http_status = 200
            else:
                status = "unhealthy"
                http_status = 503

            response = ModelHealthCheckResponse.success(
                status=status,
                version=self._version,
                details=health_details,
            )

            return web.Response(
                text=response.model_dump_json(exclude_none=True),
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

            error_response = ModelHealthCheckResponse.failure(
                version=self._version,
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=str(correlation_id),
            )

            return web.Response(
                text=error_response.model_dump_json(exclude_none=True),
                status=503,
                content_type="application/json",
            )


__all__: list[str] = ["DEFAULT_HTTP_HOST", "DEFAULT_HTTP_PORT", "ServiceHealth"]
