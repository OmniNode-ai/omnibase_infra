# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Prometheus metrics handler - EFFECT handler exposing /metrics HTTP endpoint.

This module provides an EFFECT handler that exposes Prometheus metrics via an
HTTP endpoint for scraping. It follows the ONEX handler pattern with contract-
driven lifecycle management.

Architecture Principle: "Handlers own lifecycle, sinks own hot path"
    - Handler: Contract-driven lifecycle, expose /metrics HTTP endpoint
    - Sink: Fast in-process emission (SinkMetricsPrometheus)

Supported Operations:
    - metrics.scrape: Return current metrics in Prometheus text format
    - metrics.push: Push metrics to Prometheus Pushgateway (if configured)

HTTP Server:
    The handler starts an aiohttp-based HTTP server during initialize() that
    serves metrics at the configured path (default: /metrics on port 9090).
    The server is non-blocking and runs in the background.

    **SECURE BY DEFAULT**: The server binds to localhost (127.0.0.1) by default.
    External access requires explicit configuration AND reverse proxy protection.

Thread-Safety:
    The handler is thread-safe. The underlying SinkMetricsPrometheus uses
    thread-safe metric caches with locking. The aiohttp server handles
    concurrent requests safely.

Security Model:
    **Secure Defaults:**
    - Binds to localhost (127.0.0.1) by default - NOT accessible externally
    - Request body size limited to 1MB (configurable via max_request_size_bytes)
    - Request timeout of 30 seconds (configurable via request_timeout_seconds)
    - Error responses use generic messages - no stack traces or internal details

    **Production Deployment Requirements:**

    To expose metrics externally, you MUST:

    1. **Explicit Configuration**: Set host="0.0.0.0" explicitly (a warning will
       be logged when binding to non-localhost addresses).

    2. **Reverse Proxy**: Deploy behind nginx/traefik/envoy with:
       - TLS termination (HTTPS only)
       - IP allowlisting for Prometheus scrapers
       - Rate limiting to prevent DoS (NOT implemented in this handler)
       - Authentication (mTLS or bearer tokens recommended)

    3. **Network Isolation**: Consider using internal network CIDRs for
       container/pod networks instead of 0.0.0.0.

    **Built-in Protections:**
    - Input Validation: X-Correlation-ID headers validated as UUID format
    - Error Sanitization: 500/503 errors return generic messages only
    - Request Size Limits: Configurable max_request_size_bytes (default 1MB)
    - Request Timeouts: Configurable request_timeout_seconds (default 30s)

    **NOT Implemented (Requires Reverse Proxy):**
    - Rate limiting: Use nginx/traefik/envoy limit_req/rate_limit
    - TLS/HTTPS: Use reverse proxy for TLS termination
    - Authentication: Use reverse proxy for mTLS or token auth

    **Metric Labels**: Ensure metric labels do not contain:
    - PII (personally identifiable information)
    - Secrets or credentials
    - Internal hostnames/IPs (if sensitive)

    **Threat Model:**
    - DoS via metric scraping: Mitigate with rate limiting at proxy
    - Information disclosure: Metrics may reveal system topology
    - Header injection: X-Correlation-ID is validated and sanitized
    - Request body attacks: Limited by max_request_size_bytes

Usage:
    >>> from omnibase_infra.observability.handlers import HandlerMetricsPrometheus
    >>>
    >>> # Secure default: localhost only
    >>> handler = HandlerMetricsPrometheus()
    >>> await handler.initialize({"port": 9090, "path": "/metrics"})
    >>> # Metrics available at http://127.0.0.1:9090/metrics (localhost only)
    >>>
    >>> # External access (REQUIRES reverse proxy protection)
    >>> handler = HandlerMetricsPrometheus()
    >>> await handler.initialize({
    ...     "host": "0.0.0.0",  # WARNING: Use only with reverse proxy
    ...     "port": 9090,
    ...     "path": "/metrics",
    ... })
    >>>
    >>> await handler.shutdown()

See Also:
    - SinkMetricsPrometheus: Hot-path metrics sink
    - ModelMetricsHandlerConfig: Configuration model
    - docs/patterns/observability_patterns.md: Full observability documentation
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from datetime import UTC, datetime
from uuid import UUID, uuid4

# Optional dependencies with graceful degradation
# These are checked during initialization with clear error messages
_PROMETHEUS_AVAILABLE: bool = False
_AIOHTTP_AVAILABLE: bool = False

try:
    from aiohttp import web

    _AIOHTTP_AVAILABLE = True
except ImportError:
    web = None  # type: ignore[assignment, misc]

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Histogram, generate_latest

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    # Provide stubs for type checking when prometheus_client is not installed
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    Histogram = None  # type: ignore[assignment, misc]

    def generate_latest() -> bytes:  # type: ignore[misc]
        """Stub for when prometheus_client is not installed."""
        raise ImportError("prometheus_client is required but not installed")


from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
    EnumResponseStatus,
)
from omnibase_infra.errors import (
    InfraTimeoutError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
    ProtocolConfigurationError,
    RuntimeHostError,
)
from omnibase_infra.mixins import MixinEnvelopeExtraction
from omnibase_infra.observability.handlers.model_metrics_handler_config import (
    ModelMetricsHandlerConfig,
)
from omnibase_infra.observability.handlers.model_metrics_handler_response import (
    ModelMetricsHandlerPayload,
    ModelMetricsHandlerResponse,
)

logger = logging.getLogger(__name__)

# Handler ID for ModelHandlerOutput
HANDLER_ID_METRICS: str = "metrics-prometheus-handler"

# Security: UUID validation regex for X-Correlation-ID header
# Matches standard UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
_UUID_REGEX: re.Pattern[str] = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

# Maximum length for correlation ID header to prevent memory issues
_MAX_CORRELATION_ID_LENGTH: int = 64

# Default timeout for metrics generation to prevent blocking (seconds)
# This is used when config is not yet loaded or as internal timeout
_DEFAULT_METRICS_GENERATION_TIMEOUT: float = 5.0

# Timeout for Pushgateway operations (seconds)
_PUSH_GATEWAY_TIMEOUT: float = 10.0

# Default maximum request body size (1MB) - used when config not loaded
_DEFAULT_MAX_REQUEST_SIZE_BYTES: int = 1048576

# Import scrape duration buckets from constants module (ONEX pattern: constants in constants_*.py)
# See constants_metrics.py for bucket configuration documentation.
from omnibase_infra.observability.constants_metrics import SCRAPE_DURATION_BUCKETS

_SCRAPE_DURATION_BUCKETS = SCRAPE_DURATION_BUCKETS

# Lazily initialized scrape duration histogram
# Initialized on first use to avoid import-time side effects when prometheus_client
# is not installed. The metric is observed AFTER generate_latest() returns, so the
# duration value is available in the NEXT scrape (avoiding chicken-and-egg recursion).
_scrape_duration_histogram: Histogram | None = None

# Lock for thread-safe lazy initialization of the scrape duration histogram.
# Prevents race conditions where multiple threads could create duplicate histograms.
_histogram_init_lock: threading.Lock = threading.Lock()


def _get_scrape_duration_histogram() -> Histogram | None:
    """Get or create the scrape duration histogram.

    Lazily initializes the histogram on first call to avoid import-time errors
    when prometheus_client is not installed.

    Thread Safety:
        Uses double-checked locking pattern to ensure thread-safe initialization.
        The lock is only acquired when the histogram needs to be created, minimizing
        contention after initial setup.

    Returns:
        Histogram metric for scrape duration, or None if prometheus_client unavailable.
    """
    global _scrape_duration_histogram  # noqa: PLW0603 - Module-level metric cache

    if not _PROMETHEUS_AVAILABLE or Histogram is None:
        return None

    # Double-checked locking pattern for thread-safe lazy initialization.
    # First check without lock (fast path for already-initialized case).
    if _scrape_duration_histogram is None:
        with _histogram_init_lock:
            # Second check with lock (handles race condition where multiple threads
            # passed the first check before one acquired the lock and initialized).
            if _scrape_duration_histogram is None:
                _scrape_duration_histogram = Histogram(
                    "prometheus_handler_scrape_duration_seconds",
                    "Time spent generating Prometheus metrics for scrape requests",
                    buckets=_SCRAPE_DURATION_BUCKETS,
                )

    return _scrape_duration_histogram


SUPPORTED_OPERATIONS: frozenset[str] = frozenset(
    {
        "metrics.scrape",
        "metrics.push",
    }
)


class HandlerMetricsPrometheus(MixinEnvelopeExtraction):
    """Prometheus metrics EFFECT handler exposing HTTP /metrics endpoint.

    This handler implements the ONEX handler protocol for exposing Prometheus
    metrics via an HTTP endpoint. It manages the lifecycle of an aiohttp HTTP
    server that serves metrics in Prometheus text exposition format.

    Handler Classification:
        - handler_type: INFRA_HANDLER (protocol/transport handler)
        - handler_category: EFFECT (side-effecting I/O - HTTP server)

    Lifecycle:
        - initialize(): Validates config, starts HTTP server
        - execute(): Returns metrics text or pushes to Pushgateway
        - shutdown(): Gracefully stops HTTP server

    HTTP Server Integration:
        The handler uses aiohttp for async HTTP serving. The server runs in the
        background without blocking the event loop. Multiple concurrent scrapes
        are handled safely by aiohttp's request handling.

    Push Gateway Support:
        When push_gateway_url is configured, the handler can push metrics to a
        Prometheus Pushgateway. This is useful for short-lived batch jobs that
        may not live long enough to be scraped.

    Security:
        **SECURE BY DEFAULT**: The server binds to localhost (127.0.0.1) by default.

        Built-in protections:
        - Localhost binding by default (external access requires explicit config)
        - Request size limits (configurable, default 1MB)
        - Request timeouts (configurable, default 30s)
        - Generic error messages (no stack traces or internal details)
        - Correlation ID validation (UUID format enforced)

        For external access, you MUST:
        1. Explicitly set host="0.0.0.0" (a warning will be logged)
        2. Deploy behind a reverse proxy with TLS, auth, and rate limiting

        See module docstring for complete security model.

    Attributes:
        _config: Handler configuration.
        _initialized: Whether the handler has been initialized.
        _server: aiohttp web server instance.
        _runner: aiohttp application runner.
        _site: aiohttp TCP site for serving requests.

    Example:
        >>> # Secure default: localhost only
        >>> handler = HandlerMetricsPrometheus()
        >>> await handler.initialize({"port": 9090})
        >>> # Metrics at http://127.0.0.1:9090/metrics (localhost only)
        >>>
        >>> # External access (REQUIRES reverse proxy)
        >>> handler = HandlerMetricsPrometheus()
        >>> await handler.initialize({
        ...     "host": "0.0.0.0",  # WARNING: Use only with reverse proxy
        ...     "port": 9090,
        ... })
        >>> await handler.shutdown()
    """

    def __init__(self) -> None:
        """Initialize HandlerMetricsPrometheus in uninitialized state.

        The handler is not ready for use until initialize() is called with
        a valid configuration dictionary.
        """
        self._config: ModelMetricsHandlerConfig | None = None
        self._initialized: bool = False
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role of this handler.

        Returns:
            EnumHandlerType.INFRA_HANDLER - This handler is an infrastructure
            protocol/transport handler that manages an HTTP server for metrics
            exposition.

        Note:
            handler_type determines lifecycle, protocol selection, and runtime
            invocation patterns. It answers "what is this handler in the architecture?"

        See Also:
            - handler_category: Behavioral classification (EFFECT/COMPUTE)
        """
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification of this handler.

        Returns:
            EnumHandlerTypeCategory.EFFECT - This handler performs side-effecting
            I/O operations (HTTP server, network I/O). EFFECT handlers are not
            deterministic and interact with external systems.

        Note:
            handler_category determines security rules, determinism guarantees,
            replay safety, and permissions. It answers "how does this handler
            behave at runtime?"

        See Also:
            - handler_type: Architectural role (INFRA_HANDLER)
        """
        return EnumHandlerTypeCategory.EFFECT

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize the metrics handler with configuration.

        Validates the configuration and starts the HTTP server for metric
        scraping if enable_server is True.

        Security Note:
            The default bind address is "127.0.0.1" (localhost only).
            To expose externally, explicitly set host="0.0.0.0" AND deploy
            behind a reverse proxy with authentication, TLS, and rate limiting.

        Args:
            config: Configuration dict containing:
                - host: Bind address (default: "127.0.0.1" - localhost only)
                - port: Port number (default: 9090)
                - path: Metrics endpoint path (default: "/metrics")
                - push_gateway_url: Optional Pushgateway URL
                - enable_server: Whether to start HTTP server (default: True)
                - job_name: Job name for Pushgateway (default: "onex_metrics")
                - shutdown_timeout_seconds: Shutdown timeout (default: 5.0)
                - max_request_size_bytes: Max request body size (default: 1MB)
                - request_timeout_seconds: Request processing timeout (default: 30s)

        Raises:
            ProtocolConfigurationError: If configuration validation fails.
            RuntimeHostError: If HTTP server fails to start.

        Example:
            >>> handler = HandlerMetricsPrometheus()
            >>> await handler.initialize({"port": 9091})
            >>> # Server bound to 127.0.0.1:9091 (localhost only - secure)
        """
        init_correlation_id = uuid4()

        # Validate required dependencies before proceeding
        missing_deps: list[str] = []
        if not _PROMETHEUS_AVAILABLE:
            missing_deps.append("prometheus_client")
        if not _AIOHTTP_AVAILABLE:
            missing_deps.append("aiohttp")

        if missing_deps:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=init_correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="initialize",
                target_name="metrics_prometheus_handler",
            )
            raise ProtocolConfigurationError(
                f"Missing required dependencies: {', '.join(missing_deps)}. "
                f"Install with: pip install {' '.join(missing_deps)}",
                context=context,
            )

        logger.info(
            "Initializing %s",
            self.__class__.__name__,
            extra={
                "handler": self.__class__.__name__,
                "correlation_id": str(init_correlation_id),
            },
        )

        # Validate and parse configuration
        # NOTE: Broad Exception catch is intentional here because Pydantic can raise
        # various exception types (ValidationError, TypeError, ValueError) depending
        # on the validation failure. We wrap all in ProtocolConfigurationError.
        try:
            self._config = ModelMetricsHandlerConfig(**config)
        except Exception as e:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=init_correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="initialize",
                target_name="metrics_prometheus_handler",
            )
            raise ProtocolConfigurationError(
                f"Invalid metrics handler configuration: {e}",
                context=context,
            ) from e

        # Start HTTP server if enabled
        if self._config.enable_server:
            await self._start_http_server(init_correlation_id)

        self._initialized = True

        logger.info(
            "HandlerMetricsPrometheus initialized successfully",
            extra={
                "correlation_id": str(init_correlation_id),
                "host": self._config.host,
                "port": self._config.port,
                "path": self._config.path,
                "server_enabled": self._config.enable_server,
                "push_gateway_configured": self._config.push_gateway_url is not None,
            },
        )

    async def _start_http_server(self, correlation_id: UUID) -> None:
        """Start the aiohttp HTTP server for metrics exposition.

        Creates an aiohttp web application with a single route for metrics
        and starts it as a background TCP site.

        Args:
            correlation_id: Correlation ID for tracing.

        Raises:
            RuntimeHostError: If server fails to start.
        """
        if self._config is None:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="start_http_server",
                target_name="metrics_prometheus_handler",
            )
            raise RuntimeHostError(
                "Configuration not set before starting HTTP server",
                context=context,
            )

        try:
            # Create aiohttp application with security limits
            # client_max_size limits the maximum request body size
            self._app = web.Application(
                client_max_size=self._config.max_request_size_bytes,
            )
            self._app.router.add_get(self._config.path, self._handle_metrics_request)

            # Create runner and site
            self._runner = web.AppRunner(self._app)
            await self._runner.setup()

            self._site = web.TCPSite(
                self._runner,
                self._config.host,
                self._config.port,
            )
            await self._site.start()

            # Log security-relevant configuration
            is_localhost = self._config.host in ("127.0.0.1", "localhost", "::1")
            if not is_localhost:
                logger.warning(
                    "HTTP metrics server binding to non-localhost address - "
                    "ensure reverse proxy with auth/TLS is in place",
                    extra={
                        "correlation_id": str(correlation_id),
                        "host": self._config.host,
                        "port": self._config.port,
                    },
                )

            logger.info(
                "HTTP metrics server started",
                extra={
                    "correlation_id": str(correlation_id),
                    "host": self._config.host,
                    "port": self._config.port,
                    "path": self._config.path,
                    "max_request_size_bytes": self._config.max_request_size_bytes,
                    "request_timeout_seconds": self._config.request_timeout_seconds,
                    "localhost_only": is_localhost,
                },
            )

        except OSError as e:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="start_http_server",
                target_name="metrics_prometheus_handler",
            )
            raise RuntimeHostError(
                f"Failed to start HTTP server on {self._config.host}:{self._config.port}: {e}",
                context=context,
            ) from e

    def _parse_correlation_id_header(self, header_value: str | None) -> UUID:
        """Parse and validate X-Correlation-ID header value with security guards.

        Safely extracts a correlation ID from the request header. If the header
        is missing, empty, or contains an invalid UUID format, generates a new
        UUID and logs a warning for invalid values.

        Security Measures:
            - Length check: Headers exceeding 64 chars are rejected before parsing
            - Format validation: Regex validates UUID format before UUID() parsing
            - No reflection: Invalid values are NOT echoed back in responses
            - Truncated logging: Invalid values logged with max 36 chars (UUID length)

        Args:
            header_value: The raw X-Correlation-ID header value, or None if absent.

        Returns:
            A valid UUID - either parsed from the header or newly generated.

        Note:
            Invalid correlation IDs are handled gracefully to avoid crashing
            the metrics endpoint. A warning is logged to help identify
            misconfigured clients, but the invalid value is NOT exposed in
            HTTP responses to prevent header injection attacks.
        """
        if not header_value:
            return uuid4()

        # Security: Reject excessively long headers before any processing
        if len(header_value) > _MAX_CORRELATION_ID_LENGTH:
            fallback_id = uuid4()
            logger.warning(
                "X-Correlation-ID header exceeds maximum length, using generated UUID",
                extra={
                    "header_length": len(header_value),
                    "max_length": _MAX_CORRELATION_ID_LENGTH,
                    "generated_correlation_id": str(fallback_id),
                },
            )
            return fallback_id

        # Security: Validate UUID format with regex before parsing
        # This prevents malformed input from reaching UUID() constructor
        if not _UUID_REGEX.match(header_value):
            fallback_id = uuid4()
            logger.warning(
                "Invalid X-Correlation-ID header format, using generated UUID",
                extra={
                    # Truncate to UUID length (36 chars) max for safe logging
                    "invalid_header_preview": header_value[:36],
                    "generated_correlation_id": str(fallback_id),
                },
            )
            return fallback_id

        try:
            return UUID(header_value)
        except (ValueError, AttributeError):
            # Fallback for any edge cases not caught by regex
            fallback_id = uuid4()
            logger.warning(
                "X-Correlation-ID UUID parsing failed, using generated UUID",
                extra={
                    "generated_correlation_id": str(fallback_id),
                },
            )
            return fallback_id

    async def _handle_metrics_request(self, request: web.Request) -> web.Response:
        """Handle HTTP GET requests to the metrics endpoint.

        Generates Prometheus metrics in text exposition format and returns
        them with the appropriate content type.

        Security:
            - Correlation IDs are validated before use (see _parse_correlation_id_header)
            - Error messages are sanitized - no internal exception details exposed
            - Timeout protection prevents DoS via slow metric generation

        Args:
            request: aiohttp Request object.

        Returns:
            aiohttp Response with metrics text, or generic error on failure.
        """
        # Get and validate correlation ID from request headers
        correlation_id = self._parse_correlation_id_header(
            request.headers.get("X-Correlation-ID")
        )

        logger.debug(
            "Handling metrics scrape request",
            extra={
                "correlation_id": str(correlation_id),
                "remote": str(request.remote),
            },
        )

        # Start timing for scrape duration metric
        start_time = time.perf_counter()

        # Get timeout from config if available, otherwise use default
        metrics_timeout = (
            self._config.request_timeout_seconds
            if self._config
            else _DEFAULT_METRICS_GENERATION_TIMEOUT
        )

        try:
            # Generate metrics with timeout protection to prevent DoS
            # generate_latest() is synchronous, so run in executor with timeout
            loop = asyncio.get_running_loop()
            metrics_bytes = await asyncio.wait_for(
                loop.run_in_executor(None, generate_latest),
                timeout=metrics_timeout,
            )

            # Record scrape duration after generation completes
            # The duration is observed AFTER generate_latest() returns, so it will
            # be available in the NEXT scrape (avoiding chicken-and-egg recursion)
            duration_seconds = time.perf_counter() - start_time
            histogram = _get_scrape_duration_histogram()
            if histogram is not None:
                histogram.observe(duration_seconds)

            logger.debug(
                "Metrics scrape completed",
                extra={
                    "correlation_id": str(correlation_id),
                    "duration_seconds": duration_seconds,
                    "metrics_size_bytes": len(metrics_bytes),
                },
            )

            # Use headers dict for Content-Type because CONTENT_TYPE_LATEST includes
            # charset which conflicts with aiohttp's content_type parameter validation
            return web.Response(
                body=metrics_bytes,
                headers={
                    "Content-Type": CONTENT_TYPE_LATEST,
                    "X-Correlation-ID": str(correlation_id),
                },
            )

        except TimeoutError:
            # Record duration even on timeout (will be at or near timeout threshold)
            duration_seconds = time.perf_counter() - start_time
            histogram = _get_scrape_duration_histogram()
            if histogram is not None:
                histogram.observe(duration_seconds)

            # Log timeout with full details, but return generic message
            # Using warning level since timeout is expected behavior (DoS protection)
            logger.warning(
                "Metrics generation timed out",
                extra={
                    "correlation_id": str(correlation_id),
                    "timeout_seconds": metrics_timeout,
                    "duration_seconds": duration_seconds,
                },
            )
            # Security: Generic error message - no internal details exposed
            return web.Response(
                text="Service temporarily unavailable",
                status=503,  # Service Unavailable for timeout
                headers={"X-Correlation-ID": str(correlation_id)},
            )

        except Exception:
            # NOTE: Broad Exception catch is intentional here to ensure observability
            # (duration recording) even for unexpected errors, while returning a
            # generic 500 response that prevents internal details from leaking.
            # The exception is fully logged internally for debugging.
            #
            # Record duration even on error for observability
            duration_seconds = time.perf_counter() - start_time
            histogram = _get_scrape_duration_histogram()
            if histogram is not None:
                histogram.observe(duration_seconds)

            # Log full exception details internally for debugging
            logger.exception(
                "Failed to generate metrics",
                extra={
                    "correlation_id": str(correlation_id),
                    "duration_seconds": duration_seconds,
                },
            )
            # Security: Generic error message - no exception type or details exposed
            # This prevents information leakage about internal implementation
            return web.Response(
                text="Internal server error generating metrics",
                status=500,
                headers={"X-Correlation-ID": str(correlation_id)},
            )

    async def shutdown(self) -> None:
        """Shutdown the HTTP server and release resources.

        Gracefully stops the HTTP server, waiting for pending requests
        to complete within the configured timeout. The shutdown process:

        1. Stops accepting new connections (site.stop())
        2. Waits for pending requests to drain (runner.cleanup())
        3. Releases all resources

        If cleanup times out, pending requests are forcibly terminated.
        """
        shutdown_correlation_id = uuid4()

        logger.info(
            "Shutting down HandlerMetricsPrometheus",
            extra={"correlation_id": str(shutdown_correlation_id)},
        )

        timeout = self._config.shutdown_timeout_seconds if self._config else 5.0

        # Phase 1: Stop accepting new connections
        # This allows pending requests to complete before we close
        if self._site is not None:
            try:
                await self._site.stop()
                logger.debug(
                    "HTTP server stopped accepting new connections",
                    extra={"correlation_id": str(shutdown_correlation_id)},
                )
            except OSError as e:
                # OSError can occur if socket is already closed
                logger.warning(
                    "Error stopping HTTP site: %s",
                    e,
                    extra={
                        "correlation_id": str(shutdown_correlation_id),
                        "error_type": type(e).__name__,
                    },
                )
            finally:
                self._site = None

        # Phase 2: Drain pending requests and cleanup runner
        # This gives in-flight requests time to complete
        if self._runner is not None:
            try:
                await asyncio.wait_for(
                    self._runner.cleanup(),
                    timeout=timeout,
                )
                logger.debug(
                    "HTTP server cleanup completed successfully",
                    extra={"correlation_id": str(shutdown_correlation_id)},
                )
            except TimeoutError:
                # Timeout means pending requests couldn't complete in time
                # This is a warning because it may indicate slow consumers
                logger.warning(
                    "HTTP server cleanup timed out after %.1f seconds - "
                    "pending requests may have been forcibly terminated",
                    timeout,
                    extra={
                        "correlation_id": str(shutdown_correlation_id),
                        "timeout_seconds": timeout,
                    },
                )
            except OSError as e:
                # OSError can occur during cleanup if connections are in bad state
                logger.warning(
                    "OSError during HTTP server cleanup: %s",
                    e,
                    extra={
                        "correlation_id": str(shutdown_correlation_id),
                        "error_type": type(e).__name__,
                    },
                )
            finally:
                self._runner = None

        self._app = None
        self._initialized = False
        self._config = None

        logger.info(
            "HandlerMetricsPrometheus shutdown complete",
            extra={"correlation_id": str(shutdown_correlation_id)},
        )

    def _build_response(
        self,
        payload: ModelMetricsHandlerPayload,
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelMetricsHandlerResponse]:
        """Build standardized response wrapped in ModelHandlerOutput.

        Args:
            payload: Operation-specific response payload.
            correlation_id: Correlation ID for tracing.
            input_envelope_id: Input envelope ID for causality tracking.

        Returns:
            ModelHandlerOutput wrapping the metrics handler response.
        """
        response = ModelMetricsHandlerResponse(
            status=EnumResponseStatus.SUCCESS,
            payload=payload,
            correlation_id=correlation_id,
        )
        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_METRICS,
            result=response,
        )

    async def execute(
        self, envelope: dict[str, object]
    ) -> ModelHandlerOutput[ModelMetricsHandlerResponse]:
        """Execute a metrics operation from envelope.

        Args:
            envelope: Request envelope containing:
                - operation: "metrics.scrape" or "metrics.push"
                - payload: dict with operation-specific parameters (optional)
                - correlation_id: Optional correlation ID for tracing
                - envelope_id: Optional envelope ID for causality tracking

        Returns:
            ModelHandlerOutput wrapping the operation result.

        Raises:
            RuntimeHostError: If handler not initialized or invalid operation.

        Example:
            >>> result = await handler.execute({
            ...     "operation": "metrics.scrape",
            ...     "correlation_id": str(uuid4()),
            ... })
        """
        correlation_id = self._extract_correlation_id(envelope)
        input_envelope_id = self._extract_envelope_id(envelope)

        if not self._initialized:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="execute",
                target_name="metrics_prometheus_handler",
            )
            raise RuntimeHostError(
                "Metrics handler not initialized. Call initialize() first.",
                context=context,
            )

        operation = envelope.get("operation")
        if not isinstance(operation, str):
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="execute",
                target_name="metrics_prometheus_handler",
            )
            raise RuntimeHostError(
                "Missing or invalid 'operation' in envelope",
                context=context,
            )

        if operation not in SUPPORTED_OPERATIONS:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name="metrics_prometheus_handler",
            )
            raise RuntimeHostError(
                f"Operation '{operation}' not supported. "
                f"Available: {', '.join(sorted(SUPPORTED_OPERATIONS))}",
                context=context,
            )

        # Route to appropriate handler
        if operation == "metrics.scrape":
            return await self._handle_scrape(correlation_id, input_envelope_id)
        else:  # metrics.push
            return await self._handle_push(envelope, correlation_id, input_envelope_id)

    async def _handle_scrape(
        self,
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelMetricsHandlerResponse]:
        """Handle metrics.scrape operation.

        Generates Prometheus metrics from the default registry and returns
        them in text exposition format.

        Args:
            correlation_id: Correlation ID for tracing.
            input_envelope_id: Input envelope ID for causality tracking.

        Returns:
            ModelHandlerOutput with metrics text in payload.
        """
        logger.debug(
            "Executing metrics.scrape operation",
            extra={"correlation_id": str(correlation_id)},
        )

        # Get timeout from config if available, otherwise use default
        scrape_timeout = (
            self._config.request_timeout_seconds
            if self._config
            else _DEFAULT_METRICS_GENERATION_TIMEOUT
        )

        # Generate metrics from default Prometheus registry
        # generate_latest() is synchronous, so run in executor with timeout
        # to prevent blocking the event loop
        try:
            loop = asyncio.get_running_loop()
            metrics_bytes = await asyncio.wait_for(
                loop.run_in_executor(None, generate_latest),
                timeout=scrape_timeout,
            )
        except TimeoutError:
            context = ModelTimeoutErrorContext(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="metrics.scrape",
                target_name="metrics_prometheus_handler",
                timeout_seconds=scrape_timeout,
            )
            raise InfraTimeoutError(
                f"Metrics generation timed out after {scrape_timeout}s",
                context=context,
            ) from None

        metrics_text = metrics_bytes.decode("utf-8")

        payload = ModelMetricsHandlerPayload(
            operation_type="metrics.scrape",
            metrics_text=metrics_text,
            content_type=CONTENT_TYPE_LATEST,
        )

        return self._build_response(payload, correlation_id, input_envelope_id)

    async def _handle_push(
        self,
        envelope: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelMetricsHandlerResponse]:
        """Handle metrics.push operation.

        Pushes current metrics to the configured Prometheus Pushgateway.
        Requires push_gateway_url to be configured.

        Args:
            envelope: Request envelope (may contain override configuration).
            correlation_id: Correlation ID for tracing.
            input_envelope_id: Input envelope ID for causality tracking.

        Returns:
            ModelHandlerOutput with push confirmation in payload.

        Raises:
            RuntimeHostError: If Pushgateway is not configured.
        """
        if self._config is None or self._config.push_gateway_url is None:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="metrics.push",
                target_name="metrics_prometheus_handler",
            )
            raise RuntimeHostError(
                "Pushgateway not configured. Set push_gateway_url in config.",
                context=context,
            )

        # Capture config values in local variables for lambda closure
        # This ensures type safety and avoids mypy errors with Optional access
        push_gateway_url: str = self._config.push_gateway_url
        job_name: str = self._config.job_name

        logger.debug(
            "Executing metrics.push operation",
            extra={
                "correlation_id": str(correlation_id),
                "push_gateway_url": push_gateway_url,
            },
        )

        # Use prometheus_client's push_to_gateway functionality
        # Note: This is a synchronous operation, so we run it in executor with timeout
        # NOTE: Broad Exception catch is intentional here because push_to_gateway
        # can raise various exceptions: URLError for network issues, HTTPError for
        # gateway errors, and other unexpected exceptions from the prometheus_client
        # library. We wrap all failures in RuntimeHostError for consistent handling.
        try:
            from prometheus_client import REGISTRY, push_to_gateway

            loop = asyncio.get_running_loop()
            await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: push_to_gateway(
                        push_gateway_url,
                        job=job_name,
                        registry=REGISTRY,
                    ),
                ),
                timeout=_PUSH_GATEWAY_TIMEOUT,
            )

            pushed_at = datetime.now(UTC).isoformat()

            payload = ModelMetricsHandlerPayload(
                operation_type="metrics.push",
                pushed_at=pushed_at,
                push_gateway_url=push_gateway_url,
                job_name=job_name,
            )

            logger.info(
                "Metrics pushed to Pushgateway",
                extra={
                    "correlation_id": str(correlation_id),
                    "push_gateway_url": push_gateway_url,
                    "pushed_at": pushed_at,
                },
            )

            return self._build_response(payload, correlation_id, input_envelope_id)

        except TimeoutError:
            timeout_ctx = ModelTimeoutErrorContext(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="metrics.push",
                target_name="metrics_prometheus_handler",
                timeout_seconds=_PUSH_GATEWAY_TIMEOUT,
            )
            raise InfraTimeoutError(
                f"Pushgateway request timed out after {_PUSH_GATEWAY_TIMEOUT}s",
                context=timeout_ctx,
            ) from None

        except Exception as e:
            # NOTE: Broad Exception catch is intentional here because httpx and
            # network operations can raise many exception types (ConnectionError,
            # ProtocolError, etc.). We wrap all in RuntimeHostError with correlation
            # context while preserving the error chain for debugging.
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.HTTP,
                operation="metrics.push",
                target_name="metrics_prometheus_handler",
            )
            raise RuntimeHostError(
                f"Failed to push metrics to Pushgateway: {type(e).__name__}",
                context=context,
            ) from e

    def describe(self) -> dict[str, object]:
        """Return handler metadata and capabilities for introspection.

        Returns:
            dict containing handler type, category, operations, and status.
        """
        return {
            "handler_type": self.handler_type.value,
            "handler_category": self.handler_category.value,
            "supported_operations": sorted(SUPPORTED_OPERATIONS),
            "initialized": self._initialized,
            "server_enabled": (self._config.enable_server if self._config else False),
            "host": self._config.host if self._config else None,
            "port": self._config.port if self._config else None,
            "path": self._config.path if self._config else None,
            "push_gateway_configured": (
                self._config.push_gateway_url is not None if self._config else False
            ),
            "version": "0.1.0",
        }


__all__: list[str] = ["HandlerMetricsPrometheus", "HANDLER_ID_METRICS"]
