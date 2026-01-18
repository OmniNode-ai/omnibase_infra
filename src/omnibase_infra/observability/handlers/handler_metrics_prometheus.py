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

Thread-Safety:
    The handler is thread-safe. The underlying SinkMetricsPrometheus uses
    thread-safe metric caches with locking. The aiohttp server handles
    concurrent requests safely.

Usage:
    >>> from omnibase_infra.observability.handlers import HandlerMetricsPrometheus
    >>>
    >>> handler = HandlerMetricsPrometheus()
    >>> await handler.initialize({"port": 9090, "path": "/metrics"})
    >>>
    >>> # Handler exposes metrics at http://localhost:9090/metrics
    >>> # Prometheus can scrape this endpoint
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
from datetime import UTC, datetime
from uuid import UUID, uuid4

from aiohttp import web
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
    EnumResponseStatus,
)
from omnibase_infra.errors import (
    ModelInfraErrorContext,
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

    Security Considerations:
        - The /metrics endpoint should be secured in production environments
        - Consider using reverse proxy with authentication for public exposure
        - No sensitive data should be included in metric labels

    Attributes:
        _config: Handler configuration.
        _initialized: Whether the handler has been initialized.
        _server: aiohttp web server instance.
        _runner: aiohttp application runner.
        _site: aiohttp TCP site for serving requests.

    Example:
        >>> handler = HandlerMetricsPrometheus()
        >>> await handler.initialize({
        ...     "host": "0.0.0.0",
        ...     "port": 9090,
        ...     "path": "/metrics",
        ... })
        >>> # Metrics available at http://localhost:9090/metrics
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

        Args:
            config: Configuration dict containing:
                - host: Bind address (default: "0.0.0.0")
                - port: Port number (default: 9090)
                - path: Metrics endpoint path (default: "/metrics")
                - push_gateway_url: Optional Pushgateway URL
                - enable_server: Whether to start HTTP server (default: True)
                - job_name: Job name for Pushgateway (default: "onex_metrics")
                - shutdown_timeout_seconds: Shutdown timeout (default: 5.0)

        Raises:
            ProtocolConfigurationError: If configuration validation fails.
            RuntimeHostError: If HTTP server fails to start.

        Example:
            >>> handler = HandlerMetricsPrometheus()
            >>> await handler.initialize({"port": 9091})
        """
        init_correlation_id = uuid4()

        logger.info(
            "Initializing %s",
            self.__class__.__name__,
            extra={
                "handler": self.__class__.__name__,
                "correlation_id": str(init_correlation_id),
            },
        )

        # Validate and parse configuration
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
            # Create aiohttp application
            self._app = web.Application()
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

            logger.info(
                "HTTP metrics server started",
                extra={
                    "correlation_id": str(correlation_id),
                    "host": self._config.host,
                    "port": self._config.port,
                    "path": self._config.path,
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
        """Parse and validate X-Correlation-ID header value.

        Safely extracts a correlation ID from the request header. If the header
        is missing, empty, or contains an invalid UUID format, generates a new
        UUID and logs a warning for invalid values.

        Args:
            header_value: The raw X-Correlation-ID header value, or None if absent.

        Returns:
            A valid UUID - either parsed from the header or newly generated.

        Note:
            Invalid correlation IDs are handled gracefully to avoid crashing
            the metrics endpoint. A warning is logged to help identify
            misconfigured clients.
        """
        if not header_value:
            return uuid4()

        try:
            return UUID(header_value)
        except (ValueError, AttributeError):
            # Log warning for debugging but don't crash - generate fallback UUID
            fallback_id = uuid4()
            logger.warning(
                "Invalid X-Correlation-ID header format, using generated UUID",
                extra={
                    "invalid_correlation_id": header_value[:100],  # Truncate for safety
                    "generated_correlation_id": str(fallback_id),
                },
            )
            return fallback_id

    async def _handle_metrics_request(self, request: web.Request) -> web.Response:
        """Handle HTTP GET requests to the metrics endpoint.

        Generates Prometheus metrics in text exposition format and returns
        them with the appropriate content type.

        Args:
            request: aiohttp Request object.

        Returns:
            aiohttp Response with metrics text.
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

        try:
            # Generate metrics from default Prometheus registry
            metrics_bytes = generate_latest()

            # Use headers dict for Content-Type because CONTENT_TYPE_LATEST includes
            # charset which conflicts with aiohttp's content_type parameter validation
            return web.Response(
                body=metrics_bytes,
                headers={
                    "Content-Type": CONTENT_TYPE_LATEST,
                    "X-Correlation-ID": str(correlation_id),
                },
            )

        except Exception as e:
            logger.exception(
                "Failed to generate metrics",
                extra={"correlation_id": str(correlation_id)},
            )
            return web.Response(
                text=f"Error generating metrics: {type(e).__name__}",
                status=500,
                headers={"X-Correlation-ID": str(correlation_id)},
            )

    async def shutdown(self) -> None:
        """Shutdown the HTTP server and release resources.

        Gracefully stops the HTTP server, waiting for pending requests
        to complete within the configured timeout.
        """
        shutdown_correlation_id = uuid4()

        logger.info(
            "Shutting down HandlerMetricsPrometheus",
            extra={"correlation_id": str(shutdown_correlation_id)},
        )

        timeout = self._config.shutdown_timeout_seconds if self._config else 5.0

        # Stop the HTTP server gracefully
        if self._site is not None:
            await self._site.stop()
            self._site = None

        if self._runner is not None:
            try:
                await asyncio.wait_for(
                    self._runner.cleanup(),
                    timeout=timeout,
                )
            except TimeoutError:
                logger.warning(
                    "HTTP server cleanup timed out",
                    extra={
                        "correlation_id": str(shutdown_correlation_id),
                        "timeout_seconds": timeout,
                    },
                )
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

        # Generate metrics from default Prometheus registry
        metrics_bytes = generate_latest()
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
        # Note: This is a synchronous operation, so we run it in executor
        try:
            from prometheus_client import REGISTRY, push_to_gateway

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: push_to_gateway(
                    push_gateway_url,
                    job=job_name,
                    registry=REGISTRY,
                ),
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

        except Exception as e:
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
