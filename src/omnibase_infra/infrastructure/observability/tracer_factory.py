"""
Tracer Factory for ONEX Infrastructure.

Provides OpenTelemetry tracer configuration and setup for distributed tracing
across infrastructure services. Includes auto-instrumentation for asyncpg, kafka,
and redis, along with factory methods for creating service-specific tracers.

Features:
- OpenTelemetry tracer provider configuration
- Service-specific tracer creation
- Automatic instrumentation for infrastructure dependencies
- Span context managers for operations
- Integration with omnibase_core telemetry models
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError

# OpenTelemetry imports with availability check
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.instrumentation.aiohttp_client import (
        AioHttpClientInstrumentor,
    )
    from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
    from opentelemetry.instrumentation.kafka import KafkaInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    # Stub types when OpenTelemetry is not available
    Tracer = Any  # type: ignore
    Span = Any  # type: ignore
    SpanKind = Any  # type: ignore


class TracerConfig:
    """Configuration for OpenTelemetry tracing."""

    def __init__(
        self,
        service_name: str = "omnibase_infrastructure",
        service_version: str = "1.0.0",
        environment: str | None = None,
        otlp_endpoint: str | None = None,
        enable_db_instrumentation: bool = True,
        enable_kafka_instrumentation: bool = True,
        enable_redis_instrumentation: bool = True,
        enable_http_instrumentation: bool = True,
    ):
        """
        Initialize tracer configuration.

        Args:
            service_name: Name of the service for tracing
            service_version: Version of the service
            environment: Deployment environment
            otlp_endpoint: OpenTelemetry collector endpoint
            enable_db_instrumentation: Enable automatic DB instrumentation
            enable_kafka_instrumentation: Enable automatic Kafka instrumentation
            enable_redis_instrumentation: Enable automatic Redis instrumentation
            enable_http_instrumentation: Enable automatic HTTP instrumentation
        """
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment or self._detect_environment()
        self.otlp_endpoint = otlp_endpoint or os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
        )

        # Feature flags
        self.enable_db_instrumentation = (
            os.getenv("OTEL_ENABLE_DB_INSTRUMENTATION", str(enable_db_instrumentation)).lower()
            == "true"
        )
        self.enable_kafka_instrumentation = (
            os.getenv(
                "OTEL_ENABLE_KAFKA_INSTRUMENTATION",
                str(enable_kafka_instrumentation),
            ).lower()
            == "true"
        )
        self.enable_redis_instrumentation = (
            os.getenv(
                "OTEL_ENABLE_REDIS_INSTRUMENTATION",
                str(enable_redis_instrumentation),
            ).lower()
            == "true"
        )
        self.enable_http_instrumentation = (
            os.getenv(
                "OTEL_ENABLE_HTTP_INSTRUMENTATION",
                str(enable_http_instrumentation),
            ).lower()
            == "true"
        )

    def _detect_environment(self) -> str:
        """Detect current deployment environment."""
        env_vars = ["ENVIRONMENT", "ENV", "DEPLOYMENT_ENV", "NODE_ENV"]
        for var in env_vars:
            value = os.getenv(var)
            if value:
                return value.lower()
        return "development"


class NoOpTracer:
    """No-op tracer when OpenTelemetry is not available."""

    def start_span(self, name: str, **kwargs: Any) -> "NoOpSpan":
        """Create a no-op span."""
        return NoOpSpan()


class NoOpSpan:
    """No-op span when OpenTelemetry is not available."""

    def set_attribute(self, key: str, value: Any) -> None:
        """No-op set attribute."""
        pass

    def set_status(self, status: Any) -> None:
        """No-op set status."""
        pass

    def record_exception(self, exception: Exception) -> None:
        """No-op record exception."""
        pass

    def end(self) -> None:
        """No-op end span."""
        pass

    def __enter__(self) -> "NoOpSpan":
        """No-op context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """No-op context manager exit."""
        pass


class TracerFactory:
    """
    Factory for creating and managing OpenTelemetry tracers.

    Provides centralized tracer configuration, automatic instrumentation,
    and factory methods for creating service-specific tracers.
    """

    _instance: "TracerFactory | None" = None
    _is_initialized: bool = False
    _tracer_provider: Any = None  # TracerProvider
    _tracers: dict[str, Tracer] = {}

    def __init__(self, config: TracerConfig | None = None):
        """
        Initialize tracer factory.

        Args:
            config: Tracer configuration (auto-detected if None)
        """
        self.config = config or TracerConfig()

        if not OPENTELEMETRY_AVAILABLE:
            import logging

            logging.warning("OpenTelemetry not available - tracing will be disabled")

    @classmethod
    def get_instance(cls, config: TracerConfig | None = None) -> "TracerFactory":
        """
        Get singleton tracer factory instance.

        Args:
            config: Configuration for first initialization

        Returns:
            Singleton tracer factory instance
        """
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    async def initialize(self) -> None:
        """
        Initialize OpenTelemetry tracing infrastructure.

        Sets up tracer provider, OTLP exporter, and automatic instrumentation.

        Raises:
            OnexError: If initialization fails
        """
        if self._is_initialized or not OPENTELEMETRY_AVAILABLE:
            return

        try:
            # Create resource with service information
            resource = Resource.create(
                {
                    SERVICE_NAME: self.config.service_name,
                    SERVICE_VERSION: self.config.service_version,
                    "deployment.environment": self.config.environment,
                    "service.namespace": "omnibase_infrastructure",
                }
            )

            # Create tracer provider
            self._tracer_provider = TracerProvider(resource=resource)

            # Configure OTLP exporter
            otlp_exporter = OTLPSpanExporter(endpoint=self.config.otlp_endpoint)

            # Add batch span processor
            span_processor = BatchSpanProcessor(otlp_exporter)
            self._tracer_provider.add_span_processor(span_processor)

            # Set global tracer provider
            trace.set_tracer_provider(self._tracer_provider)

            # Initialize automatic instrumentation
            await self._initialize_instrumentation()

            self._is_initialized = True

        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.CONFIGURATION_ERROR,
                message=f"Failed to initialize distributed tracing: {e!s}",
            ) from e

    async def _initialize_instrumentation(self) -> None:
        """Initialize automatic instrumentation for infrastructure dependencies."""
        # PostgreSQL instrumentation
        if self.config.enable_db_instrumentation:
            AsyncPGInstrumentor().instrument()

        # Kafka instrumentation
        if self.config.enable_kafka_instrumentation:
            KafkaInstrumentor().instrument()

        # Redis instrumentation
        if self.config.enable_redis_instrumentation:
            RedisInstrumentor().instrument()

        # HTTP client instrumentation
        if self.config.enable_http_instrumentation:
            AioHttpClientInstrumentor().instrument()

    def get_tracer(
        self,
        instrumentation_scope: str,
        instrumentation_version: str | None = None,
    ) -> Tracer:
        """
        Get or create a tracer for a specific instrumentation scope.

        Args:
            instrumentation_scope: Scope name (typically module name)
            instrumentation_version: Version of the instrumentation

        Returns:
            OpenTelemetry tracer instance (or no-op if unavailable)
        """
        if not OPENTELEMETRY_AVAILABLE or not self._is_initialized:
            return NoOpTracer()  # type: ignore

        cache_key = f"{instrumentation_scope}:{instrumentation_version or ''}"
        if cache_key not in self._tracers:
            self._tracers[cache_key] = trace.get_tracer(
                instrumenting_module_name=instrumentation_scope,
                instrumenting_library_version=instrumentation_version
                or self.config.service_version,
            )

        return self._tracers[cache_key]

    @asynccontextmanager
    async def trace_operation(
        self,
        tracer: Tracer,
        operation_name: str,
        correlation_id: str | UUID | None = None,
        span_kind: Any = None,  # SpanKind
        attributes: dict[str, Any] | None = None,
    ) -> AsyncIterator[Span]:
        """
        Create a trace span for an operation with automatic error handling.

        Args:
            tracer: Tracer to use for creating the span
            operation_name: Name of the operation being traced
            correlation_id: Correlation ID for tracking
            span_kind: OpenTelemetry span kind (defaults to INTERNAL)
            attributes: Additional span attributes

        Yields:
            Active span for the operation

        Example:
            tracer = factory.get_tracer(__name__)
            async with factory.trace_operation(tracer, "process_event") as span:
                span.set_attribute("event_type", "user_login")
                # Process event
        """
        if not OPENTELEMETRY_AVAILABLE or not self._is_initialized:
            yield NoOpSpan()  # type: ignore
            return

        # Build span attributes
        span_attributes = {
            "environment": self.config.environment,
            "service.name": self.config.service_name,
        }

        if correlation_id:
            span_attributes["correlation_id"] = str(correlation_id)

        if attributes:
            span_attributes.update(attributes)

        # Determine span kind
        if span_kind is None and OPENTELEMETRY_AVAILABLE:
            span_kind = SpanKind.INTERNAL

        # Create and manage span
        span = tracer.start_span(name=operation_name, kind=span_kind, attributes=span_attributes)

        try:
            yield span
            # Mark span as successful
            span.set_status(Status(StatusCode.OK))

        except Exception as e:
            # Mark span as failed and record exception
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise

        finally:
            span.end()

    async def shutdown(self) -> None:
        """
        Shutdown tracing and flush pending spans.

        Ensures all pending spans are exported before shutdown.
        """
        if not self._is_initialized or not OPENTELEMETRY_AVAILABLE:
            return

        try:
            if self._tracer_provider:
                # Force flush pending spans (5 second timeout)
                self._tracer_provider.force_flush(timeout_millis=5000)

            self._is_initialized = False
            self._tracers.clear()

        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.INTERNAL_ERROR,
                message=f"Error during tracing shutdown: {e!s}",
            ) from e


# Global factory accessor functions
def get_tracer_factory(config: TracerConfig | None = None) -> TracerFactory:
    """
    Get the global tracer factory instance.

    Args:
        config: Configuration for first initialization

    Returns:
        Global tracer factory instance
    """
    return TracerFactory.get_instance(config)


async def initialize_tracing(config: TracerConfig | None = None) -> None:
    """
    Initialize global distributed tracing.

    Args:
        config: Tracer configuration (auto-detected if None)
    """
    factory = get_tracer_factory(config)
    await factory.initialize()


async def shutdown_tracing() -> None:
    """Shutdown global distributed tracing."""
    factory = TracerFactory.get_instance()
    await factory.shutdown()
