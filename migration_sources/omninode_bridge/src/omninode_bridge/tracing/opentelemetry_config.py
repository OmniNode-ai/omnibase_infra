"""OpenTelemetry distributed tracing configuration for OmniNode Bridge."""

import os
from typing import Any, Optional

import structlog
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.kafka import KafkaInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from ..middleware.request_correlation import get_correlation_context

logger = structlog.get_logger(__name__)


class OpenTelemetryConfig:
    """Configuration and setup for OpenTelemetry distributed tracing."""

    def __init__(self, service_name: str, service_version: str = "1.0.0"):
        """Initialize OpenTelemetry configuration.

        Args:
            service_name: Name of the service for tracing
            service_version: Version of the service
        """
        self.service_name = service_name
        self.service_version = service_version
        self.is_initialized = False

        # Configuration from environment variables
        self.otlp_endpoint = os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
        )
        self.enable_tracing = os.getenv("ENABLE_OTEL_TRACING", "true").lower() == "true"
        self.enable_prometheus = (
            os.getenv("ENABLE_PROMETHEUS_METRICS", "true").lower() == "true"
        )
        self.trace_sampling_rate = float(os.getenv("TRACE_SAMPLING_RATE", "1.0"))
        self.enable_auto_instrumentation = (
            os.getenv("ENABLE_AUTO_INSTRUMENTATION", "true").lower() == "true"
        )

    def configure_tracing(self) -> Optional[TracerProvider]:
        """Configure OpenTelemetry tracing."""
        if not self.enable_tracing:
            logger.info("OpenTelemetry tracing disabled via configuration")
            return None

        try:
            # Create resource with service information
            resource = Resource.create(
                {
                    "service.name": self.service_name,
                    "service.version": self.service_version,
                    "service.namespace": "omninode-bridge",
                    "deployment.environment": os.getenv("ENVIRONMENT", "development"),
                }
            )

            # Create tracer provider
            tracer_provider = TracerProvider(
                resource=resource,
                sampler=self._create_sampler(),
            )

            # Configure OTLP exporter
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.otlp_endpoint,
                insecure=True,  # Use insecure=True for HTTP, insecure=False for HTTPS
            )

            # Add batch span processor
            span_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(span_processor)

            # Set the global tracer provider
            trace.set_tracer_provider(tracer_provider)

            logger.info(
                "OpenTelemetry tracing configured",
                service_name=self.service_name,
                otlp_endpoint=self.otlp_endpoint,
                sampling_rate=self.trace_sampling_rate,
            )

            return tracer_provider

        except Exception as e:
            logger.error(
                "Failed to configure OpenTelemetry tracing",
                error=str(e),
                service_name=self.service_name,
            )
            return None

    def configure_metrics(self) -> Optional[MeterProvider]:
        """Configure OpenTelemetry metrics."""
        if not self.enable_prometheus:
            logger.info("Prometheus metrics disabled via configuration")
            return None

        try:
            # Create resource with service information
            resource = Resource.create(
                {
                    "service.name": self.service_name,
                    "service.version": self.service_version,
                    "service.namespace": "omninode-bridge",
                }
            )

            # Create Prometheus metric reader
            prometheus_reader = PrometheusMetricReader()

            # Create meter provider
            meter_provider = MeterProvider(
                resource=resource,
                metric_readers=[prometheus_reader],
            )

            # Set the global meter provider
            metrics.set_meter_provider(meter_provider)

            logger.info(
                "OpenTelemetry metrics configured",
                service_name=self.service_name,
                prometheus_enabled=True,
            )

            return meter_provider

        except Exception as e:
            logger.error(
                "Failed to configure OpenTelemetry metrics",
                error=str(e),
                service_name=self.service_name,
            )
            return None

    def configure_auto_instrumentation(self) -> None:
        """Configure automatic instrumentation for common libraries."""
        if not self.enable_auto_instrumentation:
            logger.info("Auto-instrumentation disabled via configuration")
            return

        try:
            # Instrument FastAPI
            FastAPIInstrumentor().instrument(
                excluded_urls="health,metrics,favicon.ico",
                request_hook=self._fastapi_request_hook,
                response_hook=self._fastapi_response_hook,
            )

            # Instrument AsyncPG (PostgreSQL)
            AsyncPGInstrumentor().instrument()

            # Instrument AIOHTTP client
            AioHttpClientInstrumentor().instrument(
                request_hook=self._aiohttp_request_hook,
                response_hook=self._aiohttp_response_hook,
            )

            # Instrument Kafka (if available)
            try:
                KafkaInstrumentor().instrument()
            except Exception as kafka_error:
                logger.warning(
                    "Could not instrument Kafka",
                    error=str(kafka_error),
                )

            logger.info(
                "Auto-instrumentation configured",
                service_name=self.service_name,
                instrumentations=["FastAPI", "AsyncPG", "AIOHTTP", "Kafka"],
            )

        except Exception as e:
            logger.error(
                "Failed to configure auto-instrumentation",
                error=str(e),
                service_name=self.service_name,
            )

    def initialize(self) -> bool:
        """Initialize OpenTelemetry with full configuration.

        Returns:
            True if initialization successful, False otherwise
        """
        if self.is_initialized:
            logger.warning("OpenTelemetry already initialized")
            return True

        try:
            # Configure tracing
            tracer_provider = self.configure_tracing()

            # Configure metrics
            meter_provider = self.configure_metrics()

            # Configure auto-instrumentation
            self.configure_auto_instrumentation()

            self.is_initialized = True

            logger.info(
                "OpenTelemetry initialization completed",
                service_name=self.service_name,
                tracing_enabled=tracer_provider is not None,
                metrics_enabled=meter_provider is not None,
                auto_instrumentation_enabled=self.enable_auto_instrumentation,
            )

            return True

        except Exception as e:
            logger.error(
                "OpenTelemetry initialization failed",
                error=str(e),
                service_name=self.service_name,
            )
            return False

    def _create_sampler(self):
        """Create trace sampler based on configuration."""
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBasedSampler

        return TraceIdRatioBasedSampler(rate=self.trace_sampling_rate)

    def _fastapi_request_hook(self, span, scope: dict[str, Any]) -> None:
        """Hook to enrich FastAPI request spans with correlation context."""
        try:
            correlation_context = get_correlation_context()
            for key, value in correlation_context.items():
                if value:
                    span.set_attribute(f"omninode.{key}", value)

            # Add service-specific attributes
            span.set_attribute("omninode.service.name", self.service_name)
            span.set_attribute("omninode.service.version", self.service_version)

        except Exception as e:
            logger.warning("Failed to add correlation context to span", error=str(e))

    def _fastapi_response_hook(self, span, message: dict[str, Any]) -> None:
        """Hook to enrich FastAPI response spans."""
        try:
            # Add response-specific attributes
            if "status" in message:
                status_code = message["status"]
                span.set_attribute("http.status_code", status_code)
                if status_code >= 400:
                    span.set_status(trace.StatusCode.ERROR, f"HTTP {status_code}")

        except Exception as e:
            logger.warning("Failed to add response info to span", error=str(e))

    def _aiohttp_request_hook(self, span, params) -> None:
        """Hook to enrich AIOHTTP client request spans."""
        try:
            correlation_context = get_correlation_context()
            for key, value in correlation_context.items():
                if value:
                    span.set_attribute(f"omninode.{key}", value)

        except Exception as e:
            logger.warning(
                "Failed to add correlation context to HTTP client span", error=str(e)
            )

    def _aiohttp_response_hook(self, span, params) -> None:
        """Hook to enrich AIOHTTP client response spans."""
        try:
            # Response-specific attributes handled by auto-instrumentation
            pass
        except Exception as e:
            logger.warning("Failed to process HTTP client response hook", error=str(e))


# Global OpenTelemetry configuration instance
_otel_config: Optional[OpenTelemetryConfig] = None


def initialize_opentelemetry(service_name: str, service_version: str = "1.0.0") -> bool:
    """Initialize OpenTelemetry for the service.

    Args:
        service_name: Name of the service
        service_version: Version of the service

    Returns:
        True if initialization successful, False otherwise
    """
    global _otel_config

    if _otel_config is not None:
        logger.warning("OpenTelemetry already initialized")
        return True

    _otel_config = OpenTelemetryConfig(service_name, service_version)
    return _otel_config.initialize()


def get_tracer(name: str = __name__):
    """Get OpenTelemetry tracer instance.

    Args:
        name: Name for the tracer (usually __name__)

    Returns:
        Tracer instance
    """
    return trace.get_tracer(name)


def get_meter(name: str = __name__):
    """Get OpenTelemetry meter instance.

    Args:
        name: Name for the meter (usually __name__)

    Returns:
        Meter instance
    """
    return metrics.get_meter(name)


def get_current_span():
    """Get the current active span."""
    return trace.get_current_span()


def add_span_attributes(**attributes):
    """Add attributes to the current span.

    Args:
        **attributes: Key-value pairs to add as span attributes
    """
    current_span = get_current_span()
    if current_span.is_recording():
        for key, value in attributes.items():
            current_span.set_attribute(key, value)


def add_span_event(name: str, attributes: Optional[dict[str, Any]] = None):
    """Add an event to the current span.

    Args:
        name: Event name
        attributes: Optional event attributes
    """
    current_span = get_current_span()
    if current_span.is_recording():
        current_span.add_event(name, attributes or {})


def set_span_error(error: Exception):
    """Mark the current span as having an error.

    Args:
        error: The exception that occurred
    """
    current_span = get_current_span()
    if current_span.is_recording():
        current_span.set_status(trace.StatusCode.ERROR, str(error))
        current_span.record_exception(error)
