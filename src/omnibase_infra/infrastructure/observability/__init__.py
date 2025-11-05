"""
ONEX Infrastructure Observability Utilities.

Provides structured logging, distributed tracing, and metrics collection
for ONEX infrastructure components.

Key Components:
- StructuredLogger: Structured logging with correlation IDs and context
- TracerFactory: OpenTelemetry tracing setup and management
- MetricsRegistry: Prometheus metrics collection and export

Example Usage:
    # Structured logging
    from omnibase_infra.infrastructure.observability import LoggerFactory
    logger = LoggerFactory.get_logger("my_component")
    logger.info("Processing started", correlation_id="123")

    # Distributed tracing
    from omnibase_infra.infrastructure.observability import get_tracer_factory
    factory = get_tracer_factory()
    await factory.initialize()
    tracer = factory.get_tracer(__name__)

    # Metrics collection
    from omnibase_infra.infrastructure.observability import get_metrics_registry
    metrics = get_metrics_registry()
    metrics.initialize()
    metrics.record_request("postgres_adapter", "query", "success")
"""

from omnibase_infra.infrastructure.observability.metrics_registry import (
    MetricsConfig,
    MetricsRegistry,
    get_metrics_registry,
    initialize_metrics,
)
from omnibase_infra.infrastructure.observability.structured_logger import (
    LoggerFactory,
    StructuredLogger,
    StructuredLoggerConfig,
)
from omnibase_infra.infrastructure.observability.tracer_factory import (
    TracerConfig,
    TracerFactory,
    get_tracer_factory,
    initialize_tracing,
    shutdown_tracing,
)

__all__ = [
    # Structured logging
    "StructuredLogger",
    "StructuredLoggerConfig",
    "LoggerFactory",
    # Distributed tracing
    "TracerFactory",
    "TracerConfig",
    "get_tracer_factory",
    "initialize_tracing",
    "shutdown_tracing",
    # Metrics
    "MetricsRegistry",
    "MetricsConfig",
    "get_metrics_registry",
    "initialize_metrics",
]
