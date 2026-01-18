# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Observability Layer.

This module provides observability infrastructure for ONEX including:
- Structured logging sinks with buffering
- Prometheus metrics collection sinks with cardinality enforcement
- Observability hooks for pipeline instrumentation
- Factory for centralized sink creation with singleton support
- Contract-driven handlers for lifecycle management

The observability layer is designed for high-performance hot-path scenarios
where blocking I/O cannot be tolerated. All sinks implement buffering
with configurable policies for memory management.

Architecture Principle: "Handlers own lifecycle, sinks own hot path"
    - Handlers: Contract-driven lifecycle, buffer/flush management,
      configuration, initialization/shutdown
    - Sinks: Fast in-process emission, synchronous non-blocking operations

Cardinality Protection:
    The metrics sink enforces cardinality policies via ModelMetricsPolicy
    to prevent high-cardinality label explosions. By default, the following
    labels are forbidden:
        - envelope_id: Unique per-message identifier
        - correlation_id: Request correlation identifier
        - node_id: Node instance identifier
        - runtime_id: Runtime instance identifier

Submodules:
    - sinks: Output sinks for logs and metrics
    - hooks: Observability lifecycle hooks for pipeline instrumentation
    - handlers: Contract-driven handlers for sink lifecycle management

Key Components:
    - SinkLoggingStructured: Buffered structured logging sink
    - SinkMetricsPrometheus: Thread-safe Prometheus metrics sink
    - HookObservability: Pipeline hook for timing, metrics, and context tracking
    - FactoryObservabilitySink: Factory for creating sinks with configuration
    - HandlerLoggingStructured: EFFECT handler for structured logging lifecycle

Usage:
    ```python
    from omnibase_infra.observability import (
        SinkLoggingStructured,
        SinkMetricsPrometheus,
        HookObservability,
        FactoryObservabilitySink,
        ModelMetricsSinkConfig,
        ModelLoggingSinkConfig,
        HandlerLoggingStructured,
    )
    from omnibase_core.enums import EnumLogLevel

    # Using the factory (recommended for sinks)
    factory = FactoryObservabilitySink()
    metrics_sink = factory.get_or_create_metrics_sink()
    logging_sink = factory.get_or_create_logging_sink()
    hook = factory.create_hook(metrics_sink=metrics_sink)

    # Handler-managed sink (lifecycle, periodic flush)
    handler = HandlerLoggingStructured()
    await handler.initialize({
        "buffer_size": 1000,
        "flush_interval_seconds": 5.0,
    })
    await handler.execute({
        "operation": "logging.emit",
        "payload": {"level": "INFO", "message": "Event", "context": {}},
    })
    await handler.shutdown()

    # With custom configuration
    metrics_config = ModelMetricsSinkConfig(metric_prefix="myapp")
    custom_metrics = factory.create_metrics_sink(config=metrics_config)

    # Direct instantiation (also supported)
    log_sink = SinkLoggingStructured()
    log_sink.emit(EnumLogLevel.INFO, "System started", {"component": "api"})
    log_sink.flush()

    # Prometheus metrics with cardinality enforcement
    metrics_sink = SinkMetricsPrometheus()
    metrics_sink.increment_counter(
        "http_requests_total",
        {"method": "POST", "status": "200"},
    )

    # Pipeline observability hook
    hook = HookObservability(metrics_sink=metrics_sink)
    with hook.operation_context("handler.process", correlation_id="abc-123"):
        result = await handler.execute(payload)
    ```

See Also:
    - omnibase_spi.protocols.observability: Protocol definitions
    - omnibase_core.models.observability: Metrics policy models
"""

from omnibase_infra.observability.factory_observability_sink import (
    FactoryObservabilitySink,
)
from omnibase_infra.observability.handlers import (
    HandlerLoggingStructured,
    HandlerMetricsPrometheus,
    ModelLoggingHandlerConfig,
    ModelLoggingHandlerResponse,
    ModelMetricsHandlerConfig,
    ModelMetricsHandlerPayload,
    ModelMetricsHandlerResponse,
)
from omnibase_infra.observability.hooks import HookObservability
from omnibase_infra.observability.models import (
    ModelLoggingSinkConfig,
    ModelMetricsSinkConfig,
)
from omnibase_infra.observability.sinks import (
    SinkLoggingStructured,
    SinkMetricsPrometheus,
)

__all__: list[str] = [
    # Factory
    "FactoryObservabilitySink",
    "ModelMetricsSinkConfig",
    "ModelLoggingSinkConfig",
    # Handlers - Logging
    "HandlerLoggingStructured",
    "ModelLoggingHandlerConfig",
    "ModelLoggingHandlerResponse",
    # Handlers - Metrics
    "HandlerMetricsPrometheus",
    "ModelMetricsHandlerConfig",
    "ModelMetricsHandlerPayload",
    "ModelMetricsHandlerResponse",
    # Sinks
    "SinkLoggingStructured",
    "SinkMetricsPrometheus",
    # Note: DEFAULT_HISTOGRAM_BUCKETS should be imported directly from
    # omnibase_infra.observability.sinks.sink_metrics_prometheus
    # Hooks
    "HookObservability",
]
