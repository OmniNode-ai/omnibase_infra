# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Observability sinks for ONEX infrastructure.

This module provides sink implementations for the ONEX observability stack.
Sinks are responsible for buffering and outputting observability data
(logs, metrics) to various backends.

Available Sinks:
    - SinkLoggingStructured: Structured logging sink using structlog
      with JSON/console output formats and configurable buffering.
      Implements ProtocolHotPathLoggingSink.
    - SinkMetricsPrometheus: Thread-safe Prometheus metrics sink with
      cardinality policy enforcement. Implements ProtocolHotPathMetricsSink.

Cardinality Protection:
    The metrics sink enforces cardinality policies via ModelMetricsPolicy
    to prevent high-cardinality label explosions. By default, the following
    labels are forbidden:
        - envelope_id: Unique per-message identifier
        - correlation_id: Request correlation identifier
        - node_id: Node instance identifier
        - runtime_id: Runtime instance identifier

Usage:
    ```python
    from omnibase_infra.observability.sinks import (
        SinkLoggingStructured,
        SinkMetricsPrometheus,
    )
    from omnibase_core.enums import EnumLogLevel

    # Structured logging sink
    log_sink = SinkLoggingStructured(max_buffer_size=500, output_format="json")
    log_sink.emit(EnumLogLevel.INFO, "Processing item", {"item_id": "123"})
    log_sink.flush()

    # Prometheus metrics sink
    metrics_sink = SinkMetricsPrometheus()
    metrics_sink.increment_counter(
        "http_requests_total",
        {"method": "POST", "status": "200"},
    )
    ```

See Also:
    - omnibase_spi.protocols.observability.ProtocolHotPathLoggingSink:
      Protocol implemented by SinkLoggingStructured
    - omnibase_spi.protocols.observability.ProtocolHotPathMetricsSink:
      Protocol implemented by SinkMetricsPrometheus
    - omnibase_core.models.observability.ModelMetricsPolicy:
      Cardinality policy for metrics sinks
"""

from omnibase_infra.observability.sinks.sink_logging_structured import (
    SinkLoggingStructured,
)
from omnibase_infra.observability.sinks.sink_metrics_prometheus import (
    SinkMetricsPrometheus,
)

# Note: DEFAULT_HISTOGRAM_BUCKETS should be imported directly from
# sink_metrics_prometheus, not from this package __init__.py.
# This follows the pattern of importing constants from their source modules.

__all__ = [
    "SinkLoggingStructured",
    "SinkMetricsPrometheus",
]
