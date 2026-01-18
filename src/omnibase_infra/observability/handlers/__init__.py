# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Observability handlers for ONEX infrastructure.

This module provides handler implementations for the ONEX observability stack.
Handlers are responsible for managing the lifecycle of observability sinks
and providing contract-driven operation execution.

Architecture Principle: "Handlers own lifecycle, sinks own hot path"
    - Handlers: Contract-driven lifecycle, buffer/flush management,
      configuration, initialization/shutdown
    - Sinks: Fast in-process emission, synchronous non-blocking operations

Available Handlers:
    - HandlerLoggingStructured: EFFECT handler for structured logging
      with buffer/flush management. Manages SinkLoggingStructured lifecycle.
    - HandlerMetricsPrometheus: EFFECT handler exposing /metrics HTTP endpoint
      for Prometheus scraping. Supports optional push to Pushgateway.

Handler Types:
    All observability handlers are classified as:
    - handler_type: INFRA_HANDLER (infrastructure protocol handlers)
    - handler_category: EFFECT (side-effecting I/O operations)

Usage:
    ```python
    from omnibase_infra.observability.handlers import (
        HandlerLoggingStructured,
        ModelLoggingHandlerConfig,
        HandlerMetricsPrometheus,
        ModelMetricsHandlerConfig,
    )

    # Initialize logging handler
    handler = HandlerLoggingStructured()
    await handler.initialize({
        "buffer_size": 1000,
        "flush_interval_seconds": 5.0,
        "output_format": "json",
    })

    # Emit log entries via envelope-based routing
    await handler.execute({
        "operation": "logging.emit",
        "payload": {
            "level": "INFO",
            "message": "Processing started",
            "context": {"request_id": "req_123"},
        },
    })

    # Initialize metrics handler
    metrics_handler = HandlerMetricsPrometheus()
    await metrics_handler.initialize({
        "host": "0.0.0.0",
        "port": 9090,
        "path": "/metrics",
    })

    # Metrics now available at http://localhost:9090/metrics
    # Or retrieve programmatically:
    result = await metrics_handler.execute({"operation": "metrics.scrape"})
    print(result.result.payload.metrics_text)

    # Graceful shutdown
    await handler.shutdown()
    await metrics_handler.shutdown()
    ```

See Also:
    - omnibase_infra.observability.sinks: Sink implementations
    - omnibase_infra.enums.EnumHandlerType: Handler type classification
    - omnibase_infra.enums.EnumHandlerTypeCategory: Behavioral classification
"""

from omnibase_infra.observability.handlers.handler_logging_structured import (
    SUPPORTED_OPERATIONS as LOGGING_SUPPORTED_OPERATIONS,
)
from omnibase_infra.observability.handlers.handler_logging_structured import (
    HandlerLoggingStructured,
)
from omnibase_infra.observability.handlers.handler_metrics_prometheus import (
    HANDLER_ID_METRICS,
    HandlerMetricsPrometheus,
)
from omnibase_infra.observability.handlers.handler_metrics_prometheus import (
    SUPPORTED_OPERATIONS as METRICS_SUPPORTED_OPERATIONS,
)
from omnibase_infra.observability.handlers.model_logging_handler_config import (
    ModelLoggingHandlerConfig,
)
from omnibase_infra.observability.handlers.model_logging_handler_response import (
    ModelLoggingHandlerResponse,
)
from omnibase_infra.observability.handlers.model_metrics_handler_config import (
    ModelMetricsHandlerConfig,
)
from omnibase_infra.observability.handlers.model_metrics_handler_payload import (
    ModelMetricsHandlerPayload,
)
from omnibase_infra.observability.handlers.model_metrics_handler_response import (
    ModelMetricsHandlerResponse,
)

__all__: list[str] = [
    # Logging handler
    "HandlerLoggingStructured",
    "ModelLoggingHandlerConfig",
    "ModelLoggingHandlerResponse",
    "LOGGING_SUPPORTED_OPERATIONS",
    # Metrics handler
    "HandlerMetricsPrometheus",
    "ModelMetricsHandlerConfig",
    "ModelMetricsHandlerPayload",
    "ModelMetricsHandlerResponse",
    "HANDLER_ID_METRICS",
    "METRICS_SUPPORTED_OPERATIONS",
]
