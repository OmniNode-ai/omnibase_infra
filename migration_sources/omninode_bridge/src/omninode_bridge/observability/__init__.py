"""
Observability package for omninode_bridge.

Provides structured logging, distributed tracing, and correlation tracking
for comprehensive system observability.
"""

from omninode_bridge.observability.logging_config import (
    CorrelationFilter,
    JsonFormatter,
    add_extra_context,
    configure_logging,
    correlation_context,
    get_logger,
)
from omninode_bridge.observability.tracing import (
    add_span_attributes,
    add_span_event,
    get_current_span,
    get_tracer,
    set_span_error,
    trace_async,
    trace_sync,
)

__all__ = [
    # Logging
    "configure_logging",
    "get_logger",
    "correlation_context",
    "CorrelationFilter",
    "JsonFormatter",
    "add_extra_context",
    # Tracing
    "get_tracer",
    "get_current_span",
    "add_span_attributes",
    "add_span_event",
    "set_span_error",
    "trace_async",
    "trace_sync",
]
