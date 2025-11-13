"""OpenTelemetry distributed tracing components for OmniNode Bridge."""

from .opentelemetry_config import (
    OpenTelemetryConfig,
    add_span_attributes,
    add_span_event,
    get_current_span,
    get_meter,
    get_tracer,
    initialize_opentelemetry,
    set_span_error,
)

__all__ = [
    "OpenTelemetryConfig",
    "add_span_attributes",
    "add_span_event",
    "get_current_span",
    "get_meter",
    "get_tracer",
    "initialize_opentelemetry",
    "set_span_error",
]
