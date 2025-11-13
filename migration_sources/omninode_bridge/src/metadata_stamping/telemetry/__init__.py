"""
Telemetry and observability components for MetadataStampingService.

Provides comprehensive distributed tracing, metrics collection, and monitoring
capabilities using OpenTelemetry standards.
"""

from .opentelemetry import (
    MetricsCollector,
    TelemetryManager,
    TraceEnricher,
    create_span,
    get_meter,
    get_tracer,
    initialize_telemetry,
)

__all__ = [
    "initialize_telemetry",
    "get_tracer",
    "get_meter",
    "create_span",
    "TelemetryManager",
    "MetricsCollector",
    "TraceEnricher",
]
