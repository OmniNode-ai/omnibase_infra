"""Span Data Model.

Strongly-typed model for OpenTelemetry span data.
Replaces Dict[str, Any] usage to maintain ONEX compliance.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from omnibase_infra.models.infrastructure.tracing.model_span_attributes import ModelSpanAttributes


class ModelSpanEvent(BaseModel):
    """Model for span events/logs."""

    name: str = Field(
        max_length=200,
        description="Event name",
    )

    timestamp: datetime = Field(
        description="Event timestamp",
    )

    attributes: ModelSpanAttributes | None = Field(
        default=None,
        description="Event attributes",
    )


class ModelSpanLink(BaseModel):
    """Model for span links."""

    trace_id: str = Field(
        min_length=32,
        max_length=32,
        description="Linked trace ID (32 hex characters)",
    )

    span_id: str = Field(
        min_length=16,
        max_length=16,
        description="Linked span ID (16 hex characters)",
    )

    trace_flags: int = Field(
        default=1,
        ge=0,
        le=255,
        description="Trace flags for linked span",
    )

    attributes: ModelSpanAttributes | None = Field(
        default=None,
        description="Link attributes",
    )


class ModelSpanStatus(BaseModel):
    """Model for span status."""

    code: str = Field(
        pattern="^(UNSET|OK|ERROR)$",
        description="Span status code",
    )

    message: str | None = Field(
        default=None,
        max_length=1000,
        description="Status description message",
    )


class ModelSpanData(BaseModel):
    """Model for OpenTelemetry span data."""

    # Span identification
    trace_id: str = Field(
        min_length=32,
        max_length=32,
        description="OpenTelemetry trace ID (32 hex characters)",
    )

    span_id: str = Field(
        min_length=16,
        max_length=16,
        description="OpenTelemetry span ID (16 hex characters)",
    )

    parent_span_id: str | None = Field(
        default=None,
        min_length=16,
        max_length=16,
        description="Parent span ID (16 hex characters)",
    )

    # Span metadata
    name: str = Field(
        max_length=200,
        description="Span operation name",
    )

    kind: str = Field(
        pattern="^(INTERNAL|SERVER|CLIENT|PRODUCER|CONSUMER)$",
        description="Span kind",
    )

    status: ModelSpanStatus = Field(
        description="Span status information",
    )

    # Timing information
    start_time: datetime = Field(
        description="Span start timestamp",
    )

    end_time: datetime | None = Field(
        default=None,
        description="Span end timestamp (None if still active)",
    )

    duration_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Span duration in milliseconds",
    )

    # Span data
    attributes: ModelSpanAttributes | None = Field(
        default=None,
        description="Span attributes",
    )

    events: list[ModelSpanEvent] | None = Field(
        default=None,
        max_items=1000,
        description="Span events/logs",
    )

    links: list[ModelSpanLink] | None = Field(
        default=None,
        max_items=100,
        description="Span links to other spans",
    )

    # Resource information
    service_name: str = Field(
        max_length=100,
        description="Service name that created the span",
    )

    service_version: str | None = Field(
        default=None,
        max_length=50,
        description="Service version",
    )

    service_instance_id: str | None = Field(
        default=None,
        max_length=100,
        description="Service instance identifier",
    )

    # Instrumentation information
    instrumentation_library_name: str | None = Field(
        default=None,
        max_length=100,
        description="Name of the instrumentation library",
    )

    instrumentation_library_version: str | None = Field(
        default=None,
        max_length=50,
        description="Version of the instrumentation library",
    )

    # Sampling information
    is_sampled: bool = Field(
        description="Whether this span is sampled",
    )

    sampling_priority: int | None = Field(
        default=None,
        ge=0,
        le=10,
        description="Sampling priority (0-10)",
    )

    # Error information
    has_error: bool = Field(
        default=False,
        description="Whether the span contains error information",
    )

    error_type: str | None = Field(
        default=None,
        max_length=100,
        description="Error type/class name",
    )

    error_message: str | None = Field(
        default=None,
        max_length=1000,
        description="Error message",
    )

    # Performance metrics
    cpu_usage_percent: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="CPU usage during span",
    )

    memory_usage_mb: float | None = Field(
        default=None,
        ge=0.0,
        description="Memory usage during span in MB",
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
