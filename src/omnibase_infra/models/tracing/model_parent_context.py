"""Parent Context Model.

Strongly-typed model for OpenTelemetry parent context.
Replaces Dict[str, Any] usage to maintain ONEX compliance.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ModelParentContext(BaseModel):
    """Model for OpenTelemetry parent context."""

    # Trace identification
    trace_id: str = Field(
        min_length=32,
        max_length=32,
        description="OpenTelemetry trace ID (32 hex characters)"
    )

    span_id: str = Field(
        min_length=16,
        max_length=16,
        description="OpenTelemetry span ID (16 hex characters)"
    )

    # Trace flags
    trace_flags: int = Field(
        default=1,
        ge=0,
        le=255,
        description="OpenTelemetry trace flags (8-bit value)"
    )

    # Sampling decision
    is_sampled: bool = Field(
        default=True,
        description="Whether this trace is sampled"
    )

    # Trace state (W3C format)
    trace_state: Optional[str] = Field(
        default=None,
        max_length=512,
        description="W3C trace state header value"
    )

    # Parent span information
    parent_span_name: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Name of the parent span"
    )

    parent_service_name: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Name of the service that created the parent span"
    )

    parent_service_version: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Version of the parent service"
    )

    # Context propagation
    propagation_format: Optional[str] = Field(
        default=None,
        pattern="^(w3c|b3|jaeger|opencensus)$",
        description="Context propagation format used"
    )

    # Remote context indicator
    is_remote: bool = Field(
        default=False,
        description="Whether this is a remote parent context"
    )

    # Timing information
    parent_start_time: Optional[str] = Field(
        default=None,
        description="ISO timestamp when parent span started"
    )

    # Baggage (OpenTelemetry baggage)
    baggage_count: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Number of baggage items"
    )

    # Debug information
    debug_enabled: Optional[bool] = Field(
        default=None,
        description="Whether debug tracing is enabled"
    )

    force_sampling: Optional[bool] = Field(
        default=None,
        description="Whether sampling should be forced for this trace"
    )

    # Priority information
    priority: Optional[int] = Field(
        default=None,
        ge=0,
        le=10,
        description="Trace priority level (0-10)"
    )

    # Environment context
    environment: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Environment where parent span was created"
    )

    cluster: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Cluster where parent span was created"
    )