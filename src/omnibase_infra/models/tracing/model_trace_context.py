"""Trace Context Model.

Shared model for distributed trace context information.
Used for trace propagation across infrastructure components.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ModelTraceContext(BaseModel):
    """Model for distributed trace context."""

    trace_id: str = Field(
        description="Unique trace identifier",
    )

    span_id: str = Field(
        description="Current span identifier",
    )

    parent_span_id: str | None = Field(
        default=None,
        description="Parent span identifier",
    )

    correlation_id: UUID = Field(
        description="Correlation ID for request tracking",
    )

    service_name: str = Field(
        description="Name of the service creating the trace",
    )

    operation_name: str = Field(
        description="Name of the operation being traced",
    )

    timestamp: datetime = Field(
        description="Trace context creation timestamp",
    )

    environment: str = Field(
        description="Environment where trace is generated",
    )

    baggage: dict[str, str] | None = Field(
        default=None,
        description="Baggage data for cross-service propagation",
    )

    trace_flags: str | None = Field(
        default=None,
        description="Trace flags for sampling and other options",
    )

    trace_state: str | None = Field(
        default=None,
        description="Trace state for vendor-specific data",
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
