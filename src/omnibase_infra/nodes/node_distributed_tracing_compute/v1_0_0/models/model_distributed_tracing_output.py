"""Distributed Tracing Output Model.

Node-specific output model for the distributed tracing compute node.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ModelDistributedTracingOutput(BaseModel):
    """Output model for distributed tracing operations."""

    success: bool = Field(
        description="Whether the operation succeeded",
    )

    operation_type: str = Field(
        description="Type of operation that was performed",
    )

    correlation_id: UUID = Field(
        description="Correlation ID from the request",
    )

    result: dict[str, str | bool | float | list[str]] | None = Field(
        default=None,
        description="Operation-specific result data with strongly typed values",
    )

    error_message: str | None = Field(
        default=None,
        description="Error message if operation failed",
    )

    trace_id: str | None = Field(
        default=None,
        description="OpenTelemetry trace ID (if applicable)",
    )

    span_id: str | None = Field(
        default=None,
        description="OpenTelemetry span ID (if applicable)",
    )

    tracing_enabled: bool = Field(
        description="Whether tracing is currently enabled",
    )

    timestamp: datetime = Field(
        description="Response timestamp",
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
