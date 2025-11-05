"""Tracing Response Model.

Shared model for distributed tracing operation responses.
Used for returning results from tracing operations.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from .model_span_data import ModelSpanData
from .model_trace_context import ModelTraceContext


class ModelTracingResponse(BaseModel):
    """Model for distributed tracing operation responses."""

    operation_type: str = Field(
        description="Type of operation that was executed",
    )

    success: bool = Field(
        description="Whether the operation was successful",
    )

    correlation_id: UUID = Field(
        description="Request correlation ID for tracking",
    )

    timestamp: datetime = Field(
        description="Response timestamp",
    )

    execution_time_ms: float = Field(
        ge=0.0,
        description="Operation execution time in milliseconds",
    )

    span_id: str | None = Field(
        default=None,
        description="Span ID (for start_span operations)",
    )

    trace_id: str | None = Field(
        default=None,
        description="Trace ID (for span operations)",
    )

    trace_context: ModelTraceContext | None = Field(
        default=None,
        description="Extracted trace context (for extract_context operations)",
    )

    context_injected: bool | None = Field(
        default=None,
        description="Whether context was successfully injected (for inject_context operations)",
    )

    span_data: ModelSpanData | None = Field(
        default=None,
        description="Span data and attributes (for get_current_span operations)",
    )

    error_message: str | None = Field(
        default=None,
        description="Error message if operation failed",
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
