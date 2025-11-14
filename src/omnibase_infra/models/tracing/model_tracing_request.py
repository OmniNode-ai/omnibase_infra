"""Tracing Request Model.

Shared model for distributed tracing operation requests.
Used for tracing operations and span management.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from .model_event_envelope import ModelEventEnvelope
from .model_parent_context import ModelParentContext
from .model_span_attributes import ModelSpanAttributes
from .model_trace_context import ModelTraceContext


class ModelTracingRequest(BaseModel):
    """Model for distributed tracing operation requests."""

    operation_type: str = Field(
        description="Type of tracing operation",
        regex=r"^(start_span|end_span|inject_context|extract_context|get_current_span)$",
    )

    correlation_id: UUID = Field(
        description="Request correlation ID for tracking",
    )

    timestamp: datetime = Field(
        description="Request timestamp",
    )

    operation_name: str | None = Field(
        default=None,
        description="Name of the operation to trace (for start_span)",
    )

    span_kind: str | None = Field(
        default="internal",
        description="Type of span (internal, server, client, producer, consumer)",
    )

    trace_context: ModelTraceContext | None = Field(
        default=None,
        description="Trace context for context operations",
    )

    span_attributes: ModelSpanAttributes | None = Field(
        default=None,
        description="Attributes to add to span",
    )

    parent_context: ModelParentContext | None = Field(
        default=None,
        description="Parent context for span creation",
    )

    event_envelope: ModelEventEnvelope | None = Field(
        default=None,
        description="Event envelope for context injection/extraction",
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
