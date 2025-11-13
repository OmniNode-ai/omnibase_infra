"""
Correlated Event Model for Dashboard Event Tracing.

Individual event data model for correlation-based event discovery.

ONEX Compliance:
- Suffix-based naming: ModelCorrelatedEvent
- Strong typing with Pydantic validation
- Optional field handling for flexibility
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelCorrelatedEvent(BaseModel):
    """
    Individual correlated event data.

    Represents a single event in a correlation chain, enabling
    request/response matching and multi-hop event tracing.

    Attributes:
        event_id: Unique identifier for this event
        session_id: Code generation session this event belongs to
        correlation_id: Correlation ID linking related events
        event_type: Type of event (request, response, status, error)
        topic: Kafka topic this event was sent to/received from
        timestamp: When the event occurred
        status: Event processing status (sent, received, failed, processing)
        processing_time_ms: Time taken to process (optional)
        payload: Event data payload
        metadata: Additional context and metadata
    """

    # Event identification
    event_id: UUID = Field(..., description="Unique identifier for this event")
    session_id: UUID = Field(
        ..., description="Code generation session this event belongs to"
    )
    correlation_id: UUID = Field(
        ..., description="Correlation ID linking related events"
    )

    # Event metadata
    event_type: str = Field(
        ..., description="Type of event (request, response, status, error)"
    )
    topic: str = Field(
        ..., description="Kafka topic this event was sent to/received from"
    )
    timestamp: datetime = Field(..., description="When the event occurred")
    status: str = Field(
        ..., description="Event processing status (sent, received, failed, processing)"
    )

    # Performance tracking
    processing_time_ms: Optional[int] = Field(
        default=None,
        description="Time taken to process (optional, for response events)",
    )

    # Event data
    payload: dict[str, Any] = Field(..., description="Event data payload")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional context and metadata"
    )

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "event_id": "550e8400-e29b-41d4-a716-446655440000",
                "session_id": "650e8400-e29b-41d4-a716-446655440001",
                "correlation_id": "750e8400-e29b-41d4-a716-446655440002",
                "event_type": "request",
                "topic": "omninode_codegen_request_analyze_v1",
                "timestamp": "2025-01-20T10:00:00Z",
                "status": "sent",
                "processing_time_ms": None,
                "payload": {
                    "file_path": "/data/code/analysis.py",
                    "content": "def analyze(): pass",
                },
                "metadata": {"source": "omniclaude", "priority": "high"},
            }
        },
    )
