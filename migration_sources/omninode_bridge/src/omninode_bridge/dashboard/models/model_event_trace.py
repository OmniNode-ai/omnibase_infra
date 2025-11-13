"""
Event Trace Model for Dashboard Session Tracing.

Complete event trace results for code generation sessions.

ONEX Compliance:
- Suffix-based naming: ModelEventTrace
- Strong typing with Pydantic validation
- Nested model composition
"""

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omninode_bridge.dashboard.models.model_correlated_event import ModelCorrelatedEvent


class ModelEventTrace(BaseModel):
    """
    Complete event trace for a code generation session.

    Contains all events for a session within a time range, with
    session-level metrics and status information.

    Attributes:
        session_id: Code generation session ID
        events: List of events in chronological order
        total_events: Total number of events found
        session_duration_ms: Duration from first to last event
        status: Session completion status (completed, in_progress, failed, unknown)
        start_time: Timestamp of first event (None if no events)
        end_time: Timestamp of last event (None if no events)
        time_range_hours: Time range searched for events
    """

    # Session identification
    session_id: UUID = Field(..., description="Code generation session ID")

    # Event data
    events: list[ModelCorrelatedEvent] = Field(
        ..., description="List of events in chronological order"
    )
    total_events: int = Field(..., description="Total number of events found", ge=0)

    # Session metrics
    session_duration_ms: int = Field(
        ..., description="Duration from first to last event in milliseconds", ge=0
    )
    status: Literal["completed", "in_progress", "failed", "unknown"] = Field(
        ..., description="Session completion status"
    )

    # Timing
    start_time: Optional[datetime] = Field(
        default=None, description="Timestamp of first event (None if no events)"
    )
    end_time: Optional[datetime] = Field(
        default=None, description="Timestamp of last event (None if no events)"
    )

    # Query metadata
    time_range_hours: int = Field(
        ..., description="Time range searched for events in hours", gt=0
    )

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "session_id": "650e8400-e29b-41d4-a716-446655440001",
                "events": [
                    {
                        "event_id": "550e8400-e29b-41d4-a716-446655440000",
                        "session_id": "650e8400-e29b-41d4-a716-446655440001",
                        "correlation_id": "750e8400-e29b-41d4-a716-446655440002",
                        "event_type": "request",
                        "topic": "omninode_codegen_request_analyze_v1",
                        "timestamp": "2025-01-20T10:00:00Z",
                        "status": "sent",
                        "payload": {},
                        "metadata": {},
                    },
                    {
                        "event_id": "550e8400-e29b-41d4-a716-446655440003",
                        "session_id": "650e8400-e29b-41d4-a716-446655440001",
                        "correlation_id": "750e8400-e29b-41d4-a716-446655440002",
                        "event_type": "response",
                        "topic": "omninode_codegen_response_analyze_v1",
                        "timestamp": "2025-01-20T10:00:05Z",
                        "status": "received",
                        "processing_time_ms": 5000,
                        "payload": {},
                        "metadata": {},
                    },
                ],
                "total_events": 2,
                "session_duration_ms": 5000,
                "status": "completed",
                "start_time": "2025-01-20T10:00:00Z",
                "end_time": "2025-01-20T10:00:05Z",
                "time_range_hours": 24,
            }
        },
    )
