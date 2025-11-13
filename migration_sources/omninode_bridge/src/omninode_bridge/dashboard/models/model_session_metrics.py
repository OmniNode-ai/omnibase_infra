"""
Session Metrics Model for Dashboard Performance Analysis.

Comprehensive performance metrics for code generation sessions.

ONEX Compliance:
- Suffix-based naming: ModelSessionMetrics
- Strong typing with Pydantic validation
- Nested model composition for complex structures
"""

from datetime import datetime
from typing import Any, ClassVar, Literal, Optional

from pydantic import BaseModel, Field


class ModelBottleneck(BaseModel):
    """
    Performance bottleneck identification.

    Represents a topic or operation with unusually high response times.

    Attributes:
        topic: Kafka topic with performance issues
        avg_response_time_ms: Average response time for this topic
        count: Number of events on this topic
        severity: Bottleneck severity (high, medium, low)
    """

    topic: str = Field(..., description="Kafka topic with performance issues")
    avg_response_time_ms: float = Field(
        ..., description="Average response time for this topic", gt=0
    )
    count: int = Field(..., description="Number of events on this topic", gt=0)
    severity: Literal["high", "medium", "low"] = Field(
        ..., description="Bottleneck severity level"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "topic": "omninode_codegen_request_validate_v1",
                "avg_response_time_ms": 4200.0,
                "count": 2,
                "severity": "high",
            }
        }


class ModelTimeline(BaseModel):
    """
    Session timeline metrics.

    Timing information for session execution.

    Attributes:
        start_time: When session started (None if no events)
        end_time: When session ended (None if no events)
        duration_ms: Total session duration in milliseconds
    """

    start_time: Optional[datetime] = Field(
        default=None, description="When session started (None if no events)"
    )
    end_time: Optional[datetime] = Field(
        default=None, description="When session ended (None if no events)"
    )
    duration_ms: int = Field(
        ..., description="Total session duration in milliseconds", ge=0
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "start_time": "2025-01-20T10:00:00Z",
                "end_time": "2025-01-20T10:00:22.5Z",
                "duration_ms": 22500,
            }
        }


class ModelSessionMetrics(BaseModel):
    """
    Comprehensive performance metrics for a code generation session.

    Includes success rates, response times, bottleneck analysis,
    and event type breakdowns.

    Attributes:
        session_id: Code generation session ID
        total_events: Total number of events in session
        successful_events: Number of successful events
        failed_events: Number of failed events
        success_rate: Ratio of successful to total events (0.0-1.0)
        avg_response_time_ms: Average response time across all events
        min_response_time_ms: Fastest response time
        max_response_time_ms: Slowest response time
        p50_response_time_ms: Median response time (50th percentile)
        p95_response_time_ms: 95th percentile response time
        p99_response_time_ms: 99th percentile response time
        total_processing_time_ms: Sum of all processing times
        event_type_breakdown: Count of events by type
        topic_breakdown: Count of events by topic
        bottlenecks: List of identified performance bottlenecks
        timeline: Session timing information
    """

    # Session identification
    session_id: str = Field(..., description="Code generation session ID")

    # Event counts
    total_events: int = Field(..., description="Total number of events", ge=0)
    successful_events: int = Field(..., description="Number of successful events", ge=0)
    failed_events: int = Field(..., description="Number of failed events", ge=0)
    success_rate: float = Field(
        ..., description="Ratio of successful to total events", ge=0.0, le=1.0
    )

    # Response time metrics
    avg_response_time_ms: float = Field(
        ..., description="Average response time across all events", ge=0.0
    )
    min_response_time_ms: int = Field(..., description="Fastest response time", ge=0)
    max_response_time_ms: int = Field(..., description="Slowest response time", ge=0)
    p50_response_time_ms: int = Field(
        ..., description="Median response time (50th percentile)", ge=0
    )
    p95_response_time_ms: int = Field(
        ..., description="95th percentile response time", ge=0
    )
    p99_response_time_ms: int = Field(
        ..., description="99th percentile response time", ge=0
    )
    total_processing_time_ms: int = Field(
        ..., description="Sum of all processing times", ge=0
    )

    # Breakdowns
    event_type_breakdown: dict[str, int] = Field(
        ..., description="Count of events by type (request, response, status, error)"
    )
    topic_breakdown: dict[str, int] = Field(
        ..., description="Count of events by Kafka topic"
    )

    # Performance analysis
    bottlenecks: list[ModelBottleneck] = Field(
        ..., description="List of identified performance bottlenecks"
    )
    timeline: ModelTimeline = Field(..., description="Session timing information")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "session_id": "650e8400-e29b-41d4-a716-446655440001",
                "total_events": 20,
                "successful_events": 18,
                "failed_events": 2,
                "success_rate": 0.90,
                "avg_response_time_ms": 1250.5,
                "min_response_time_ms": 450,
                "max_response_time_ms": 4500,
                "p50_response_time_ms": 1000,
                "p95_response_time_ms": 3500,
                "p99_response_time_ms": 4200,
                "total_processing_time_ms": 22500,
                "event_type_breakdown": {
                    "request": 10,
                    "response": 8,
                    "status": 1,
                    "error": 1,
                },
                "topic_breakdown": {
                    "omninode_codegen_request_analyze_v1": 3,
                    "omninode_codegen_response_analyze_v1": 3,
                    "omninode_codegen_request_validate_v1": 2,
                },
                "bottlenecks": [
                    {
                        "topic": "omninode_codegen_request_validate_v1",
                        "avg_response_time_ms": 4200.0,
                        "count": 2,
                        "severity": "high",
                    }
                ],
                "timeline": {
                    "start_time": "2025-01-20T10:00:00Z",
                    "end_time": "2025-01-20T10:00:22.5Z",
                    "duration_ms": 22500,
                },
            }
        }
