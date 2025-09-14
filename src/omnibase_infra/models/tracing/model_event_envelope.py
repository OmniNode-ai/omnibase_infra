"""Event Envelope Model.

Strongly-typed model for event envelope used in tracing context.
Replaces Dict[str, Any] usage to maintain ONEX compliance.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID
from datetime import datetime


class ModelEventEnvelope(BaseModel):
    """Model for event envelope used in tracing context."""

    # Event identification
    event_id: UUID = Field(
        description="Unique event identifier"
    )

    event_type: str = Field(
        max_length=100,
        description="Type of event"
    )

    event_version: str = Field(
        max_length=20,
        description="Event schema version"
    )

    correlation_id: UUID = Field(
        description="Request correlation ID"
    )

    # Timing information
    timestamp: datetime = Field(
        description="Event timestamp"
    )

    processing_started_at: Optional[datetime] = Field(
        default=None,
        description="When processing started"
    )

    processing_completed_at: Optional[datetime] = Field(
        default=None,
        description="When processing completed"
    )

    # Event routing
    source_service: str = Field(
        max_length=100,
        description="Service that generated the event"
    )

    target_service: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Intended target service"
    )

    routing_key: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Message routing key"
    )

    # Event metadata
    priority: Optional[int] = Field(
        default=None,
        ge=0,
        le=10,
        description="Event processing priority (0-10)"
    )

    retry_count: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Number of processing retries"
    )

    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retries allowed"
    )

    # Content information
    content_type: str = Field(
        default="application/json",
        max_length=100,
        description="Content type of event payload"
    )

    content_encoding: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Content encoding (if compressed)"
    )

    content_size_bytes: Optional[int] = Field(
        default=None,
        ge=0,
        description="Size of event payload in bytes"
    )

    # Security and validation
    checksum: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Payload checksum for integrity verification"
    )

    signature: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Digital signature for authenticity"
    )

    # Processing context
    processing_mode: Optional[str] = Field(
        default=None,
        pattern="^(sync|async|batch|stream)$",
        description="Event processing mode"
    )

    batch_id: Optional[UUID] = Field(
        default=None,
        description="Batch identifier (if part of batch processing)"
    )

    partition_key: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Partitioning key for distributed processing"
    )

    # Error handling
    dead_letter_queue_eligible: bool = Field(
        default=True,
        description="Whether event can be sent to dead letter queue"
    )

    error_message: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Error message from last processing attempt"
    )

    # Environment and deployment
    environment: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Environment where event was generated"
    )

    region: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Geographic region"
    )

    deployment_version: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Deployment version of source service"
    )

    # Tracing integration
    trace_headers: Optional[List[str]] = Field(
        default=None,
        max_items=20,
        description="List of tracing header names present in envelope"
    )

    span_context_injected: bool = Field(
        default=False,
        description="Whether span context has been injected into headers"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }