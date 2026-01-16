"""Event headers model for ONEX event bus messages.

This module provides ModelEventHeaders, a Pydantic model implementing
ProtocolEventHeaders from omnibase_spi for use with event bus implementations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.utils import validate_timezone_aware_datetime


class ModelEventHeaders(BaseModel):
    """Headers for ONEX event bus messages implementing ProtocolEventHeaders.

    Standardized headers for ONEX event bus messages ensuring strict
    interoperability across all agents and preventing integration failures.
    Includes tracing, routing, and retry configuration.

    Attributes:
        content_type: MIME type of the message body.
        correlation_id: UUID for correlating related messages.
        message_id: Unique identifier for this message.
        timestamp: Message creation timestamp.
        source: Service that produced the message.
        event_type: Type identifier for the event.
        schema_version: Version of the message schema (simplified from ProtocolSemVer).
        destination: Optional target destination.
        trace_id: Distributed tracing trace ID.
        span_id: Distributed tracing span ID.
        parent_span_id: Parent span for trace hierarchy.
        operation_name: Name of the operation being traced.
        priority: Message priority level.
        routing_key: Key for message routing.
        partition_key: Key for partition assignment.
        retry_count: Current MESSAGE-LEVEL retry attempt number (application-level).
            This tracks end-to-end message delivery attempts across services,
            incremented when a handler fails and the message is republished.
            Distinct from bus-level retry in EventBusKafka (max_retry_attempts)
            which handles transient Kafka connection failures.
        max_retries: Maximum MESSAGE-LEVEL retry attempts allowed (application-level).
            When retry_count >= max_retries, message should be sent to DLQ.
        ttl_seconds: Message time-to-live in seconds.

    Example:
        ```python
        from datetime import UTC, datetime
        headers = ModelEventHeaders(
            source="order-service",
            event_type="order.created",
            routing_key="orders.us-east",
            timestamp=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
        )
        is_valid = await headers.validate_headers()
        ```
    """

    content_type: str = Field(default="application/json")
    correlation_id: UUID = Field(default_factory=uuid4)
    message_id: UUID = Field(default_factory=uuid4)
    # Timestamps - MUST be explicitly injected (no default_factory for testability)
    timestamp: datetime = Field(
        ..., description="Message creation timestamp (must be explicitly provided)"
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_timezone_aware(cls, v: datetime) -> datetime:
        """Validate that timestamp is timezone-aware.

        Delegates to shared utility for consistent validation across all models.
        """
        return validate_timezone_aware_datetime(v)

    source: str
    event_type: str
    schema_version: str = Field(default="1.0.0")
    destination: str | None = Field(default=None)
    trace_id: str | None = Field(default=None)
    span_id: str | None = Field(default=None)
    parent_span_id: str | None = Field(default=None)
    operation_name: str | None = Field(default=None)
    priority: Literal["low", "normal", "high", "critical"] = Field(default="normal")
    routing_key: str | None = Field(default=None)
    partition_key: str | None = Field(default=None)
    retry_count: int = Field(
        default=0,
        description=(
            "Current MESSAGE-LEVEL retry attempt (application-level). "
            "Distinct from EventBusKafka.max_retry_attempts (bus-level)."
        ),
    )
    max_retries: int = Field(
        default=3,
        description=(
            "Maximum MESSAGE-LEVEL retry attempts (application-level). "
            "When retry_count >= max_retries, message should go to DLQ."
        ),
    )
    ttl_seconds: int | None = Field(default=None)

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    async def validate_headers(self) -> bool:
        """Validate that required headers are present and valid.

        Returns:
            True if correlation_id and event_type are valid, False otherwise.
        """
        return bool(self.correlation_id and self.event_type)
