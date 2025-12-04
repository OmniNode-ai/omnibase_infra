"""Event headers model for ONEX event bus messages.

This module provides ModelEventHeaders, a Pydantic model implementing
ProtocolEventHeaders from omnibase_spi for use with event bus implementations.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


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
        retry_count: Current retry attempt number.
        max_retries: Maximum retry attempts allowed.
        ttl_seconds: Message time-to-live in seconds.

    Example:
        ```python
        headers = ModelEventHeaders(
            source="order-service",
            event_type="order.created",
            routing_key="orders.us-east",
        )
        is_valid = await headers.validate_headers()
        ```
    """

    content_type: str = Field(default="application/json")
    correlation_id: UUID = Field(default_factory=uuid4)
    message_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str
    event_type: str
    schema_version: str = Field(default="1.0.0")
    destination: Optional[str] = Field(default=None)
    trace_id: Optional[str] = Field(default=None)
    span_id: Optional[str] = Field(default=None)
    parent_span_id: Optional[str] = Field(default=None)
    operation_name: Optional[str] = Field(default=None)
    priority: Optional[Literal["low", "normal", "high", "critical"]] = Field(
        default="normal"
    )
    routing_key: Optional[str] = Field(default=None)
    partition_key: Optional[str] = Field(default=None)
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)
    ttl_seconds: Optional[int] = Field(default=None)

    model_config = {"frozen": False, "extra": "forbid"}

    async def validate_headers(self) -> bool:
        """Validate that required headers are present and valid.

        Returns:
            True if correlation_id and event_type are valid, False otherwise.
        """
        return bool(self.correlation_id and self.event_type)
