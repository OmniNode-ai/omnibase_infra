"""Event message model for ONEX event bus.

This module provides ModelEventMessage, a Pydantic model implementing
ProtocolEventMessage from omnibase_spi for use with event bus implementations.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders


class ModelEventMessage(BaseModel):
    """Event bus message implementing ProtocolEventMessage.

    Defines the contract that all event message implementations must satisfy
    for Kafka/RedPanda compatibility following ONEX Messaging Design.
    Supports partitioning, offset tracking, and acknowledgment.

    Attributes:
        topic: Kafka topic the message belongs to.
        key: Optional message key for partitioning (bytes).
        value: Serialized message body as bytes.
        headers: Structured message headers.
        offset: Kafka offset (for consumed messages).
        partition: Kafka partition number.

    Example:
        ```python
        message = ModelEventMessage(
            topic="onex.orders.created",
            key=b"customer-123",
            value=b'{"order_id": "ORD-123", "amount": 99.99}',
            headers=ModelEventHeaders(
                source="order-service",
                event_type="order.created",
            ),
        )
        await message.ack()  # Acknowledge after processing
        ```
    """

    topic: str
    key: bytes | None = Field(default=None)
    value: bytes
    headers: ModelEventHeaders
    offset: str | None = Field(default=None)
    partition: int | None = Field(default=None)

    model_config = ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=True, from_attributes=True
    )

    async def ack(self) -> None:
        """Acknowledge message processing.

        For in-memory event bus, this is a no-op.
        For real Kafka/RedPanda implementations, this would commit the offset.
        """
