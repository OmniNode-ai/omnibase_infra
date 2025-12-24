# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Idempotency Record Model.

This module provides the Pydantic model for storing idempotency records
that track processed messages to prevent duplicate processing.

The record captures essential metadata about when and how a message was
processed, enabling deduplication across distributed systems.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelIdempotencyRecord(BaseModel):
    """Record of a processed message for idempotency tracking.

    This model represents a single idempotency record that tracks whether
    a specific message has been processed. It is used by idempotency stores
    to persist deduplication state.

    Attributes:
        id: Unique identifier for this record (auto-generated UUID4).
        domain: Optional domain/namespace for message categorization.
            Used to partition idempotency checks (e.g., "orders", "payments").
        message_id: The unique identifier of the processed message.
            This is the primary key for idempotency lookups.
        correlation_id: Optional correlation ID for distributed tracing.
            Links this record to a broader request context.
        processed_at: Timestamp when the message was processed.
            Used for TTL-based cleanup of old records.
        handler_type: Optional handler type identifier for debugging.
            Records which handler processed this message.

    Example:
        >>> from uuid import uuid4
        >>> from datetime import datetime, timezone
        >>> record = ModelIdempotencyRecord(
        ...     message_id=uuid4(),
        ...     domain="orders",
        ...     correlation_id=uuid4(),
        ...     processed_at=datetime.now(timezone.utc),
        ...     handler_type="OrderCreatedHandler",
        ... )
        >>> print(record.message_id)
        UUID('...')
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this idempotency record",
    )
    domain: str | None = Field(
        default=None,
        description="Domain/namespace for message categorization (e.g., 'orders', 'payments')",
        max_length=255,
    )
    message_id: UUID = Field(
        description="Unique identifier of the processed message (primary idempotency key)",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for distributed tracing",
    )
    processed_at: datetime = Field(
        description="Timestamp when the message was processed",
    )
    handler_type: str | None = Field(
        default=None,
        description="Handler type identifier for debugging purposes",
        max_length=255,
    )


__all__: list[str] = ["ModelIdempotencyRecord"]
