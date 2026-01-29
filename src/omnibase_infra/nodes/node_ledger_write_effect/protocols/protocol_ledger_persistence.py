# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Protocol definition for event ledger persistence operations.

This module defines the ProtocolLedgerPersistence interface for ledger
storage and retrieval operations. Handlers implementing this protocol
can be used interchangeably for testing and production.

Design Decisions:
    - runtime_checkable: Enables isinstance() checks for duck typing
    - Async methods: All operations are async for non-blocking I/O
    - Typed models: Uses Pydantic models for type safety
    - Nullable metadata: Query filters are optional (None = no filter)
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_ledger_write_effect.models import (
        ModelLedgerAppendResult,
        ModelLedgerEntry,
    )
    from omnibase_infra.nodes.reducers.models import ModelPayloadLedgerAppend


@runtime_checkable
class ProtocolLedgerPersistence(Protocol):
    """Protocol for event ledger persistence operations.

    This protocol defines the interface for appending events to the audit
    ledger and querying events by various criteria. Implementations must
    provide idempotent append operations via the (topic, partition, kafka_offset)
    unique constraint.

    Implementations:
        - HandlerLedgerAppend: Production handler composing with HandlerDb
        - MockLedgerPersistence: Test double for unit testing

    Example:
        >>> async def process_events(
        ...     persistence: ProtocolLedgerPersistence,
        ...     payload: ModelPayloadLedgerAppend,
        ... ) -> ModelLedgerAppendResult:
        ...     return await persistence.append(payload)
    """

    async def append(
        self,
        payload: ModelPayloadLedgerAppend,
    ) -> ModelLedgerAppendResult:
        """Append an event to the audit ledger with idempotent write support.

        Uses INSERT ... ON CONFLICT DO NOTHING with the (topic, partition, kafka_offset)
        unique constraint. Duplicate events are detected without raising errors.

        Args:
            payload: Event payload containing Kafka position and event data.
                The payload includes base64-encoded event_key and event_value
                which are decoded to BYTEA for storage.

        Returns:
            ModelLedgerAppendResult with:
                - success: True if operation completed without error
                - ledger_entry_id: UUID of created entry, None if duplicate
                - duplicate: True if ON CONFLICT was triggered

        Raises:
            InfraConnectionError: If database connection fails
            InfraTimeoutError: If operation times out
        """
        ...

    async def query_by_correlation_id(
        self,
        correlation_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ModelLedgerEntry]:
        """Query ledger entries by correlation ID for distributed tracing.

        Args:
            correlation_id: The correlation ID to search for.
            limit: Maximum number of entries to return (default: 100).
            offset: Number of entries to skip for pagination (default: 0).

        Returns:
            List of ModelLedgerEntry matching the correlation ID,
            ordered by event_timestamp descending.
        """
        ...

    async def query_by_time_range(
        self,
        start: datetime,
        end: datetime,
        correlation_id: UUID | None = None,
        event_type: str | None = None,
        topic: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ModelLedgerEntry]:
        """Query ledger entries within a time range.

        Uses COALESCE(event_timestamp, ledger_written_at) for consistent
        ordering even when event_timestamp is NULL.

        Args:
            start: Start of time range (inclusive).
            end: End of time range (exclusive).
            correlation_id: Correlation ID for distributed tracing (auto-generated if None).
            event_type: Optional filter by event type.
            topic: Optional filter by Kafka topic.
            limit: Maximum number of entries to return (default: 100, max: 10000).
            offset: Number of entries to skip for pagination (default: 0).

        Returns:
            List of ModelLedgerEntry within the time range,
            ordered by timestamp descending.
        """
        ...


__all__ = ["ProtocolLedgerPersistence"]
