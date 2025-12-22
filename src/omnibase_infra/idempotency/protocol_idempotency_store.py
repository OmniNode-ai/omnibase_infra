# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol definition for Idempotency Store.

This module defines the ProtocolIdempotencyStore protocol that all idempotency
store implementations must follow. The protocol defines the contract for
message deduplication in distributed systems.

Note:
    This protocol is defined locally in omnibase_infra because it is not
    available in omnibase_spi versions 0.4.0/0.4.1. The protocol should be
    migrated to omnibase_spi in a future release.

    TODO(OMN-XXX): Migrate to omnibase_spi 0.5.0+ and update imports
    in InMemoryIdempotencyStore and PostgresIdempotencyStore.

Protocol Methods:
    - check_and_record: Atomically check if message was processed and record if not
    - is_processed: Check if a message was already processed (read-only)
    - mark_processed: Mark a message as processed (upsert)
    - cleanup_expired: Remove entries older than TTL

Implementations:
    - InMemoryIdempotencyStore: In-memory store for testing (OMN-945)
    - PostgresIdempotencyStore: Production PostgreSQL store (OMN-945)
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable
from uuid import UUID


@runtime_checkable
class ProtocolIdempotencyStore(Protocol):
    """Protocol for idempotency store implementations.

    Defines the contract for message deduplication stores that track processed
    messages and prevent duplicate processing in distributed systems.

    All implementations must provide atomic check-and-record semantics to
    ensure exactly-once processing guarantees.

    Key Properties:
        - Thread-safe: All operations must be safe for concurrent access
        - Atomic: check_and_record must provide atomic check-and-set semantics
        - Domain-isolated: Messages can be namespaced by domain for isolated deduplication

    Example:
        >>> store: ProtocolIdempotencyStore = InMemoryIdempotencyStore()
        >>> message_id = uuid4()
        >>> is_new = await store.check_and_record(message_id, domain="orders")
        >>> if is_new:
        ...     # Process the message
        ...     pass
    """

    async def check_and_record(
        self,
        message_id: UUID,
        domain: str | None = None,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Atomically check if message was processed and record if not.

        This is the primary idempotency operation. It must be atomic to ensure
        that when multiple coroutines call this method simultaneously with the
        same (domain, message_id), exactly ONE caller receives True.

        Args:
            message_id: Unique identifier for the message.
            domain: Optional domain namespace for isolated deduplication.
                Messages with the same message_id but different domains are
                treated as distinct messages.
            correlation_id: Optional correlation ID for distributed tracing.
                Stored with the record for observability purposes.

        Returns:
            True if message is new (should be processed).
            False if message is duplicate (should be skipped).
        """
        ...

    async def is_processed(
        self,
        message_id: UUID,
        domain: str | None = None,
    ) -> bool:
        """Check if a message was already processed.

        Read-only check that does not modify the store. Useful for querying
        message status without affecting the idempotency state.

        Args:
            message_id: Unique identifier for the message.
            domain: Optional domain namespace.

        Returns:
            True if the message has been processed.
            False if the message has not been processed.
        """
        ...

    async def mark_processed(
        self,
        message_id: UUID,
        domain: str | None = None,
        correlation_id: UUID | None = None,
        processed_at: datetime | None = None,
    ) -> None:
        """Mark a message as processed.

        Records a message as processed without checking if it already exists.
        If the record already exists, updates it with the new values.

        This is an upsert operation - it will create a new record if one
        doesn't exist, or update the existing record if it does.

        Args:
            message_id: Unique identifier for the message.
            domain: Optional domain namespace for isolated deduplication.
            correlation_id: Optional correlation ID for tracing.
            processed_at: Optional timestamp of when processing occurred.
                If None, implementations should use the current UTC time.
        """
        ...

    async def cleanup_expired(
        self,
        ttl_seconds: int,
    ) -> int:
        """Remove entries older than TTL.

        Cleans up old idempotency records based on their processed_at timestamp.
        This prevents unbounded storage growth.

        Args:
            ttl_seconds: Time-to-live in seconds. Records older than this
                value (based on processed_at timestamp) are removed.

        Returns:
            Number of entries removed.
        """
        ...


__all__ = ["ProtocolIdempotencyStore"]
