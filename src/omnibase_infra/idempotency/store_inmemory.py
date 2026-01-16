# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""In-Memory Idempotency Store.

This module provides an in-memory implementation of ProtocolIdempotencyStore
for testing purposes. It uses a dict with asyncio.Lock for coroutine-safe
operations.

This store is NOT suitable for production use:
- Data is lost on process restart
- Not distributed (single-process only)
- Memory grows unbounded without cleanup

Use StoreIdempotencyPostgres or ValkeyIdempotencyStore for production.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import UUID

from omnibase_infra.idempotency.models import ModelIdempotencyRecord
from omnibase_infra.idempotency.protocol_idempotency_store import (
    ProtocolIdempotencyStore,
)


class StoreIdempotencyInmemory(ProtocolIdempotencyStore):
    """In-memory idempotency store for testing.

    Implements ProtocolIdempotencyStore using a dict for storage and
    asyncio.Lock for coroutine-safe operations. Designed for unit testing
    scenarios where external dependencies are not available.

    Storage Structure:
        Uses dict[tuple[str | None, UUID], ModelIdempotencyRecord] where:
        - Key: (domain, message_id) composite key
        - Value: ModelIdempotencyRecord with full message metadata

    Concurrency Safety:
        All operations are protected by an asyncio.Lock to ensure atomic
        check-and-record semantics even under concurrent coroutine access.
        Note: This is coroutine-safe, not thread-safe. For multi-threaded
        access, additional synchronization would be required.

    Test Utilities:
        - clear(): Reset store to empty state between tests
        - get_record_count(): Assert on number of stored records
        - get_all_records(): Inspect stored records for assertions

    Example:
        >>> store = StoreIdempotencyInmemory()
        >>> result = await store.check_and_record(message_id, domain="test")
        >>> assert result is True  # First call returns True
        >>> result = await store.check_and_record(message_id, domain="test")
        >>> assert result is False  # Duplicate returns False

    See Also:
        - ProtocolIdempotencyStore: Protocol interface definition
        - ModelIdempotencyRecord: Record model used for storage
    """

    def __init__(self) -> None:
        """Initialize the in-memory store with empty storage and lock."""
        self._storage: dict[tuple[str | None, UUID], ModelIdempotencyRecord] = {}
        self._lock = asyncio.Lock()

    async def check_and_record(
        self,
        message_id: UUID,
        domain: str | None = None,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Atomically check if message was processed and record if not.

        Uses asyncio.Lock to ensure atomic check-and-set semantics.
        When multiple coroutines call this method simultaneously with
        the same (domain, message_id), exactly ONE caller receives True.

        Args:
            message_id: Unique identifier for the message.
            domain: Optional domain namespace for isolated deduplication.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            True if message is new (should be processed).
            False if message is duplicate (should be skipped).
        """
        key = (domain, message_id)

        async with self._lock:
            if key in self._storage:
                return False

            record = ModelIdempotencyRecord(
                message_id=message_id,
                domain=domain,
                correlation_id=correlation_id,
                processed_at=datetime.now(UTC),
            )
            self._storage[key] = record
            return True

    async def is_processed(
        self,
        message_id: UUID,
        domain: str | None = None,
    ) -> bool:
        """Check if a message was already processed.

        Read-only check that does not modify the store.

        Args:
            message_id: Unique identifier for the message.
            domain: Optional domain namespace.

        Returns:
            True if the message has been processed.
            False if the message has not been processed.
        """
        key = (domain, message_id)

        async with self._lock:
            return key in self._storage

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

        Args:
            message_id: Unique identifier for the message.
            domain: Optional domain namespace for isolated deduplication.
            correlation_id: Optional correlation ID for tracing.
            processed_at: Optional timestamp of when processing occurred.
                If None, uses datetime.now(timezone.utc).
        """
        key = (domain, message_id)
        timestamp = processed_at if processed_at is not None else datetime.now(UTC)

        record = ModelIdempotencyRecord(
            message_id=message_id,
            domain=domain,
            correlation_id=correlation_id,
            processed_at=timestamp,
        )

        async with self._lock:
            self._storage[key] = record

    async def cleanup_expired(
        self,
        ttl_seconds: int,
    ) -> int:
        """Remove entries older than TTL.

        Cleans up old idempotency records based on their processed_at timestamp.
        Uses single-pass identification to minimize lock hold time.

        Args:
            ttl_seconds: Time-to-live in seconds. Records older than this
                value (based on processed_at timestamp) are removed.

        Returns:
            Number of entries removed.
        """
        now = datetime.now(UTC)

        async with self._lock:
            # Single-pass identification using list comprehension
            expired_keys = [
                key
                for key, record in self._storage.items()
                if (now - record.processed_at).total_seconds() > ttl_seconds
            ]

            # Delete expired keys
            for key in expired_keys:
                del self._storage[key]

            return len(expired_keys)

    # Test utility methods

    async def clear(self) -> None:
        """Clear all records from the store.

        Test utility method to reset store state between tests.

        Example:
            >>> await store.clear()
            >>> assert await store.get_record_count() == 0
        """
        async with self._lock:
            self._storage.clear()

    async def get_record_count(self) -> int:
        """Get the number of records in the store.

        Test utility method for assertions on store size.

        Returns:
            Number of records currently in the store.

        Example:
            >>> await store.check_and_record(uuid4(), domain="test")
            >>> assert await store.get_record_count() == 1
        """
        async with self._lock:
            return len(self._storage)

    async def get_all_records(self) -> list[ModelIdempotencyRecord]:
        """Get all records from the store.

        Test utility method for inspecting stored records.

        Returns:
            List of all ModelIdempotencyRecord instances in the store.

        Example:
            >>> await store.check_and_record(message_id, domain="test")
            >>> records = await store.get_all_records()
            >>> assert len(records) == 1
            >>> assert records[0].message_id == message_id
        """
        async with self._lock:
            return list(self._storage.values())

    async def get_record(
        self,
        message_id: UUID,
        domain: str | None = None,
    ) -> ModelIdempotencyRecord | None:
        """Get a specific record by message_id and domain.

        Test utility method for retrieving individual records.

        Args:
            message_id: Unique identifier for the message.
            domain: Optional domain namespace.

        Returns:
            The ModelIdempotencyRecord if found, None otherwise.

        Example:
            >>> await store.check_and_record(message_id, domain="test")
            >>> record = await store.get_record(message_id, domain="test")
            >>> assert record is not None
            >>> assert record.message_id == message_id
        """
        key = (domain, message_id)

        async with self._lock:
            return self._storage.get(key)


__all__ = ["StoreIdempotencyInmemory"]
