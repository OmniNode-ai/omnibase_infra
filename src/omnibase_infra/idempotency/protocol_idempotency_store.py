# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol definition for Idempotency Store.

This module defines the ProtocolIdempotencyStore protocol that all idempotency
store implementations must follow. The protocol defines the contract for
message deduplication in distributed systems.

Migration Note:
    This protocol is defined locally in omnibase_infra because it is not
    available in omnibase_spi versions 0.4.0/0.4.1. This is a TEMPORARY
    definition that should be migrated to omnibase_spi in a future release.

    Migration Path (OMN-1000):
        1. When omnibase_spi 0.5.0+ is released with ProtocolIdempotencyStore,
           update pyproject.toml to require the new version
        2. Update imports in StoreIdempotencyInmemory and StoreIdempotencyPostgres
           to use: `from omnibase_spi.protocols import ProtocolIdempotencyStore`
        3. Remove this local protocol definition
        4. Run tests to verify compatibility

    The protocol contract is intentionally designed to match the expected
    omnibase_spi interface to ensure a smooth migration.

Protocol Methods:
    - check_and_record: Atomically check if message was processed and record if not
    - is_processed: Check if a message was already processed (read-only)
    - mark_processed: Mark a message as processed (upsert)
    - cleanup_expired: Remove entries older than TTL

Implementations:
    - StoreIdempotencyInmemory: In-memory store for testing (OMN-945)
    - StoreIdempotencyPostgres: Production PostgreSQL store (OMN-945)

Security Considerations:
    - Concurrency Safety: All implementations MUST be safe for concurrent access.
      Multiple coroutines may call check_and_record simultaneously with the
      same message_id. Implementations must use appropriate synchronization
      (e.g., asyncio.Lock for in-memory, database transactions for PostgreSQL).

      Note: This is coroutine-safe for asyncio concurrent access, not thread-safe.
      For multi-threaded access, additional synchronization would be required.

    - Atomicity: The check_and_record method MUST provide atomic check-and-set
      semantics. When multiple callers race with the same (domain, message_id),
      exactly ONE caller must receive True. This prevents duplicate processing
      in concurrent scenarios.

    - Domain Isolation: Messages are namespaced by domain to prevent cross-tenant
      conflicts. The (domain, message_id) tuple forms the unique key. Different
      domains can safely use the same message_id without collision. This is
      critical for multi-tenant deployments where tenant data must be isolated.

    - Correlation ID Usage: The correlation_id parameter is used ONLY for
      distributed tracing and observability. It is NOT used for authentication
      or authorization. Security-sensitive operations must implement their own
      authentication layer; do not rely on correlation_id for access control.

    - Input Validation: Implementations should validate that message_id is a
      valid UUID. Domain names should be sanitized if used in storage backends
      (e.g., as part of database keys or Redis key prefixes).
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
        - Coroutine-safe: All operations must be safe for concurrent async access
        - Atomic: check_and_record must provide atomic check-and-set semantics
        - Domain-isolated: Messages can be namespaced by domain for isolated deduplication

    Example:
        >>> store: ProtocolIdempotencyStore = StoreIdempotencyInmemory()
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
