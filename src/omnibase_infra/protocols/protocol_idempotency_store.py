# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Idempotency Store Protocol Definition.

This module defines the protocol interface for idempotency stores used
to track processed messages and prevent duplicate processing.

Note: This protocol is defined locally in omnibase_infra until it is
promoted to omnibase_spi. Once available in omnibase_spi, imports should
be updated to use the canonical location.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable
from uuid import UUID


@runtime_checkable
class ProtocolIdempotencyStore(Protocol):
    """Protocol for idempotency stores.

    Idempotency stores track processed message IDs to enable exactly-once
    or at-least-once-with-dedup semantics in distributed message processing.

    Key Operations:
        - check_and_record: Atomically check if message is new and record it
        - is_processed: Read-only check if message was already processed
        - mark_processed: Explicitly mark a message as processed
        - cleanup_expired: Remove old records based on TTL

    Concurrency Safety:
        All implementations MUST be safe for concurrent coroutine access. The
        check_and_record method MUST provide atomic check-and-set semantics to
        prevent race conditions. Use asyncio.Lock for coroutine-safety.

    Domain Isolation:
        The optional `domain` parameter allows different message namespaces
        to be tracked independently. The same message_id can exist in different
        domains without conflict.

    Example Usage:
        >>> store: ProtocolIdempotencyStore = get_store()
        >>> is_new = await store.check_and_record(
        ...     message_id=uuid4(),
        ...     domain="orders",
        ...     correlation_id=uuid4(),
        ... )
        >>> if is_new:
        ...     # Process the message
        ...     await process_order(message)
        >>> else:
        ...     # Duplicate - skip processing
        ...     log.info("Skipping duplicate message")
    """

    async def check_and_record(
        self,
        message_id: UUID,
        domain: str | None = None,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Atomically check if message was processed and record if not.

        This is the primary idempotency check method. It MUST be atomic:
        when multiple concurrent calls are made with the same (domain, message_id),
        exactly ONE caller receives True.

        Args:
            message_id: Unique identifier for the message.
            domain: Optional domain namespace for isolated deduplication.
            correlation_id: Optional correlation ID for distributed tracing.

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

        Read-only check that does not modify the store. Use this for
        inspection or status queries where recording is not desired.

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
        Use this for explicit marking, such as when replaying events or
        seeding the store with known processed messages.

        Args:
            message_id: Unique identifier for the message.
            domain: Optional domain namespace for isolated deduplication.
            correlation_id: Optional correlation ID for tracing.
            processed_at: Optional timestamp of when processing occurred.
                If None, implementations should use current time.
        """
        ...

    async def cleanup_expired(
        self,
        ttl_seconds: int,
    ) -> int:
        """Remove entries older than TTL.

        Cleans up old idempotency records based on their processed_at timestamp.
        This prevents unbounded growth of the store.

        Args:
            ttl_seconds: Time-to-live in seconds. Records older than this
                value (based on processed_at timestamp) are removed.

        Returns:
            Number of entries removed.
        """
        ...


__all__: list[str] = ["ProtocolIdempotencyStore"]
