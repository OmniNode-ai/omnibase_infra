# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for Effect layer idempotency stores.

This module defines the protocol interface for pluggable idempotency backends
used by Effect nodes to track completed backends during dual-backend operations.

Design Rationale:
    This protocol is separate from ProtocolIdempotencyStore (omnibase_spi)
    because Effect-level idempotency has different semantics:

    1. Tracks "completed backends" per correlation_id (not just message_id)
    2. Backend granularity: "consul", "postgres", etc.
    3. Optimized for dual-backend retry scenarios
    4. Does not need full message metadata (domain, processed_at patterns)

    The ProtocolIdempotencyStore from omnibase_spi is designed for message-level
    deduplication with domain isolation, which is a different use case.

Available Implementations:
    - InMemoryEffectIdempotencyStore: Bounded LRU cache with TTL (default)
    - Future: RedisEffectIdempotencyStore, PostgresEffectIdempotencyStore

Production Considerations:
    The in-memory implementation is NOT suitable for:
    - Distributed deployments (multi-instance)
    - Scenarios requiring persistence across restarts
    - Long-running operations exceeding TTL

    For production, implement a persistent backend using this protocol.

Related:
    - ModelEffectIdempotencyConfig: Configuration for in-memory store
    - InMemoryEffectIdempotencyStore: Default in-memory implementation
    - NodeRegistryEffect: Consumes this protocol for dual-backend idempotency
    - OMN-954: Registry effect idempotency requirements
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID


class ProtocolEffectIdempotencyStore(Protocol):
    """Protocol for Effect layer idempotency storage backends.

    Defines the interface for tracking completed backends per correlation_id.
    Implementations must be async-safe and support bounded memory usage.

    Concurrency Safety:
        Implementations MUST be safe for concurrent coroutine access.
        Use asyncio.Lock or equivalent async synchronization primitives.
        Note: asyncio.Lock provides coroutine-safety, not thread-safety.

    Memory Bounds:
        Implementations SHOULD support:
        - Maximum entry count (LRU eviction)
        - TTL-based expiration

    Example Implementation:
        >>> class MyIdempotencyStore:
        ...     async def mark_completed(
        ...         self, correlation_id: UUID, backend: str
        ...     ) -> None:
        ...         # Persist completion
        ...         await self._db.insert(correlation_id, backend)
        ...
        ...     async def is_completed(
        ...         self, correlation_id: UUID, backend: str
        ...     ) -> bool:
        ...         return await self._db.exists(correlation_id, backend)
        ...
        ...     async def get_completed_backends(
        ...         self, correlation_id: UUID
        ...     ) -> set[str]:
        ...         return await self._db.get_backends(correlation_id)
        ...
        ...     async def clear(self, correlation_id: UUID) -> None:
        ...         await self._db.delete(correlation_id)
    """

    async def mark_completed(self, correlation_id: UUID, backend: str) -> None:
        """Mark a backend as completed for a correlation ID.

        Records that the specified backend has successfully completed its
        operation for the given correlation_id. Subsequent calls with the
        same (correlation_id, backend) pair are idempotent.

        Args:
            correlation_id: Unique identifier for the operation.
            backend: Backend identifier (e.g., "consul", "postgres").

        Raises:
            InfraUnavailableError: If the store is unavailable.
        """
        ...

    async def is_completed(self, correlation_id: UUID, backend: str) -> bool:
        """Check if a backend is completed for a correlation ID.

        Args:
            correlation_id: Unique identifier for the operation.
            backend: Backend identifier to check.

        Returns:
            True if the backend is completed, False otherwise.

        Raises:
            InfraUnavailableError: If the store is unavailable.
        """
        ...

    async def get_completed_backends(self, correlation_id: UUID) -> set[str]:
        """Get all completed backends for a correlation ID.

        Args:
            correlation_id: Unique identifier for the operation.

        Returns:
            Set of completed backend identifiers. Empty set if none.

        Raises:
            InfraUnavailableError: If the store is unavailable.
        """
        ...

    async def clear(self, correlation_id: UUID) -> None:
        """Clear completed backends for a correlation ID.

        Removes all backend completion records for the given correlation_id.
        Used for testing or to force re-registration.

        Args:
            correlation_id: The correlation ID to clear.

        Raises:
            InfraUnavailableError: If the store is unavailable.
        """
        ...


__all__ = ["ProtocolEffectIdempotencyStore"]
