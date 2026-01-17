# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""In-Memory Effect Idempotency Store with LRU Eviction and TTL.

This module provides a bounded in-memory implementation of
ProtocolEffectIdempotencyStore for tracking completed backends during
dual-backend Effect operations.

Memory Management:
    The store implements two complementary strategies to bound memory:

    1. LRU Eviction (max_cache_size):
       When the cache exceeds max_cache_size, the least recently accessed
       entries are evicted. Uses OrderedDict for O(1) LRU operations.

    2. TTL Expiration (cache_ttl_seconds):
       Entries older than TTL are eligible for cleanup. Cleanup is triggered
       on access (lazy) and periodically based on cleanup_interval_seconds.

Memory Characteristics:
    - Entry size: ~100 bytes per correlation_id
      - UUID key: 16 bytes
      - set[str] for backends: ~40 bytes (2 backends average)
      - float timestamp: 8 bytes
      - Python object overhead: ~40 bytes
    - Default max (10,000 entries): ~1MB

Production Warning:
    This store is NOT suitable for production use:
    - Data is lost on process restart
    - Not distributed (single-process only)
    - Not suitable for long-running operations exceeding TTL

    For production distributed deployments, implement
    ProtocolEffectIdempotencyStore with a persistent backend.

Concurrency Safety:
    All operations are protected by asyncio.Lock for safe concurrent coroutine
    access. Note: This is coroutine-safe, not thread-safe. For multi-threaded
    access, additional synchronization would be required.

Related:
    - ProtocolEffectIdempotencyStore: Protocol interface
    - ModelEffectIdempotencyConfig: Configuration model
    - NodeRegistryEffect: Consumes this store for dual-backend idempotency
    - OMN-954: Registry effect idempotency requirements
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from uuid import UUID

from omnibase_infra.nodes.effects.models.model_effect_idempotency_config import (
    ModelEffectIdempotencyConfig,
)
from omnibase_infra.nodes.effects.protocol_effect_idempotency_store import (
    ProtocolEffectIdempotencyStore,
)


class CacheEntry:
    """Internal cache entry with timestamp for TTL tracking.

    Attributes:
        backends: Set of completed backend identifiers.
        created_at: Monotonic timestamp when entry was created.
        accessed_at: Monotonic timestamp of last access (for LRU).
    """

    __slots__ = ("accessed_at", "backends", "created_at")

    def __init__(self, backends: set[str] | None = None) -> None:
        """Initialize cache entry with optional initial backends.

        Args:
            backends: Initial set of completed backends.
        """
        now = time.monotonic()
        self.backends: set[str] = backends if backends is not None else set()
        self.created_at: float = now
        self.accessed_at: float = now

    def touch(self) -> None:
        """Update accessed_at timestamp for LRU tracking."""
        self.accessed_at = time.monotonic()


class InMemoryEffectIdempotencyStore(ProtocolEffectIdempotencyStore):
    """In-memory idempotency store with LRU eviction and TTL expiration.

    Implements ProtocolEffectIdempotencyStore using an OrderedDict for
    efficient LRU operations and time-based TTL cleanup.

    Storage Structure:
        Uses OrderedDict[UUID, CacheEntry] where:
        - Key: correlation_id (UUID)
        - Value: CacheEntry with backends set and timestamps

    Eviction Strategy:
        1. On each write, check if cache exceeds max_cache_size
        2. If exceeded, evict oldest entries (LRU order) until under limit
        3. Periodically cleanup expired entries (TTL-based)

    Concurrency Safety:
        All operations are protected by asyncio.Lock for atomic access.
        Safe for concurrent async access from multiple coroutines. Note:
        This is coroutine-safe, not thread-safe.

    Memory Characteristics:
        - Per-entry overhead: ~100 bytes
        - max_cache_size=10000 (default): ~1MB
        - max_cache_size=100000: ~10MB
        - Memory formula: total = max_cache_size * 100 bytes + ~50KB overhead

    Performance Characteristics:
        All core operations are O(1) amortized:
        - mark_completed: O(1) - hash lookup + set add + OrderedDict.move_to_end
        - is_completed: O(1) - hash lookup + set contains
        - get_completed_backends: O(k) - where k = backends per entry (typically 2)
        - clear: O(1) - hash delete
        - LRU eviction: O(1) - OrderedDict.popitem(last=False)
        - TTL cleanup: O(n) - but runs lazily on configurable interval

        Throughput (measured with mocks):
        - Sequential: > 500 ops/sec
        - Concurrent (10 workers): > 10,000 ops/sec
        - Sustained single-worker: > 5,000 ops/sec

    Scalability Limits:
        - Maximum practical size: 1,000,000 entries (~100MB)
        - Beyond this, consider persistent backend for durability anyway
        - TTL cleanup may cause brief latency spikes at very high entry counts

    Production Warning:
        This implementation is NOT persistent and NOT distributed.
        Data is lost on process restart. For production deployments:
        - Use a persistent backend (Redis, PostgreSQL)
        - Consider the existing StoreIdempotencyPostgres for full persistence
        - Multi-instance deployments MUST use a shared backend

    Example:
        >>> config = ModelEffectIdempotencyConfig(
        ...     max_cache_size=5000,
        ...     cache_ttl_seconds=1800.0,
        ... )
        >>> store = InMemoryEffectIdempotencyStore(config)
        >>> await store.mark_completed(correlation_id, "consul")
        >>> assert await store.is_completed(correlation_id, "consul")

    See Also:
        - ProtocolEffectIdempotencyStore: Protocol interface definition
        - ModelEffectIdempotencyConfig: Configuration model
        - NodeRegistryEffect: Primary consumer of this store
        - README.md: Comprehensive documentation with configuration guide
    """

    def __init__(
        self,
        config: ModelEffectIdempotencyConfig | None = None,
    ) -> None:
        """Initialize the in-memory store with configuration.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self._config = config or ModelEffectIdempotencyConfig()
        self._cache: OrderedDict[UUID, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._last_cleanup: float = time.monotonic()

    @property
    def max_cache_size(self) -> int:
        """Maximum number of entries before LRU eviction."""
        return self._config.max_cache_size

    @property
    def cache_ttl_seconds(self) -> float:
        """Time-to-live for entries in seconds."""
        return self._config.cache_ttl_seconds

    async def mark_completed(self, correlation_id: UUID, backend: str) -> None:
        """Mark a backend as completed for a correlation ID.

        Records completion and maintains cache bounds via LRU eviction.
        Updates LRU order by moving entry to end of OrderedDict.

        Args:
            correlation_id: Unique identifier for the operation.
            backend: Backend identifier (e.g., "consul", "postgres").
        """
        async with self._lock:
            # Check for TTL cleanup opportunity (sync - lock already held)
            self._maybe_cleanup_expired()

            if correlation_id in self._cache:
                # Update existing entry and move to end (most recently used)
                entry = self._cache[correlation_id]
                entry.backends.add(backend)
                entry.touch()
                self._cache.move_to_end(correlation_id)
            else:
                # Create new entry
                self._cache[correlation_id] = CacheEntry(backends={backend})
                # Evict LRU entries if over capacity (sync - lock already held)
                self._evict_lru_if_needed()

    async def is_completed(self, correlation_id: UUID, backend: str) -> bool:
        """Check if a backend is completed for a correlation ID.

        Updates LRU order on access (read-through updates).

        Args:
            correlation_id: Unique identifier for the operation.
            backend: Backend identifier to check.

        Returns:
            True if the backend is completed, False otherwise.
        """
        async with self._lock:
            entry = self._cache.get(correlation_id)
            if entry is None:
                return False

            # Check if entry is expired
            if self._is_expired(entry):
                # Expired entries return False but aren't immediately deleted
                # to avoid modifying cache during read
                return False

            # Update LRU order
            entry.touch()
            self._cache.move_to_end(correlation_id)

            return backend in entry.backends

    async def get_completed_backends(self, correlation_id: UUID) -> set[str]:
        """Get all completed backends for a correlation ID.

        Args:
            correlation_id: Unique identifier for the operation.

        Returns:
            Copy of completed backend set. Empty set if none or expired.
        """
        async with self._lock:
            entry = self._cache.get(correlation_id)
            if entry is None:
                return set()

            # Check if entry is expired
            if self._is_expired(entry):
                return set()

            # Update LRU order
            entry.touch()
            self._cache.move_to_end(correlation_id)

            # Return copy to prevent external mutation
            return entry.backends.copy()

    async def clear(self, correlation_id: UUID) -> None:
        """Clear completed backends for a correlation ID.

        Args:
            correlation_id: The correlation ID to clear.
        """
        async with self._lock:
            self._cache.pop(correlation_id, None)

    async def clear_all(self) -> None:
        """Clear all entries from the cache.

        Test utility method for resetting state between tests.
        """
        async with self._lock:
            self._cache.clear()
            self._last_cleanup = time.monotonic()

    async def get_cache_size(self) -> int:
        """Get current number of entries in cache.

        Test utility for assertions on cache size.

        Returns:
            Number of entries currently in cache.
        """
        async with self._lock:
            return len(self._cache)

    async def cleanup_expired(self) -> int:
        """Force cleanup of all expired entries.

        Removes all entries that have exceeded the TTL.

        Returns:
            Number of entries removed.
        """
        async with self._lock:
            return self._cleanup_expired_entries()

    async def get_estimated_memory_bytes(self) -> int:
        """Get estimated memory usage of the cache in bytes.

        Calculates approximate memory consumption based on:
        - UUID key: 16 bytes
        - CacheEntry object overhead: ~40 bytes
        - set[str] for backends: ~40 bytes (assumes 2 backends average)
        - float timestamps (created_at, accessed_at): 16 bytes
        - Python object overhead per entry: ~40 bytes

        Total estimated per entry: ~152 bytes (rounded to 150 for estimation)

        This is an approximation useful for monitoring and capacity planning.
        Actual memory usage may vary based on:
        - Number of backends per entry
        - Backend string lengths
        - Python version and implementation
        - Memory allocator behavior

        Returns:
            Estimated memory usage in bytes.

        Example:
            >>> store = InMemoryEffectIdempotencyStore()
            >>> await store.mark_completed(uuid4(), "consul")
            >>> memory = await store.get_estimated_memory_bytes()
            >>> assert memory > 0
        """
        # Constants for memory estimation (in bytes)
        base_overhead = 200  # OrderedDict base overhead
        per_entry_bytes = 150  # Estimated bytes per cache entry

        async with self._lock:
            entry_count = len(self._cache)

            # Calculate additional backend string memory
            total_backend_bytes = 0
            for entry in self._cache.values():
                # Each backend string: ~50 bytes average (object + chars)
                total_backend_bytes += len(entry.backends) * 50

            return base_overhead + (entry_count * per_entry_bytes) + total_backend_bytes

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if an entry has exceeded TTL.

        Args:
            entry: Cache entry to check.

        Returns:
            True if entry is expired, False otherwise.
        """
        now = time.monotonic()
        age = now - entry.created_at
        return age > self._config.cache_ttl_seconds

    def _maybe_cleanup_expired(self) -> None:
        """Cleanup expired entries if cleanup interval has passed.

        Called during write operations to lazily maintain cache hygiene.
        Only runs if cleanup_interval_seconds has passed since last cleanup.

        Note:
            This is a synchronous method because it performs only in-memory
            operations (dict iteration, timestamp checks, deletions). It must
            be called with self._lock held by the caller. The lock provides
            thread safety; async is not needed for CPU-bound dict operations.
        """
        now = time.monotonic()
        if now - self._last_cleanup >= self._config.cleanup_interval_seconds:
            self._cleanup_expired_entries()
            self._last_cleanup = now

    def _cleanup_expired_entries(self) -> int:
        """Remove all expired entries from cache.

        Note:
            This is a synchronous method because it performs only in-memory
            operations (dict iteration, timestamp checks, deletions). It must
            be called with self._lock held by the caller. The lock provides
            thread safety; async is not needed for CPU-bound dict operations.

        Returns:
            Number of entries removed.
        """
        now = time.monotonic()
        expired_keys: list[UUID] = []

        for key, entry in self._cache.items():
            if now - entry.created_at > self._config.cache_ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)

    def _evict_lru_if_needed(self) -> int:
        """Evict least recently used entries if over capacity.

        Removes oldest entries (front of OrderedDict) until cache size
        is within max_cache_size limit.

        Note:
            This is a synchronous method because it performs only in-memory
            operations (len check, dict popitem). It must be called with
            self._lock held by the caller. The lock provides thread safety;
            async is not needed for CPU-bound dict operations.

        Returns:
            Number of entries evicted.
        """
        evicted = 0
        while len(self._cache) > self._config.max_cache_size:
            # popitem(last=False) removes from front (oldest/LRU)
            self._cache.popitem(last=False)
            evicted += 1

        return evicted


__all__ = ["InMemoryEffectIdempotencyStore"]
