# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for InMemoryEffectIdempotencyStore.

This test suite validates the bounded in-memory idempotency store used by
Effect nodes to track completed backends during dual-backend operations.

Test Coverage:
    1. Basic operations (mark_completed, is_completed, get_completed_backends, clear)
    2. LRU eviction when cache exceeds max_size
    3. TTL-based expiration of stale entries
    4. Concurrent access safety via asyncio.Lock
    5. Memory bounds verification

Related:
    - InMemoryEffectIdempotencyStore: Store implementation
    - ModelEffectIdempotencyConfig: Configuration model
    - ProtocolEffectIdempotencyStore: Protocol interface
    - NodeRegistryEffect: Primary consumer of this store
    - OMN-954: Registry effect idempotency requirements
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch
from uuid import uuid4

import pytest

from omnibase_infra.nodes.effects.models.model_effect_idempotency_config import (
    ModelEffectIdempotencyConfig,
)
from omnibase_infra.nodes.effects.store_effect_idempotency_inmemory import (
    InMemoryEffectIdempotencyStore,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def default_store() -> InMemoryEffectIdempotencyStore:
    """Create a store with default configuration."""
    return InMemoryEffectIdempotencyStore()


@pytest.fixture
def small_cache_store() -> InMemoryEffectIdempotencyStore:
    """Create a store with small cache for LRU testing."""
    config = ModelEffectIdempotencyConfig(
        max_cache_size=3,
        cache_ttl_seconds=3600.0,
    )
    return InMemoryEffectIdempotencyStore(config=config)


@pytest.fixture
def short_ttl_store() -> InMemoryEffectIdempotencyStore:
    """Create a store with minimum TTL for expiration testing."""
    config = ModelEffectIdempotencyConfig(
        max_cache_size=100,
        cache_ttl_seconds=1.0,  # Minimum TTL (1 second)
        cleanup_interval_seconds=1.0,  # Minimum interval
    )
    return InMemoryEffectIdempotencyStore(config=config)


# -----------------------------------------------------------------------------
# Test Basic Operations
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestBasicOperations:
    """Test basic idempotency store operations."""

    @pytest.mark.asyncio
    async def test_mark_and_check_completed(
        self,
        default_store: InMemoryEffectIdempotencyStore,
    ) -> None:
        """Test marking a backend as completed and checking it."""
        correlation_id = uuid4()

        # Initially not completed
        assert await default_store.is_completed(correlation_id, "consul") is False

        # Mark as completed
        await default_store.mark_completed(correlation_id, "consul")

        # Now completed
        assert await default_store.is_completed(correlation_id, "consul") is True

        # Other backends still not completed
        assert await default_store.is_completed(correlation_id, "postgres") is False

    @pytest.mark.asyncio
    async def test_get_completed_backends(
        self,
        default_store: InMemoryEffectIdempotencyStore,
    ) -> None:
        """Test getting all completed backends for a correlation ID."""
        correlation_id = uuid4()

        # Initially empty
        completed = await default_store.get_completed_backends(correlation_id)
        assert completed == set()

        # Mark backends
        await default_store.mark_completed(correlation_id, "consul")
        await default_store.mark_completed(correlation_id, "postgres")

        # Both completed
        completed = await default_store.get_completed_backends(correlation_id)
        assert completed == {"consul", "postgres"}

    @pytest.mark.asyncio
    async def test_clear_correlation_id(
        self,
        default_store: InMemoryEffectIdempotencyStore,
    ) -> None:
        """Test clearing completed backends for a correlation ID."""
        correlation_id = uuid4()

        # Mark backends
        await default_store.mark_completed(correlation_id, "consul")
        await default_store.mark_completed(correlation_id, "postgres")

        # Verify completed
        completed = await default_store.get_completed_backends(correlation_id)
        assert len(completed) == 2

        # Clear
        await default_store.clear(correlation_id)

        # Verify cleared
        completed = await default_store.get_completed_backends(correlation_id)
        assert completed == set()

    @pytest.mark.asyncio
    async def test_clear_all(
        self,
        default_store: InMemoryEffectIdempotencyStore,
    ) -> None:
        """Test clearing all entries from the store."""
        # Add multiple correlation IDs
        for _ in range(5):
            await default_store.mark_completed(uuid4(), "consul")

        # Verify cache has entries
        assert await default_store.get_cache_size() == 5

        # Clear all
        await default_store.clear_all()

        # Verify empty
        assert await default_store.get_cache_size() == 0

    @pytest.mark.asyncio
    async def test_returns_copy_of_backends(
        self,
        default_store: InMemoryEffectIdempotencyStore,
    ) -> None:
        """Test that get_completed_backends returns a copy."""
        correlation_id = uuid4()
        await default_store.mark_completed(correlation_id, "consul")

        # Get backends
        completed = await default_store.get_completed_backends(correlation_id)

        # Modify the returned set
        completed.add("fake")

        # Internal set should be unchanged
        internal = await default_store.get_completed_backends(correlation_id)
        assert "fake" not in internal


# -----------------------------------------------------------------------------
# Test LRU Eviction
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestLRUEviction:
    """Test LRU eviction when cache exceeds max_size."""

    @pytest.mark.asyncio
    async def test_lru_eviction_on_overflow(
        self,
        small_cache_store: InMemoryEffectIdempotencyStore,
    ) -> None:
        """Test that oldest entries are evicted when cache exceeds max_size."""
        # Cache max size is 3
        id1, id2, id3, id4 = uuid4(), uuid4(), uuid4(), uuid4()

        # Add 3 entries (at capacity)
        await small_cache_store.mark_completed(id1, "consul")
        await small_cache_store.mark_completed(id2, "consul")
        await small_cache_store.mark_completed(id3, "consul")

        # Verify all 3 present
        assert await small_cache_store.get_cache_size() == 3
        assert await small_cache_store.is_completed(id1, "consul") is True
        assert await small_cache_store.is_completed(id2, "consul") is True
        assert await small_cache_store.is_completed(id3, "consul") is True

        # Add 4th entry - should evict id1 (oldest)
        await small_cache_store.mark_completed(id4, "consul")

        # Cache should still be at max size
        assert await small_cache_store.get_cache_size() == 3

        # id1 should be evicted (LRU)
        assert await small_cache_store.is_completed(id1, "consul") is False
        # Others should remain
        assert await small_cache_store.is_completed(id2, "consul") is True
        assert await small_cache_store.is_completed(id3, "consul") is True
        assert await small_cache_store.is_completed(id4, "consul") is True

    @pytest.mark.asyncio
    async def test_access_updates_lru_order(
        self,
        small_cache_store: InMemoryEffectIdempotencyStore,
    ) -> None:
        """Test that accessing an entry updates its LRU position."""
        # Cache max size is 3
        id1, id2, id3, id4 = uuid4(), uuid4(), uuid4(), uuid4()

        # Add 3 entries
        await small_cache_store.mark_completed(id1, "consul")
        await small_cache_store.mark_completed(id2, "consul")
        await small_cache_store.mark_completed(id3, "consul")

        # Access id1 (moves it to end of LRU)
        await small_cache_store.is_completed(id1, "consul")

        # Add 4th entry - should evict id2 (now oldest)
        await small_cache_store.mark_completed(id4, "consul")

        # id1 should remain (recently accessed)
        assert await small_cache_store.is_completed(id1, "consul") is True
        # id2 should be evicted (oldest after id1 access)
        assert await small_cache_store.is_completed(id2, "consul") is False
        # Others remain
        assert await small_cache_store.is_completed(id3, "consul") is True
        assert await small_cache_store.is_completed(id4, "consul") is True

    @pytest.mark.asyncio
    async def test_update_existing_entry_moves_to_end(
        self,
        small_cache_store: InMemoryEffectIdempotencyStore,
    ) -> None:
        """Test that updating an entry moves it to end of LRU."""
        id1, id2, id3, id4 = uuid4(), uuid4(), uuid4(), uuid4()

        # Add 3 entries
        await small_cache_store.mark_completed(id1, "consul")
        await small_cache_store.mark_completed(id2, "consul")
        await small_cache_store.mark_completed(id3, "consul")

        # Update id1 with another backend (moves to end)
        await small_cache_store.mark_completed(id1, "postgres")

        # Add 4th entry - should evict id2 (now oldest)
        await small_cache_store.mark_completed(id4, "consul")

        # id1 should remain with both backends
        assert await small_cache_store.is_completed(id1, "consul") is True
        assert await small_cache_store.is_completed(id1, "postgres") is True
        # id2 should be evicted
        assert await small_cache_store.is_completed(id2, "consul") is False


# -----------------------------------------------------------------------------
# Test TTL Expiration
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestTTLExpiration:
    """Test TTL-based expiration of stale entries.

    Uses mocked time.monotonic() to test TTL behavior without slow sleeps.
    """

    @pytest.mark.asyncio
    async def test_expired_entry_returns_not_completed(
        self,
        short_ttl_store: InMemoryEffectIdempotencyStore,
    ) -> None:
        """Test that expired entries return False for is_completed."""
        correlation_id = uuid4()

        # Record the start time
        start_time = time.monotonic()

        with patch(
            "omnibase_infra.nodes.effects.store_effect_idempotency_inmemory.time.monotonic"
        ) as mock_monotonic:
            # Initial time
            mock_monotonic.return_value = start_time

            # Mark as completed
            await short_ttl_store.mark_completed(correlation_id, "consul")
            assert await short_ttl_store.is_completed(correlation_id, "consul") is True

            # Advance time past TTL (1 second TTL + some buffer)
            mock_monotonic.return_value = start_time + 1.5

            # Should return False (expired)
            assert await short_ttl_store.is_completed(correlation_id, "consul") is False

    @pytest.mark.asyncio
    async def test_expired_entry_returns_empty_set(
        self,
        short_ttl_store: InMemoryEffectIdempotencyStore,
    ) -> None:
        """Test that get_completed_backends returns empty set for expired entries."""
        correlation_id = uuid4()
        start_time = time.monotonic()

        with patch(
            "omnibase_infra.nodes.effects.store_effect_idempotency_inmemory.time.monotonic"
        ) as mock_monotonic:
            mock_monotonic.return_value = start_time

            # Mark backends
            await short_ttl_store.mark_completed(correlation_id, "consul")
            await short_ttl_store.mark_completed(correlation_id, "postgres")

            # Advance time past TTL
            mock_monotonic.return_value = start_time + 1.5

            # Should return empty set
            completed = await short_ttl_store.get_completed_backends(correlation_id)
            assert completed == set()

    @pytest.mark.asyncio
    async def test_cleanup_expired_removes_entries(
        self,
        short_ttl_store: InMemoryEffectIdempotencyStore,
    ) -> None:
        """Test that cleanup_expired removes stale entries."""
        start_time = time.monotonic()

        with patch(
            "omnibase_infra.nodes.effects.store_effect_idempotency_inmemory.time.monotonic"
        ) as mock_monotonic:
            mock_monotonic.return_value = start_time

            # Add entries
            for _ in range(5):
                await short_ttl_store.mark_completed(uuid4(), "consul")

            assert await short_ttl_store.get_cache_size() == 5

            # Advance time past TTL
            mock_monotonic.return_value = start_time + 1.5

            # Force cleanup
            removed = await short_ttl_store.cleanup_expired()

            # All should be removed
            assert removed == 5
            assert await short_ttl_store.get_cache_size() == 0

    @pytest.mark.asyncio
    async def test_cleanup_only_removes_expired(
        self,
    ) -> None:
        """Test that cleanup only removes expired entries, not fresh ones."""
        config = ModelEffectIdempotencyConfig(
            max_cache_size=100,
            cache_ttl_seconds=10.0,  # 10 second TTL
            cleanup_interval_seconds=60.0,  # High to avoid auto-cleanup
        )
        store = InMemoryEffectIdempotencyStore(config=config)
        start_time = time.monotonic()

        with patch(
            "omnibase_infra.nodes.effects.store_effect_idempotency_inmemory.time.monotonic"
        ) as mock_monotonic:
            mock_monotonic.return_value = start_time

            # Add old entry
            old_id = uuid4()
            await store.mark_completed(old_id, "consul")

            # Advance time by 8 seconds (still within TTL)
            mock_monotonic.return_value = start_time + 8.0

            # Add new entry
            new_id = uuid4()
            await store.mark_completed(new_id, "consul")

            # Advance time to expire old entry but not new (12 seconds total)
            # Old entry: 12 seconds old (> 10 second TTL) -> expired
            # New entry: 4 seconds old (< 10 second TTL) -> valid
            mock_monotonic.return_value = start_time + 12.0

            # Force cleanup
            removed = await store.cleanup_expired()

            # Only old entry should be removed
            assert removed == 1
            assert await store.is_completed(old_id, "consul") is False
            assert await store.is_completed(new_id, "consul") is True


# -----------------------------------------------------------------------------
# Test Configuration
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestConfiguration:
    """Test configuration options."""

    def test_default_config_values(self) -> None:
        """Test that default config has expected values."""
        config = ModelEffectIdempotencyConfig()
        assert config.max_cache_size == 10000
        assert config.cache_ttl_seconds == 3600.0
        assert config.cleanup_interval_seconds == 300.0

    def test_config_is_frozen(self) -> None:
        """Test that config is immutable."""
        config = ModelEffectIdempotencyConfig()
        with pytest.raises(Exception):  # Pydantic raises ValidationError
            config.max_cache_size = 5000  # type: ignore[misc]

    def test_store_uses_config(self) -> None:
        """Test that store respects config values."""
        config = ModelEffectIdempotencyConfig(
            max_cache_size=50,
            cache_ttl_seconds=120.0,
        )
        store = InMemoryEffectIdempotencyStore(config=config)

        assert store.max_cache_size == 50
        assert store.cache_ttl_seconds == 120.0


# -----------------------------------------------------------------------------
# Test Concurrent Access
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestConcurrentAccess:
    """Test thread-safety under concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_writes_are_safe(
        self,
        default_store: InMemoryEffectIdempotencyStore,
    ) -> None:
        """Test that concurrent writes don't corrupt state."""

        # Create many tasks that write concurrently
        async def write_entry(store: InMemoryEffectIdempotencyStore, n: int) -> None:
            cid = uuid4()
            await store.mark_completed(cid, f"backend_{n}")
            # Verify our write is visible
            assert await store.is_completed(cid, f"backend_{n}")

        # Run 100 concurrent writes
        tasks = [write_entry(default_store, i) for i in range(100)]
        await asyncio.gather(*tasks)

        # All entries should be present
        assert await default_store.get_cache_size() == 100

    @pytest.mark.asyncio
    async def test_concurrent_reads_and_writes(
        self,
        default_store: InMemoryEffectIdempotencyStore,
    ) -> None:
        """Test concurrent reads and writes don't cause issues."""
        shared_id = uuid4()
        await default_store.mark_completed(shared_id, "consul")

        read_count = 0

        async def reader(store: InMemoryEffectIdempotencyStore) -> None:
            nonlocal read_count
            for _ in range(50):
                result = await store.is_completed(shared_id, "consul")
                if result:
                    read_count += 1
                await asyncio.sleep(0)  # Yield to other tasks

        async def writer(store: InMemoryEffectIdempotencyStore) -> None:
            for i in range(50):
                await store.mark_completed(uuid4(), f"backend_{i}")
                await asyncio.sleep(0)  # Yield to other tasks

        # Run readers and writers concurrently
        await asyncio.gather(
            reader(default_store),
            reader(default_store),
            writer(default_store),
            writer(default_store),
        )

        # Reader should have seen the entry at least some times
        assert read_count > 0


# -----------------------------------------------------------------------------
# Test Memory Bounds
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestMemoryBounds:
    """Test that memory usage is bounded."""

    @pytest.mark.asyncio
    async def test_cache_does_not_exceed_max_size(self) -> None:
        """Test that cache never exceeds configured max_size."""
        config = ModelEffectIdempotencyConfig(max_cache_size=10)
        store = InMemoryEffectIdempotencyStore(config=config)

        # Add more than max_size entries
        for _ in range(50):
            await store.mark_completed(uuid4(), "consul")

        # Cache should be at max_size
        assert await store.get_cache_size() == 10

    @pytest.mark.asyncio
    async def test_repeated_updates_same_entry(self) -> None:
        """Test that updating same entry repeatedly doesn't grow cache."""
        config = ModelEffectIdempotencyConfig(max_cache_size=10)
        store = InMemoryEffectIdempotencyStore(config=config)

        cid = uuid4()

        # Update same entry many times with different backends
        for i in range(100):
            await store.mark_completed(cid, f"backend_{i}")

        # Cache should have exactly 1 entry
        assert await store.get_cache_size() == 1

        # Entry should have all backends
        completed = await store.get_completed_backends(cid)
        assert len(completed) == 100
