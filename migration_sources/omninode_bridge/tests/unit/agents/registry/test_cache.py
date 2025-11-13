"""Unit tests for CacheManager."""

import time

import pytest

from omninode_bridge.agents.registry.cache import CacheManager


@pytest.fixture
def cache():
    """Create CacheManager for tests."""
    return CacheManager(max_size=10, ttl_seconds=1)


class TestCacheBasicOperations:
    """Tests for basic cache operations."""

    def test_set_and_get(self, cache):
        """Test setting and getting values."""
        cache.set("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"

    def test_get_nonexistent_key(self, cache):
        """Test getting nonexistent key returns None."""
        result = cache.get("nonexistent")
        assert result is None

    def test_update_existing_key(self, cache):
        """Test updating existing key."""
        cache.set("key1", "value1")
        cache.set("key1", "value2")
        result = cache.get("key1")
        assert result == "value2"

    def test_invalidate_key(self, cache):
        """Test invalidating specific key."""
        cache.set("key1", "value1")
        cache.invalidate("key1")
        result = cache.get("key1")
        assert result is None

    def test_invalidate_all(self, cache):
        """Test invalidating all keys."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.invalidate_all()

        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestCacheTTL:
    """Tests for TTL expiration."""

    def test_ttl_expiration(self, cache):
        """Test values expire after TTL."""
        cache.set("key1", "value1")

        # Value should be available immediately
        assert cache.get("key1") == "value1"

        # Wait for TTL to expire (1 second + buffer)
        time.sleep(1.1)

        # Value should be expired
        assert cache.get("key1") is None

    def test_ttl_not_expired(self, cache):
        """Test values don't expire before TTL."""
        cache.set("key1", "value1")

        # Value should be available within TTL
        time.sleep(0.5)
        assert cache.get("key1") == "value1"


class TestCacheLRU:
    """Tests for LRU eviction."""

    def test_lru_eviction(self, cache):
        """Test LRU eviction when max_size exceeded."""
        # Fill cache to max_size
        for i in range(10):
            cache.set(f"key{i}", f"value{i}")

        # All keys should be present
        for i in range(10):
            assert cache.get(f"key{i}") == f"value{i}"

        # Add one more key (should evict oldest)
        cache.set("key10", "value10")

        # First key should be evicted (LRU)
        assert cache.get("key0") is None

        # Newest key should be present
        assert cache.get("key10") == "value10"

    def test_lru_move_to_end(self, cache):
        """Test accessing key moves it to end (most recently used)."""
        # Add keys
        for i in range(10):
            cache.set(f"key{i}", f"value{i}")

        # Access first key (move to end)
        cache.get("key0")

        # Add one more key
        cache.set("key10", "value10")

        # key0 should still be present (was moved to end)
        assert cache.get("key0") == "value0"

        # key1 should be evicted (was LRU)
        assert cache.get("key1") is None


class TestCacheStatistics:
    """Tests for cache statistics."""

    def test_cache_hits(self, cache):
        """Test cache hit counting."""
        cache.set("key1", "value1")

        # First access (hit)
        cache.get("key1")

        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 0

    def test_cache_misses(self, cache):
        """Test cache miss counting."""
        # Access nonexistent key
        cache.get("nonexistent")

        stats = cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 1

    def test_hit_rate_calculation(self, cache):
        """Test hit rate calculation."""
        cache.set("key1", "value1")

        # 5 hits
        for _ in range(5):
            cache.get("key1")

        # 5 misses
        for i in range(5):
            cache.get(f"nonexistent{i}")

        stats = cache.get_stats()
        assert stats.hit_rate == 0.5  # 50% hit rate

    def test_eviction_counting(self, cache):
        """Test eviction counting."""
        # Fill cache beyond max_size
        for i in range(15):
            cache.set(f"key{i}", f"value{i}")

        stats = cache.get_stats()
        assert stats.evictions == 5  # 15 - 10 = 5 evictions

    def test_cache_size(self, cache):
        """Test cache size tracking."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        stats = cache.get_stats()
        assert stats.size == 2
        assert stats.max_size == 10


class TestCacheEdgeCases:
    """Tests for edge cases."""

    def test_empty_cache_stats(self, cache):
        """Test statistics for empty cache."""
        stats = cache.get_stats()
        assert stats.size == 0
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0
        assert stats.evictions == 0

    def test_clear_cache(self, cache):
        """Test clearing cache."""
        cache.set("key1", "value1")
        cache.get("key1")  # Generate hit

        cache.clear()

        assert len(cache) == 0
        stats = cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0

    def test_contains_check(self, cache):
        """Test __contains__ check."""
        cache.set("key1", "value1")
        assert "key1" in cache
        assert "nonexistent" not in cache

    def test_contains_expired(self, cache):
        """Test __contains__ returns False for expired keys."""
        cache.set("key1", "value1")
        assert "key1" in cache

        # Wait for expiration
        time.sleep(1.1)
        assert "key1" not in cache

    def test_len(self, cache):
        """Test __len__ returns cache size."""
        assert len(cache) == 0

        cache.set("key1", "value1")
        assert len(cache) == 1

        cache.set("key2", "value2")
        assert len(cache) == 2


class TestCachePerformance:
    """Performance tests for cache."""

    def test_get_performance(self, cache):
        """Test get operation meets <5ms target."""
        cache.set("key1", "value1")

        start = time.time()
        cache.get("key1")
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < 5

    def test_set_performance(self, cache):
        """Test set operation meets <10ms target."""
        start = time.time()
        cache.set("key1", "value1")
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < 10

    def test_large_cache_performance(self):
        """Test performance with large cache."""
        large_cache = CacheManager(max_size=1000, ttl_seconds=60)

        # Fill cache
        for i in range(1000):
            large_cache.set(f"key{i}", f"value{i}")

        # Test get performance
        start = time.time()
        large_cache.get("key500")
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < 5  # Should still be <5ms
