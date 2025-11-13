"""LRU cache with TTL for agent discovery results."""

import sys
import time
from collections import OrderedDict
from typing import Any, Optional

from pydantic import BaseModel, Field


class CacheEntry(BaseModel):
    """Cache entry with value and timestamp."""

    value: Any
    timestamp: float
    size_bytes: int = Field(default=0, ge=0)

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True


class CacheStats(BaseModel):
    """Cache statistics."""

    size: int = Field(ge=0)
    max_size: int = Field(ge=0)
    hits: int = Field(default=0, ge=0)
    misses: int = Field(default=0, ge=0)
    hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    evictions: int = Field(default=0, ge=0)
    memory_bytes: int = Field(default=0, ge=0)
    memory_mb: float = Field(default=0.0, ge=0.0)
    memory_limit_evictions: int = Field(default=0, ge=0)


class CacheManager:
    """
    LRU cache with TTL for agent discovery results.

    Performance Targets:
    - Get: <5ms (cache hit)
    - Set: <10ms
    - Cache hit rate: 85-95%

    Eviction Policy:
    - LRU (Least Recently Used) when max_size exceeded
    - TTL-based expiration (default: 300s = 5min)

    Example:
        ```python
        cache = CacheManager(max_size=1000, ttl_seconds=300)

        # Set value
        cache.set("task_key", {"agent_id": "agent1", "confidence": 0.85})

        # Get value (returns None if expired or not found)
        result = cache.get("task_key")

        # Invalidate specific entry
        cache.invalidate("task_key")

        # Get statistics
        stats = cache.get_stats()
        print(f"Hit rate: {stats.hit_rate:.2%}")
        ```
    """

    def __init__(
        self, max_size: int = 1000, ttl_seconds: int = 300, max_memory_mb: float = 100.0
    ) -> None:
        """
        Initialize cache manager.

        Args:
            max_size: Maximum number of cache entries (default: 1000)
            ttl_seconds: Time-to-live in seconds (default: 300s = 5min)
            max_memory_mb: Maximum memory usage in MB (default: 100.0)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)

        # LRU cache using OrderedDict
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.memory_limit_evictions = 0
        self.total_memory_bytes = 0

    @staticmethod
    def _estimate_size(value: Any) -> int:
        """
        Estimate object size in bytes.

        Uses sys.getsizeof() with basic handling for nested structures.

        Args:
            value: Object to estimate

        Returns:
            Estimated size in bytes
        """
        try:
            # Use sys.getsizeof for basic estimation
            size = sys.getsizeof(value)

            # Add size of nested structures if dict or list
            if isinstance(value, dict):
                size += sum(
                    sys.getsizeof(k) + sys.getsizeof(v) for k, v in value.items()
                )
            elif isinstance(value, (list, tuple)):
                size += sum(sys.getsizeof(item) for item in value)

            return size
        except Exception:
            # Fallback to conservative estimate
            return 1024  # 1KB default

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Performance Target: <5ms

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        if key not in self.cache:
            self.misses += 1
            return None

        entry = self.cache[key]

        # Check TTL
        if time.time() - entry.timestamp > self.ttl_seconds:
            # Expired - remove
            self.total_memory_bytes -= entry.size_bytes
            del self.cache[key]
            self.misses += 1
            return None

        # Cache hit - move to end (most recently used)
        self.cache.move_to_end(key)
        self.hits += 1

        return entry.value

    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.

        Eviction triggers:
        - Cache reaches max_size entries
        - Cache exceeds max_memory_mb
        - Entry exceeds TTL

        Performance Target: <10ms

        Args:
            key: Cache key
            value: Value to cache
        """
        # Estimate value size
        value_size = self._estimate_size(value)

        # Check if key exists
        if key in self.cache:
            # Update existing - adjust memory tracking
            old_entry = self.cache[key]
            self.total_memory_bytes -= old_entry.size_bytes
            self.cache[key] = CacheEntry(
                value=value, timestamp=time.time(), size_bytes=value_size
            )
            self.total_memory_bytes += value_size
            self.cache.move_to_end(key)
        else:
            # New entry - check size limit
            if len(self.cache) >= self.max_size:
                # Evict LRU (first item)
                evicted_key, evicted_entry = self.cache.popitem(last=False)
                self.total_memory_bytes -= evicted_entry.size_bytes
                self.evictions += 1

            # Check memory limit - evict LRU entries until under limit
            while (
                self.cache
                and self.total_memory_bytes + value_size > self.max_memory_bytes
            ):
                evicted_key, evicted_entry = self.cache.popitem(last=False)
                self.total_memory_bytes -= evicted_entry.size_bytes
                self.evictions += 1
                self.memory_limit_evictions += 1

            self.cache[key] = CacheEntry(
                value=value, timestamp=time.time(), size_bytes=value_size
            )
            self.total_memory_bytes += value_size

    def invalidate(self, key: str) -> None:
        """
        Invalidate specific cache entry.

        Args:
            key: Cache key to invalidate
        """
        if key in self.cache:
            entry = self.cache[key]
            self.total_memory_bytes -= entry.size_bytes
            del self.cache[key]

    def invalidate_all(self) -> None:
        """Invalidate entire cache."""
        self.cache.clear()
        self.total_memory_bytes = 0

    def get_stats(self) -> CacheStats:
        """
        Get cache statistics.

        Returns:
            CacheStats with hit rate and other metrics
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) if total_requests > 0 else 0.0

        return CacheStats(
            size=len(self.cache),
            max_size=self.max_size,
            hits=self.hits,
            misses=self.misses,
            hit_rate=hit_rate,
            evictions=self.evictions,
            memory_bytes=self.total_memory_bytes,
            memory_mb=self.total_memory_bytes / (1024 * 1024),
            memory_limit_evictions=self.memory_limit_evictions,
        )

    def clear(self) -> None:
        """Clear cache and reset metrics."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.memory_limit_evictions = 0
        self.total_memory_bytes = 0

    def __len__(self) -> int:
        """Get number of entries in cache."""
        return len(self.cache)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        if key not in self.cache:
            return False

        # Check TTL
        entry = self.cache[key]
        if time.time() - entry.timestamp > self.ttl_seconds:
            # Expired - remove and update memory tracking
            self.total_memory_bytes -= entry.size_bytes
            del self.cache[key]
            return False

        return True
