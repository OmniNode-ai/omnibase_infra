"""
TTL Cache Implementation

Provides a thread-safe TTL (Time To Live) cache with automatic cleanup,
memory monitoring, and configurable eviction policies.
"""

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

from ..config.registry_config import get_registry_config

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with TTL and access tracking."""

    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl_seconds: float = 3600.0

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return (time.time() - self.created_at) > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at

    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheMetrics:
    """Metrics for cache operations."""

    total_operations: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expired_cleanups: int = 0
    manual_cleanups: int = 0
    memory_usage_bytes: int = 0
    max_size_reached: int = 0
    cleanup_duration_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as percentage."""
        if self.total_operations == 0:
            return 0.0
        return (self.hits / self.total_operations) * 100.0

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate as percentage."""
        if self.total_operations == 0:
            return 0.0
        return (self.misses / self.total_operations) * 100.0

    @property
    def memory_usage_mb(self) -> float:
        """Calculate memory usage in megabytes."""
        return self.memory_usage_bytes / (1024 * 1024)


class TTLCache:
    """
    Thread-safe TTL cache with automatic cleanup and memory monitoring.

    Features:
    - Time-based expiration with configurable TTL
    - Size-based eviction with LRU policy
    - Automatic cleanup in background asyncio task (non-blocking)
    - Memory usage monitoring and alerts
    - Comprehensive metrics tracking
    - Thread-safe operations

    Background Cleanup:
    - Uses asyncio.sleep() for non-blocking periodic cleanup
    - Properly integrates with async test cleanup without hanging
    - Cleanup task responds quickly to stop() calls via _cleanup_active flag
    - No threading.Event blocking issues in async contexts
    """

    def __init__(
        self,
        name: str,
        max_size: int = 10000,
        default_ttl_seconds: float = 3600.0,
        cleanup_interval_seconds: float = 300.0,
        enable_background_cleanup: bool = True,
        environment: str = "development",
    ):
        """
        Initialize TTL cache.

        Args:
            name: Cache name for logging and metrics
            max_size: Maximum number of entries
            default_ttl_seconds: Default TTL for entries
            cleanup_interval_seconds: Interval for background cleanup
            enable_background_cleanup: Enable background cleanup task
            environment: Environment name for configuration
        """
        self.name = name
        self._max_size = max_size
        self._default_ttl_seconds = default_ttl_seconds
        self._cleanup_interval_seconds = cleanup_interval_seconds
        self._enable_background_cleanup = enable_background_cleanup
        self._environment = environment

        # Thread-safe storage
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._access_order = OrderedDict()  # For LRU eviction

        # Background cleanup
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_active = False

        # Metrics
        self._metrics = CacheMetrics()

        # Get registry configuration for memory monitoring
        try:
            registry_config = get_registry_config(environment)
            self._memory_warning_threshold_mb = (
                registry_config.memory_warning_threshold_mb
            )
            self._memory_critical_threshold_mb = (
                registry_config.memory_critical_threshold_mb
            )
        except (ImportError, AttributeError, KeyError, ValueError) as e:
            # Fallback defaults if config not available
            logger.debug(f"Using default memory thresholds, config unavailable: {e}")
            self._memory_warning_threshold_mb = 256.0
            self._memory_critical_threshold_mb = 512.0

        logger.info(
            f"TTL cache '{name}' initialized: max_size={max_size}, ttl={default_ttl_seconds}s, cleanup_interval={cleanup_interval_seconds}s"
        )

        # Start background cleanup if enabled
        if self._enable_background_cleanup:
            self._start_background_cleanup()

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value if found and not expired, None otherwise
        """
        with self._lock:
            self._metrics.total_operations += 1

            if key not in self._cache:
                self._metrics.misses += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired:
                self._remove_entry(key)
                self._metrics.misses += 1
                self._metrics.expired_cleanups += 1
                logger.debug(f"Cache '{self.name}': expired entry removed: {key}")
                return None

            # Update access tracking
            entry.touch()
            self._update_access_order(key)

            self._metrics.hits += 1
            return entry.value

    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        """
        Put value in cache with optional TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Optional TTL override
        """
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds
        current_time = time.time()

        with self._lock:
            # Check if entry already exists
            if key in self._cache:
                old_entry = self._cache[key]
                # Update existing entry
                new_entry = CacheEntry(
                    value=value,
                    created_at=current_time,
                    last_accessed=current_time,
                    access_count=old_entry.access_count + 1,
                    ttl_seconds=ttl,
                )
                self._cache[key] = new_entry
                self._update_access_order(key)
            else:
                # Add new entry
                new_entry = CacheEntry(
                    value=value,
                    created_at=current_time,
                    last_accessed=current_time,
                    access_count=1,
                    ttl_seconds=ttl,
                )

                # Check size limit and evict if necessary
                if len(self._cache) >= self._max_size:
                    self._evict_lru()

                self._cache[key] = new_entry
                self._update_access_order(key)

                # Update max size reached metric
                self._metrics.max_size_reached = max(
                    self._metrics.max_size_reached, len(self._cache)
                )

            self._metrics.total_operations += 1

        # Check memory usage
        self._check_memory_usage()

    def remove(self, key: str) -> bool:
        """
        Remove entry from cache.

        Args:
            key: Cache key

        Returns:
            True if entry was removed, False if not found
        """
        with self._lock:
            if key not in self._cache:
                return False

            self._remove_entry(key)
            return True

    def clear(self) -> int:
        """
        Clear all entries from cache.

        Returns:
            Number of entries removed
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._access_order.clear()
            logger.info(f"Cache '{self.name}': cleared {count} entries")
            return count

    def cleanup_expired(self) -> int:
        """
        Manually cleanup expired entries.

        Returns:
            Number of expired entries removed
        """
        start_time = time.time()

        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired
            ]

            for key in expired_keys:
                self._remove_entry(key)

            self._metrics.manual_cleanups += 1
            self._metrics.expired_cleanups += len(expired_keys)

        cleanup_duration_ms = (time.time() - start_time) * 1000
        self._metrics.cleanup_duration_ms = cleanup_duration_ms

        if expired_keys:
            logger.info(
                f"Cache '{self.name}': cleanup removed {len(expired_keys)} expired entries in {cleanup_duration_ms:.2f}ms"
            )

        return len(expired_keys)

    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    def keys(self) -> set[str]:
        """Get all cache keys."""
        with self._lock:
            return set(self._cache.keys())

    def get_metrics(self) -> CacheMetrics:
        """Get cache metrics."""
        with self._lock:
            # Update memory usage estimate
            self._metrics.memory_usage_bytes = self._estimate_memory_usage()
            return self._metrics

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive cache status."""
        with self._lock:
            return {
                "name": self.name,
                "size": len(self._cache),
                "max_size": self._max_size,
                "default_ttl_seconds": self._default_ttl_seconds,
                "cleanup_interval_seconds": self._cleanup_interval_seconds,
                "background_cleanup_active": self._cleanup_active,
                "metrics": {
                    "total_operations": self._metrics.total_operations,
                    "hits": self._metrics.hits,
                    "misses": self._metrics.misses,
                    "hit_rate": round(self._metrics.hit_rate, 2),
                    "miss_rate": round(self._metrics.miss_rate, 2),
                    "evictions": self._metrics.evictions,
                    "expired_cleanups": self._metrics.expired_cleanups,
                    "manual_cleanups": self._metrics.manual_cleanups,
                    "memory_usage_bytes": self._estimate_memory_usage(),
                    "memory_usage_mb": round(
                        self._estimate_memory_usage() / (1024 * 1024), 2
                    ),
                    "max_size_reached": self._metrics.max_size_reached,
                    "cleanup_duration_ms": round(self._metrics.cleanup_duration_ms, 2),
                },
            }

    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache and access order."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            del self._access_order[key]

    def _update_access_order(self, key: str) -> None:
        """Update access order for LRU."""
        if key in self._access_order:
            del self._access_order[key]
        self._access_order[key] = time.time()

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._access_order:
            lru_key = next(iter(self._access_order))
            self._remove_entry(lru_key)
            self._metrics.evictions += 1
            logger.debug(f"Cache '{self.name}': evicted LRU entry: {lru_key}")

    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        # Rough estimation - actual usage may vary
        estimated_size = 0
        for key, entry in self._cache.items():
            # Key size + value size estimate + metadata overhead
            estimated_size += len(key.encode("utf-8")) * 2  # Unicode
            estimated_size += 100  # Value estimate (varies by type)
            estimated_size += 64  # Entry metadata

        return estimated_size

    def _check_memory_usage(self) -> None:
        """Check memory usage and log warnings if needed."""
        memory_usage_mb = self._estimate_memory_usage() / (1024 * 1024)

        if memory_usage_mb >= self._memory_critical_threshold_mb:
            logger.error(
                f"Cache '{self.name}' memory usage CRITICAL: {memory_usage_mb:.2f}MB "
                f"(threshold: {self._memory_critical_threshold_mb:.2f}MB)"
            )
            # Trigger immediate cleanup
            self.cleanup_expired()
        elif memory_usage_mb >= self._memory_warning_threshold_mb:
            logger.warning(
                f"Cache '{self.name}' memory usage WARNING: {memory_usage_mb:.2f}MB "
                f"(threshold: {self._memory_warning_threshold_mb:.2f}MB)"
            )

    def _start_background_cleanup(self) -> None:
        """Start background cleanup task."""
        try:
            # Try to get current event loop
            loop = asyncio.get_running_loop()
            if self._cleanup_task is None or self._cleanup_task.done():
                self._cleanup_task = loop.create_task(self._background_cleanup_loop())
                logger.debug(f"Cache '{self.name}': started background cleanup task")
        except RuntimeError:
            # No running event loop - background cleanup will start when needed
            logger.debug(
                f"Cache '{self.name}': no event loop available, background cleanup deferred"
            )
            self._cleanup_task = None

    async def _background_cleanup_loop(self) -> None:
        """
        Background cleanup loop.

        Uses asyncio.sleep() instead of threading.Event to ensure proper
        async integration and clean test teardown without blocking.
        """
        self._cleanup_active = True
        logger.debug(f"Cache '{self.name}': background cleanup loop started")

        try:
            while self._cleanup_active:
                try:
                    # Sleep for cleanup interval, checking stop flag periodically
                    # This ensures responsive shutdown without blocking
                    await asyncio.sleep(self._cleanup_interval_seconds)

                    # Check if we should stop before performing cleanup
                    if not self._cleanup_active:
                        break

                    # Perform cleanup
                    expired_count = self.cleanup_expired()
                    if expired_count > 0:
                        logger.debug(
                            f"Cache '{self.name}': background cleanup removed {expired_count} expired entries"
                        )

                except (RuntimeError, ValueError, KeyError) as e:
                    # Expected cleanup errors
                    logger.warning(
                        f"Cache '{self.name}': background cleanup error: {e}"
                    )
                    await asyncio.sleep(60)  # Wait before retrying
                except Exception as e:
                    # Unexpected errors - log with full context
                    logger.error(
                        f"Cache '{self.name}': unexpected background cleanup error: {e}",
                        exc_info=True,
                    )
                    await asyncio.sleep(60)  # Wait before retrying

        except asyncio.CancelledError:
            logger.debug(f"Cache '{self.name}': background cleanup cancelled")
        finally:
            self._cleanup_active = False
            logger.debug(f"Cache '{self.name}': background cleanup loop stopped")

    async def stop(self) -> None:
        """
        Stop background cleanup and cleanup resources.

        Sets cleanup_active flag to False and cancels the cleanup task,
        ensuring clean shutdown without blocking.
        """
        self._cleanup_active = False

        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Cache '{self.name}': stopped")

    def __len__(self) -> int:
        """Get cache size."""
        return self.size()

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self.get(key) is not None


# Factory function for creating TTL caches with registry config
def create_ttl_cache(
    name: str,
    environment: str = "development",
    max_size: Optional[int] = None,
    ttl_seconds: Optional[float] = None,
    cleanup_interval_seconds: Optional[float] = None,
) -> TTLCache:
    """
    Create TTL cache with registry configuration.

    Args:
        name: Cache name
        environment: Environment name
        max_size: Optional max size override
        ttl_seconds: Optional TTL override
        cleanup_interval_seconds: Optional cleanup interval override

    Returns:
        Configured TTLCache instance
    """
    registry_config = get_registry_config(environment)

    # Extract configuration from registry config
    max_size_val = max_size or registry_config.max_tracked_offsets
    ttl_seconds_val = ttl_seconds or registry_config.offset_cache_ttl_seconds
    cleanup_interval_val = (
        cleanup_interval_seconds or registry_config.offset_cleanup_interval_seconds
    )

    # Disable background cleanup for test environments to prevent hanging fixtures
    enable_cleanup = environment != "test"

    return TTLCache(
        name=name,
        max_size=max_size_val,
        default_ttl_seconds=ttl_seconds_val,
        cleanup_interval_seconds=cleanup_interval_val,
        enable_background_cleanup=enable_cleanup,
        environment=environment,
    )
