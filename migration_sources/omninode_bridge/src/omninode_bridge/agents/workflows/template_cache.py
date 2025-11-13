"""
LRU Template Cache for high-performance template management.

Provides thread-safe LRU cache with:
- Least Recently Used eviction policy
- Cache hit rate tracking (target: 85-95%)
- Configurable cache size (default: 100 templates)
- Thread-safe operations using RLock
- Template invalidation support

Performance Targets:
- Cached lookup: <1ms
- Cache hit rate: 85-95%
- Memory per template: 5-20KB
"""

import logging
import threading
import time
from collections import OrderedDict
from typing import Optional

from omninode_bridge.agents.workflows.template_models import (
    Template,
    TemplateCacheStats,
)

logger = logging.getLogger(__name__)


class TemplateLRUCache:
    """
    Thread-safe LRU cache for templates.

    Design:
    - OrderedDict for O(1) get/put with LRU tracking
    - Threading.RLock for thread-safe operations
    - Move-to-end for LRU updates on access
    - Automatic eviction when cache is full

    Performance:
    - get(): O(1) with <1ms target
    - put(): O(1) with automatic eviction
    - Thread-safe with minimal lock contention

    Example:
        ```python
        cache = TemplateLRUCache(max_size=100)

        # Add template
        cache.put(template)

        # Get template (updates LRU)
        template = cache.get("template_id")

        # Get statistics
        stats = cache.get_stats()
        print(f"Hit rate: {stats.hit_rate:.2%}")
        ```
    """

    def __init__(self, max_size: int = 100, max_memory_mb: float = 100.0) -> None:
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of templates to cache (default: 100)
            max_memory_mb: Maximum memory usage in MB (default: 100.0)
        """
        self._cache: OrderedDict[str, Template] = OrderedDict()
        self._max_size = max_size
        self._max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self._lock = threading.RLock()

        # Performance metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._total_size_bytes = 0
        self._memory_limit_evictions = 0

        # Timing metrics
        self._get_times: list[float] = []
        self._put_times: list[float] = []
        self._max_timing_samples = 1000

        logger.info(
            f"TemplateLRUCache initialized with max_size={max_size}, "
            f"max_memory={max_memory_mb:.1f}MB"
        )

    def get(self, template_id: str) -> Optional[Template]:
        """
        Get template from cache (updates LRU on access).

        Performance Target: <1ms

        Args:
            template_id: Template identifier

        Returns:
            Template if found, None otherwise

        Example:
            ```python
            template = cache.get("node_effect_v1")
            if template:
                print(f"Cache hit: {template.template_id}")
            else:
                print("Cache miss")
            ```
        """
        start_time = time.perf_counter()

        with self._lock:
            # Check if template exists
            if template_id not in self._cache:
                self._misses += 1
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self._record_get_time(elapsed_ms)
                return None

            # Cache hit - move to end (most recently used)
            self._cache.move_to_end(template_id)
            self._hits += 1

            template = self._cache[template_id]
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._record_get_time(elapsed_ms)

            return template

    def put(self, template: Template) -> None:
        """
        Add template to cache (evicts LRU if full).

        Eviction triggers:
        - Cache reaches max_size entries
        - Cache exceeds max_memory_mb

        Performance Target: <2ms (including eviction)

        Args:
            template: Template to cache

        Example:
            ```python
            template = Template(
                template_id="node_effect_v1",
                template_type=TemplateType.EFFECT,
                content="...",
                metadata=TemplateMetadata()
            )
            cache.put(template)
            ```
        """
        start_time = time.perf_counter()

        with self._lock:
            # If template already exists, remove old entry first
            if template.template_id in self._cache:
                old_template = self._cache[template.template_id]
                self._total_size_bytes -= old_template.size_bytes
                del self._cache[template.template_id]

            # Check if cache is full (entry count) - evict LRU if needed
            if len(self._cache) >= self._max_size:
                lru_id, lru_template = self._cache.popitem(last=False)
                self._total_size_bytes -= lru_template.size_bytes
                self._evictions += 1
                logger.debug(
                    f"Evicted LRU template (size limit): {lru_id} "
                    f"(size: {lru_template.size_bytes}B)"
                )

            # Check memory limit - evict LRU entries until under limit
            while (
                self._cache
                and self._total_size_bytes + template.size_bytes
                > self._max_memory_bytes
            ):
                lru_id, lru_template = self._cache.popitem(last=False)
                self._total_size_bytes -= lru_template.size_bytes
                self._evictions += 1
                self._memory_limit_evictions += 1
                logger.debug(
                    f"Evicted LRU template (memory limit): {lru_id} "
                    f"(size: {lru_template.size_bytes}B, "
                    f"total: {self._total_size_bytes / (1024 * 1024):.2f}MB)"
                )

            # Add new template to end (most recently used)
            self._cache[template.template_id] = template
            self._total_size_bytes += template.size_bytes

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._record_put_time(elapsed_ms)

            logger.debug(
                f"Cached template: {template.template_id} "
                f"(size: {template.size_bytes}B, "
                f"cache: {len(self._cache)}/{self._max_size}, "
                f"memory: {self._total_size_bytes / (1024 * 1024):.2f}MB/"
                f"{self._max_memory_bytes / (1024 * 1024):.0f}MB)"
            )

    def invalidate(self, template_id: str) -> bool:
        """
        Invalidate (remove) template from cache.

        Args:
            template_id: Template identifier to invalidate

        Returns:
            True if template was found and removed, False otherwise

        Example:
            ```python
            if cache.invalidate("node_effect_v1"):
                print("Template invalidated")
            ```
        """
        with self._lock:
            if template_id not in self._cache:
                return False

            template = self._cache[template_id]
            self._total_size_bytes -= template.size_bytes
            del self._cache[template_id]

            logger.debug(f"Invalidated template: {template_id}")
            return True

    def clear(self) -> None:
        """
        Clear all cached templates.

        Example:
            ```python
            cache.clear()
            print("Cache cleared")
            ```
        """
        with self._lock:
            self._cache.clear()
            self._total_size_bytes = 0
            logger.info("Template cache cleared")

    def get_stats(self) -> TemplateCacheStats:
        """
        Get cache statistics.

        Returns:
            TemplateCacheStats with hit rate and cache metrics

        Example:
            ```python
            stats = cache.get_stats()
            print(f"Hit rate: {stats.hit_rate:.2%}")
            print(f"Cache size: {stats.current_size}/{stats.max_size}")
            print(f"Evictions: {stats.evictions}")
            print(f"Memory: {stats.total_size_bytes / (1024 * 1024):.2f}MB")
            ```
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return TemplateCacheStats(
                total_requests=total_requests,
                cache_hits=self._hits,
                cache_misses=self._misses,
                hit_rate=hit_rate,
                current_size=len(self._cache),
                max_size=self._max_size,
                evictions=self._evictions,
                total_size_bytes=self._total_size_bytes,
            )

    def get_hit_rate(self) -> float:
        """
        Calculate cache hit rate.

        Returns:
            Hit rate as float (0.0-1.0)

        Example:
            ```python
            hit_rate = cache.get_hit_rate()
            if hit_rate < 0.85:
                print(f"Warning: Low hit rate {hit_rate:.2%}")
            ```
        """
        with self._lock:
            total_requests = self._hits + self._misses
            return self._hits / total_requests if total_requests > 0 else 0.0

    def get_timing_stats(self) -> dict[str, float]:
        """
        Get timing statistics for cache operations.

        Returns:
            Dictionary with average, p50, p95, p99 timing metrics in milliseconds

        Example:
            ```python
            timing = cache.get_timing_stats()
            print(f"Avg get time: {timing['get_avg_ms']:.2f}ms")
            print(f"P99 get time: {timing['get_p99_ms']:.2f}ms")
            ```
        """
        with self._lock:
            stats = {}

            # Calculate get timing statistics
            if self._get_times:
                sorted_get_times = sorted(self._get_times)
                stats["get_avg_ms"] = sum(sorted_get_times) / len(sorted_get_times)
                stats["get_p50_ms"] = sorted_get_times[len(sorted_get_times) // 2]
                stats["get_p95_ms"] = sorted_get_times[
                    int(len(sorted_get_times) * 0.95)
                ]
                stats["get_p99_ms"] = sorted_get_times[
                    int(len(sorted_get_times) * 0.99)
                ]
            else:
                stats["get_avg_ms"] = 0.0
                stats["get_p50_ms"] = 0.0
                stats["get_p95_ms"] = 0.0
                stats["get_p99_ms"] = 0.0

            # Calculate put timing statistics
            if self._put_times:
                sorted_put_times = sorted(self._put_times)
                stats["put_avg_ms"] = sum(sorted_put_times) / len(sorted_put_times)
                stats["put_p50_ms"] = sorted_put_times[len(sorted_put_times) // 2]
                stats["put_p95_ms"] = sorted_put_times[
                    int(len(sorted_put_times) * 0.95)
                ]
                stats["put_p99_ms"] = sorted_put_times[
                    int(len(sorted_put_times) * 0.99)
                ]
            else:
                stats["put_avg_ms"] = 0.0
                stats["put_p50_ms"] = 0.0
                stats["put_p95_ms"] = 0.0
                stats["put_p99_ms"] = 0.0

            return stats

    def _record_get_time(self, elapsed_ms: float) -> None:
        """Record get operation timing (internal)."""
        self._get_times.append(elapsed_ms)
        if len(self._get_times) > self._max_timing_samples:
            self._get_times.pop(0)

    def _record_put_time(self, elapsed_ms: float) -> None:
        """Record put operation timing (internal)."""
        self._put_times.append(elapsed_ms)
        if len(self._put_times) > self._max_timing_samples:
            self._put_times.pop(0)

    def has(self, template_id: str) -> bool:
        """
        Check if template exists in cache (doesn't update LRU).

        Args:
            template_id: Template identifier

        Returns:
            True if template exists in cache, False otherwise

        Example:
            ```python
            if cache.has("node_effect_v1"):
                print("Template is cached")
            ```
        """
        with self._lock:
            return template_id in self._cache

    def size(self) -> int:
        """
        Get current cache size.

        Returns:
            Number of templates in cache

        Example:
            ```python
            print(f"Cache size: {cache.size()}")
            ```
        """
        with self._lock:
            return len(self._cache)

    def max_size(self) -> int:
        """
        Get maximum cache size.

        Returns:
            Maximum number of templates

        Example:
            ```python
            print(f"Max size: {cache.max_size()}")
            ```
        """
        return self._max_size

    def get_memory_usage_mb(self) -> float:
        """
        Get current memory usage in MB.

        Returns:
            Current memory usage in megabytes

        Example:
            ```python
            memory_mb = cache.get_memory_usage_mb()
            print(f"Memory usage: {memory_mb:.2f}MB")
            ```
        """
        with self._lock:
            return self._total_size_bytes / (1024 * 1024)

    def get_memory_utilization(self) -> float:
        """
        Get memory utilization as percentage (0.0-1.0).

        Returns:
            Memory utilization ratio

        Example:
            ```python
            utilization = cache.get_memory_utilization()
            if utilization > 0.9:
                print(f"Warning: High memory usage {utilization:.1%}")
            ```
        """
        with self._lock:
            return self._total_size_bytes / self._max_memory_bytes

    def get_memory_limit_evictions(self) -> int:
        """
        Get number of evictions due to memory limit.

        Returns:
            Count of memory-triggered evictions

        Example:
            ```python
            mem_evictions = cache.get_memory_limit_evictions()
            print(f"Memory limit evictions: {mem_evictions}")
            ```
        """
        with self._lock:
            return self._memory_limit_evictions

    def __len__(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, template_id: str) -> bool:
        """Check if template exists in cache."""
        with self._lock:
            return template_id in self._cache

    def __repr__(self) -> str:
        """String representation."""
        with self._lock:
            hit_rate = self.get_hit_rate()
            memory_mb = self._total_size_bytes / (1024 * 1024)
            max_memory_mb = self._max_memory_bytes / (1024 * 1024)
            return (
                f"TemplateLRUCache(size={len(self._cache)}/{self._max_size}, "
                f"memory={memory_mb:.1f}/{max_memory_mb:.0f}MB, "
                f"hit_rate={hit_rate:.2%}, "
                f"evictions={self._evictions})"
            )
