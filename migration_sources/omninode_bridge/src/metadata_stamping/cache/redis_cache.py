"""
Redis caching layer for MetadataStampingService Phase 2.

Provides high-performance caching with TTL management, cache warming,
and intelligent cache strategies for metadata stamps and hash operations.
"""

import asyncio
import hashlib
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import redis.asyncio as redis
from pydantic import BaseModel, Field

# Import batch size configuration
from omninode_bridge.config.batch_sizes import get_batch_manager

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategies for different types of data."""

    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    READ_THROUGH = "read_through"
    CACHE_ASIDE = "cache_aside"


class CacheLevel(Enum):
    """Cache levels for hierarchical caching."""

    L1_MEMORY = "l1_memory"  # In-memory cache
    L2_REDIS = "l2_redis"  # Redis distributed cache
    L3_DATABASE = "l3_database"  # Database fallback


@dataclass
class CacheConfig:
    """Redis cache configuration with performance tuning."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    socket_timeout: float = 30.0
    socket_connect_timeout: float = 30.0
    socket_keepalive: bool = True
    socket_keepalive_options: dict[str, int] = field(default_factory=lambda: {})
    health_check_interval: int = 30

    # Connection pool settings
    max_connections: int = 100
    retry_on_timeout: bool = True
    retry_on_error: list[type] = field(
        default_factory=lambda: [redis.ConnectionError, redis.TimeoutError]
    )

    # Performance settings
    decode_responses: bool = True
    encoding: str = "utf-8"
    encoding_errors: str = "strict"

    # Cache strategy settings
    default_ttl: int = 3600  # 1 hour
    hash_cache_ttl: int = 86400  # 24 hours for hash results
    metadata_cache_ttl: int = 7200  # 2 hours for metadata
    batch_cache_ttl: int = 1800  # 30 minutes for batch operations

    # Cache warming settings
    enable_cache_warming: bool = True
    warm_cache_size: int = 1000
    warm_cache_interval: int = 300  # 5 minutes


class CacheMetrics:
    """Cache performance metrics tracking."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.total_operations = 0
        self.total_response_time = 0.0
        self.cache_size = 0
        self.evictions = 0

    @property
    def hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        return (
            self.total_response_time / self.total_operations
            if self.total_operations > 0
            else 0.0
        )

    def record_hit(self, response_time: float):
        """Record a cache hit."""
        self.hits += 1
        self.total_operations += 1
        self.total_response_time += response_time

    def record_miss(self, response_time: float):
        """Record a cache miss."""
        self.misses += 1
        self.total_operations += 1
        self.total_response_time += response_time

    def record_error(self):
        """Record a cache error."""
        self.errors += 1


class StampCacheEntry(BaseModel):
    """Cache entry for metadata stamps."""

    stamp_id: str
    file_hash: str
    stamp_data: dict[str, Any]
    created_at: float = Field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = Field(default_factory=time.time)


class HashCacheEntry(BaseModel):
    """Cache entry for hash operations."""

    file_hash: str
    execution_time_ms: float
    file_size_bytes: int
    cpu_usage_percent: float
    performance_grade: str
    created_at: float = Field(default_factory=time.time)


class MetadataStampingRedisCache:
    """
    High-performance Redis cache implementation for MetadataStampingService.

    Features:
    - Multi-level caching with L1 (memory) and L2 (Redis)
    - Intelligent TTL management based on access patterns
    - Cache warming and preloading
    - Performance metrics and monitoring
    - Batch operations for optimal throughput
    - Circuit breaker for resilience
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.metrics = CacheMetrics()

        # L1 memory cache for hot data
        self._memory_cache: dict[str, Any] = {}
        self._memory_cache_access: dict[str, float] = {}
        self._memory_cache_max_size = 1000

        # Cache warming task
        self._cache_warming_task: Optional[asyncio.Task] = None

        # Performance tracking
        self._performance_history: list[dict[str, Any]] = []

        # Circuit breaker state
        self._circuit_breaker_open = False
        self._circuit_breaker_last_failure = 0.0
        self._circuit_breaker_failure_count = 0
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_timeout = 60.0

    async def initialize(self) -> bool:
        """Initialize Redis connection and start background tasks."""
        try:
            # Create Redis connection pool
            connection_kwargs = {
                "host": self.config.host,
                "port": self.config.port,
                "db": self.config.db,
                "password": self.config.password,
                "ssl": self.config.ssl,
                "socket_timeout": self.config.socket_timeout,
                "socket_connect_timeout": self.config.socket_connect_timeout,
                "socket_keepalive": self.config.socket_keepalive,
                "socket_keepalive_options": self.config.socket_keepalive_options,
                "health_check_interval": self.config.health_check_interval,
                "max_connections": self.config.max_connections,
                "retry_on_timeout": self.config.retry_on_timeout,
                "decode_responses": self.config.decode_responses,
                "encoding": self.config.encoding,
                "encoding_errors": self.config.encoding_errors,
            }

            self.redis_client = redis.Redis(**connection_kwargs)

            # Test connection
            await self.redis_client.ping()

            # Start cache warming if enabled
            if self.config.enable_cache_warming:
                self._cache_warming_task = asyncio.create_task(
                    self._cache_warming_worker()
                )

            logger.info(
                f"Redis cache initialized: {self.config.host}:{self.config.port}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            return False

    async def close(self):
        """Close Redis connection and cleanup resources."""
        if self._cache_warming_task:
            self._cache_warming_task.cancel()
            try:
                await self._cache_warming_task
            except asyncio.CancelledError:
                pass

        if self.redis_client:
            await self.redis_client.aclose()
            logger.info("Redis cache connection closed")

    @asynccontextmanager
    async def _circuit_breaker(self):
        """Circuit breaker pattern for Redis operations."""
        if self._circuit_breaker_open:
            if (
                time.time() - self._circuit_breaker_last_failure
                > self._circuit_breaker_timeout
            ):
                self._circuit_breaker_open = False
                self._circuit_breaker_failure_count = 0
                logger.info("Circuit breaker closed - attempting Redis operations")
            else:
                raise redis.ConnectionError(
                    "Circuit breaker open - Redis operations disabled"
                )

        try:
            yield
            # Reset failure count on success
            self._circuit_breaker_failure_count = 0
        except (redis.ConnectionError, redis.TimeoutError) as e:
            self._circuit_breaker_failure_count += 1
            self._circuit_breaker_last_failure = time.time()

            if self._circuit_breaker_failure_count >= self._circuit_breaker_threshold:
                self._circuit_breaker_open = True
                logger.warning(
                    f"Circuit breaker opened after {self._circuit_breaker_failure_count} failures"
                )

            raise e

    def _generate_cache_key(self, prefix: str, identifier: str) -> str:
        """Generate standardized cache key."""
        # Use hash to keep keys short and consistent
        key_hash = hashlib.sha256(f"{prefix}:{identifier}".encode()).hexdigest()[:16]
        return f"metadata_stamping:{prefix}:{key_hash}"

    def _manage_memory_cache(self, key: str, value: Any):
        """Manage L1 memory cache with LRU eviction."""
        current_time = time.time()

        # Add/update entry
        self._memory_cache[key] = value
        self._memory_cache_access[key] = current_time

        # Evict oldest entries if cache is full
        if len(self._memory_cache) > self._memory_cache_max_size:
            # Sort by access time and remove oldest
            sorted_keys = sorted(self._memory_cache_access.items(), key=lambda x: x[1])
            keys_to_remove = [
                k
                for k, _ in sorted_keys[
                    : len(self._memory_cache) - self._memory_cache_max_size
                ]
            ]

            for key_to_remove in keys_to_remove:
                self._memory_cache.pop(key_to_remove, None)
                self._memory_cache_access.pop(key_to_remove, None)
                self.metrics.evictions += 1

    async def cache_stamp(
        self, stamp_entry: StampCacheEntry, ttl: Optional[int] = None
    ) -> bool:
        """Cache a metadata stamp with intelligent TTL."""
        start_time = time.perf_counter()
        ttl = ttl or self.config.metadata_cache_ttl

        try:
            cache_key = self._generate_cache_key("stamp", stamp_entry.file_hash)
            cache_data = stamp_entry.model_dump_json()

            # Store in L1 memory cache
            self._manage_memory_cache(cache_key, cache_data)

            # Store in L2 Redis cache
            async with self._circuit_breaker():
                await self.redis_client.setex(cache_key, ttl, cache_data)

            response_time = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"Cached stamp {stamp_entry.file_hash} in {response_time:.2f}ms"
            )

            return True

        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Failed to cache stamp {stamp_entry.file_hash}: {e}")
            return False

    async def get_stamp(self, file_hash: str) -> Optional[StampCacheEntry]:
        """Retrieve cached metadata stamp with L1/L2 hierarchy."""
        start_time = time.perf_counter()
        cache_key = self._generate_cache_key("stamp", file_hash)

        try:
            # Check L1 memory cache first
            if cache_key in self._memory_cache:
                cache_data = self._memory_cache[cache_key]
                self._memory_cache_access[cache_key] = time.time()

                response_time = (time.perf_counter() - start_time) * 1000
                self.metrics.record_hit(response_time)

                return StampCacheEntry.model_validate_json(cache_data)

            # Check L2 Redis cache
            async with self._circuit_breaker():
                cache_data = await self.redis_client.get(cache_key)

            if cache_data:
                # Store in L1 for future access
                self._manage_memory_cache(cache_key, cache_data)

                response_time = (time.perf_counter() - start_time) * 1000
                self.metrics.record_hit(response_time)

                return StampCacheEntry.model_validate_json(cache_data)

            # Cache miss
            response_time = (time.perf_counter() - start_time) * 1000
            self.metrics.record_miss(response_time)
            return None

        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Failed to get cached stamp {file_hash}: {e}")
            return None

    async def cache_hash_result(
        self, hash_entry: HashCacheEntry, ttl: Optional[int] = None
    ) -> bool:
        """Cache hash operation result."""
        start_time = time.perf_counter()
        ttl = ttl or self.config.hash_cache_ttl

        try:
            cache_key = self._generate_cache_key("hash", hash_entry.file_hash)
            cache_data = hash_entry.model_dump_json()

            # Store in L1 memory cache
            self._manage_memory_cache(cache_key, cache_data)

            # Store in L2 Redis cache
            async with self._circuit_breaker():
                await self.redis_client.setex(cache_key, ttl, cache_data)

            response_time = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"Cached hash result {hash_entry.file_hash} in {response_time:.2f}ms"
            )

            return True

        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Failed to cache hash result {hash_entry.file_hash}: {e}")
            return False

    async def get_hash_result(self, file_hash: str) -> Optional[HashCacheEntry]:
        """Retrieve cached hash result."""
        start_time = time.perf_counter()
        cache_key = self._generate_cache_key("hash", file_hash)

        try:
            # Check L1 memory cache first
            if cache_key in self._memory_cache:
                cache_data = self._memory_cache[cache_key]
                self._memory_cache_access[cache_key] = time.time()

                response_time = (time.perf_counter() - start_time) * 1000
                self.metrics.record_hit(response_time)

                return HashCacheEntry.model_validate_json(cache_data)

            # Check L2 Redis cache
            async with self._circuit_breaker():
                cache_data = await self.redis_client.get(cache_key)

            if cache_data:
                # Store in L1 for future access
                self._manage_memory_cache(cache_key, cache_data)

                response_time = (time.perf_counter() - start_time) * 1000
                self.metrics.record_hit(response_time)

                return HashCacheEntry.model_validate_json(cache_data)

            # Cache miss
            response_time = (time.perf_counter() - start_time) * 1000
            self.metrics.record_miss(response_time)
            return None

        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Failed to get cached hash result {file_hash}: {e}")
            return None

    async def batch_cache_stamps(
        self, stamp_entries: list[StampCacheEntry], ttl: Optional[int] = None
    ) -> int:
        """Cache multiple metadata stamps in batch for optimal performance."""
        if not stamp_entries:
            return 0

        ttl = ttl or self.config.metadata_cache_ttl
        successful_caches = 0

        try:
            # Prepare batch data
            cache_data = {}
            for entry in stamp_entries:
                cache_key = self._generate_cache_key("stamp", entry.file_hash)
                cache_data[cache_key] = entry.model_dump_json()

                # Store in L1 memory cache
                self._manage_memory_cache(cache_key, cache_data[cache_key])

            # Batch set in Redis
            async with self._circuit_breaker():
                async with self.redis_client.pipeline(transaction=True) as pipeline:
                    for cache_key, data in cache_data.items():
                        await pipeline.setex(cache_key, ttl, data)

                    results = await pipeline.execute()
                    successful_caches = sum(1 for result in results if result)

            logger.info(f"Batch cached {successful_caches}/{len(stamp_entries)} stamps")
            return successful_caches

        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Failed to batch cache stamps: {e}")
            return 0

    async def batch_get_stamps(
        self, file_hashes: list[str]
    ) -> dict[str, Optional[StampCacheEntry]]:
        """Retrieve multiple cached stamps in batch."""
        if not file_hashes:
            return {}

        results = {}
        cache_keys = {
            self._generate_cache_key("stamp", hash_val): hash_val
            for hash_val in file_hashes
        }

        try:
            # Check L1 memory cache first
            memory_hits = {}
            missing_keys = []

            for cache_key, file_hash in cache_keys.items():
                if cache_key in self._memory_cache:
                    cache_data = self._memory_cache[cache_key]
                    self._memory_cache_access[cache_key] = time.time()
                    results[file_hash] = StampCacheEntry.model_validate_json(cache_data)
                    self.metrics.hits += 1
                else:
                    missing_keys.append(cache_key)

            # Batch get from Redis for missing keys
            if missing_keys:
                async with self._circuit_breaker():
                    redis_results = await self.redis_client.mget(missing_keys)

                for i, cache_data in enumerate(redis_results):
                    cache_key = missing_keys[i]
                    file_hash = cache_keys[cache_key]

                    if cache_data:
                        # Store in L1 for future access
                        self._manage_memory_cache(cache_key, cache_data)
                        results[file_hash] = StampCacheEntry.model_validate_json(
                            cache_data
                        )
                        self.metrics.hits += 1
                    else:
                        results[file_hash] = None
                        self.metrics.misses += 1

            return results

        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Failed to batch get stamps: {e}")
            return {hash_val: None for hash_val in file_hashes}

    async def invalidate_cache(self, pattern: str = "*") -> int:
        """Invalidate cache entries matching pattern."""
        try:
            full_pattern = f"metadata_stamping:{pattern}"

            async with self._circuit_breaker():
                # Get all matching keys
                keys = []
                async for key in self.redis_client.scan_iter(match=full_pattern):
                    keys.append(key)

                # Delete in batches using configurable batch size
                deleted_count = 0
                if keys:
                    batch_manager = get_batch_manager()
                    batch_size = batch_manager.redis_batch_size
                    for i in range(0, len(keys), batch_size):
                        batch = keys[i : i + batch_size]
                        deleted_count += await self.redis_client.delete(*batch)

            # Clear memory cache if pattern matches
            if pattern == "*" or "stamp" in pattern or "hash" in pattern:
                self._memory_cache.clear()
                self._memory_cache_access.clear()

            logger.info(
                f"Invalidated {deleted_count} cache entries matching '{pattern}'"
            )
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")
            return 0

    async def _cache_warming_worker(self):
        """Background worker for cache warming."""
        while True:
            try:
                await asyncio.sleep(self.config.warm_cache_interval)

                # This would be implemented to warm cache with frequently accessed data
                # For now, just log the warming cycle
                logger.debug("Cache warming cycle completed")

            except asyncio.CancelledError:
                logger.info("Cache warming worker cancelled")
                break
            except Exception as e:
                logger.error(f"Cache warming error: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        try:
            redis_info = {}
            if self.redis_client and not self._circuit_breaker_open:
                async with self._circuit_breaker():
                    redis_info = await self.redis_client.info("memory")

            stats = {
                "metrics": {
                    "hits": self.metrics.hits,
                    "misses": self.metrics.misses,
                    "errors": self.metrics.errors,
                    "hit_ratio": self.metrics.hit_ratio,
                    "total_operations": self.metrics.total_operations,
                    "average_response_time_ms": self.metrics.average_response_time,
                    "evictions": self.metrics.evictions,
                },
                "memory_cache": {
                    "size": len(self._memory_cache),
                    "max_size": self._memory_cache_max_size,
                    "utilization": len(self._memory_cache)
                    / self._memory_cache_max_size,
                },
                "redis_cache": {
                    "connected": not self._circuit_breaker_open,
                    "memory_usage": redis_info.get("used_memory_human", "N/A"),
                    "peak_memory": redis_info.get("used_memory_peak_human", "N/A"),
                    "fragmentation_ratio": redis_info.get("mem_fragmentation_ratio", 0),
                },
                "circuit_breaker": {
                    "open": self._circuit_breaker_open,
                    "failure_count": self._circuit_breaker_failure_count,
                    "last_failure": self._circuit_breaker_last_failure,
                },
                "config": {
                    "host": self.config.host,
                    "port": self.config.port,
                    "default_ttl": self.config.default_ttl,
                    "cache_warming_enabled": self.config.enable_cache_warming,
                },
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}

    async def health_check(self) -> dict[str, Any]:
        """Perform cache health check."""
        start_time = time.perf_counter()

        try:
            # Test Redis connectivity
            redis_healthy = False
            if self.redis_client and not self._circuit_breaker_open:
                async with self._circuit_breaker():
                    await self.redis_client.ping()
                    redis_healthy = True

            response_time = (time.perf_counter() - start_time) * 1000

            return {
                "status": "healthy" if redis_healthy else "degraded",
                "redis_connected": redis_healthy,
                "response_time_ms": response_time,
                "circuit_breaker_open": self._circuit_breaker_open,
                "memory_cache_size": len(self._memory_cache),
                "hit_ratio": self.metrics.hit_ratio,
            }

        except Exception as e:
            response_time = (time.perf_counter() - start_time) * 1000
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time_ms": response_time,
                "circuit_breaker_open": self._circuit_breaker_open,
            }


# Cache factory function for easy integration
async def create_redis_cache(
    config: Optional[CacheConfig] = None,
) -> MetadataStampingRedisCache:
    """Factory function to create and initialize Redis cache."""
    if config is None:
        config = CacheConfig()

    cache = MetadataStampingRedisCache(config)
    await cache.initialize()
    return cache
