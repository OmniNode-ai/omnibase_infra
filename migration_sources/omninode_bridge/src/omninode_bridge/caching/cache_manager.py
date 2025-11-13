"""Cache manager for expensive operations with TTL-based invalidation.

This module provides a caching layer for expensive operations like:
- Intelligence query results
- Contract templates
- Validated patterns
- Generated code snippets

Performance Targets:
- Cache hit rate >70% after warmup
- <5ms cache read/write operations
- Memory-efficient with automatic cleanup
- TTL-based expiration

Supported Backends:
- Memory (development): Fast, no external dependencies
- Redis (production): Distributed, persistent, scalable

Environment Configuration:
- CACHE_BACKEND: Cache backend (memory, redis) [default: memory]
- REDIS_HOST: Redis host [default: localhost]
- REDIS_PORT: Redis port [default: 6379]
- REDIS_DB: Redis database [default: 0]
- REDIS_PASSWORD: Redis password [optional]
- CACHE_DEFAULT_TTL: Default TTL in seconds [default: 3600]
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CacheType(Enum):
    """Cache type enumeration for different data categories."""

    INTELLIGENCE = "intelligence"  # Intelligence query results (1hr TTL)
    CONTRACTS = "contracts"  # Contract templates (24hr TTL)
    PATTERNS = "patterns"  # Validated patterns (12hr TTL)
    CODE_SNIPPETS = "code_snippets"  # Generated code (30min TTL)


@dataclass
class CacheMetrics:
    """Metrics for cache performance."""

    hits: int = 0
    misses: int = 0
    writes: int = 0
    invalidations: int = 0
    errors: int = 0
    total_read_time_ms: float = 0.0
    total_write_time_ms: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def get_average_read_time_ms(self) -> float:
        """Calculate average read time."""
        total_reads = self.hits + self.misses
        if total_reads == 0:
            return 0.0
        return self.total_read_time_ms / total_reads

    def get_average_write_time_ms(self) -> float:
        """Calculate average write time."""
        if self.writes == 0:
            return 0.0
        return self.total_write_time_ms / self.writes


class CacheManager:
    """Manages caching for expensive operations with TTL-based expiration.

    This cache manager provides:
    - Multiple TTL strategies for different data types
    - Hit/miss rate tracking
    - Pattern-based invalidation
    - Memory or Redis backend support

    TTL Defaults:
    - Intelligence queries: 1 hour (3600s)
    - Contract templates: 24 hours (86400s)
    - Validated patterns: 12 hours (43200s)
    - Generated code snippets: 30 minutes (1800s)

    Example:
        ```python
        # Initialize cache manager
        cache = CacheManager(backend="redis")
        await cache.initialize()

        # Cache intelligence result
        await cache.set_intelligence_result(
            query="optimization patterns",
            context={"domain": "api"},
            result={"patterns": ["pattern1"]},
            ttl=3600
        )

        # Retrieve cached result
        result = await cache.get_intelligence_result(
            query="optimization patterns",
            context={"domain": "api"}
        )

        # Invalidate pattern
        await cache.invalidate_pattern("intelligence:*")

        # Get metrics
        metrics = cache.get_metrics()
        print(f"Hit rate: {metrics['hit_rate']:.1%}")
        ```
    """

    # TTL constants (seconds)
    TTL_INTELLIGENCE = 3600  # 1 hour
    TTL_CONTRACTS = 86400  # 24 hours
    TTL_PATTERNS = 43200  # 12 hours
    TTL_CODE_SNIPPETS = 1800  # 30 minutes

    def __init__(
        self,
        backend: str | None = None,
        redis_host: str | None = None,
        redis_port: int | None = None,
        redis_db: int | None = None,
        redis_password: str | None = None,
        default_ttl: int | None = None,
    ):
        """Initialize cache manager.

        Args:
            backend: Cache backend (memory, redis) [default: memory]
            redis_host: Redis host [default: localhost]
            redis_port: Redis port [default: 6379]
            redis_db: Redis database [default: 0]
            redis_password: Redis password [optional]
            default_ttl: Default TTL in seconds [default: 3600]
        """
        self.backend = backend or os.getenv("CACHE_BACKEND", "memory")
        self.redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
        self.redis_port = redis_port or int(os.getenv("REDIS_PORT", "6379"))
        self.redis_db = redis_db or int(os.getenv("REDIS_DB", "0"))
        self.redis_password = redis_password or os.getenv("REDIS_PASSWORD")
        self.default_ttl = default_ttl or int(os.getenv("CACHE_DEFAULT_TTL", "3600"))

        # Cache backend
        self._cache: Any = None
        self._initialized = False

        # Metrics
        self.metrics = CacheMetrics()

    async def initialize(self) -> None:
        """Initialize cache backend."""
        if self._initialized:
            return

        if self.backend == "redis":
            await self._initialize_redis()
        else:
            await self._initialize_memory()

        self._initialized = True
        logger.info(f"Cache manager initialized with {self.backend} backend")

    async def _initialize_redis(self) -> None:
        """Initialize Redis backend."""
        try:
            import redis.asyncio as redis

            self._cache = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password,
                decode_responses=True,
                socket_connect_timeout=5,
            )

            # Test connection
            await self._cache.ping()
            logger.info(
                f"Connected to Redis at {self.redis_host}:{self.redis_port} (db={self.redis_db})"
            )

        except ImportError:
            logger.warning(
                "redis package not installed, falling back to memory cache. "
                "Install with: pip install redis"
            )
            await self._initialize_memory()
        except Exception as e:
            logger.error(
                f"Failed to connect to Redis: {e}. Falling back to memory cache"
            )
            await self._initialize_memory()

    async def _initialize_memory(self) -> None:
        """Initialize memory-based cache."""
        self.backend = "memory"
        self._cache = {}
        logger.info("Using memory-based cache (not persistent)")

    def _generate_key(self, prefix: str, params: dict[str, Any]) -> str:
        """Generate cache key from parameters.

        Args:
            prefix: Key prefix (e.g., 'intelligence', 'contract')
            params: Parameters to hash

        Returns:
            Cache key string
        """
        # Sort keys for consistent hashing
        param_str = json.dumps(params, sort_keys=True, default=str)
        hash_key = hashlib.sha256(param_str.encode()).hexdigest()[:16]
        return f"{prefix}:{hash_key}"

    async def get_intelligence_result(
        self, query: str, context: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Get cached intelligence query result.

        Args:
            query: Query string
            context: Query context

        Returns:
            Cached result or None if not found
        """
        key = self._generate_key("intelligence", {"query": query, "context": context})
        return await self._get(key)

    async def set_intelligence_result(
        self,
        query: str,
        context: dict[str, Any],
        result: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """Cache intelligence query result.

        Args:
            query: Query string
            context: Query context
            result: Query result to cache
            ttl: Time to live in seconds (default: 1 hour)

        Returns:
            True if cached successfully
        """
        key = self._generate_key("intelligence", {"query": query, "context": context})
        return await self._set(key, result, ttl or self.TTL_INTELLIGENCE)

    async def get_contract_template(
        self, contract_type: str, node_type: str
    ) -> dict[str, Any] | None:
        """Get cached contract template.

        Args:
            contract_type: Contract type
            node_type: Node type

        Returns:
            Cached template or None if not found
        """
        key = self._generate_key(
            "contract", {"type": contract_type, "node_type": node_type}
        )
        return await self._get(key)

    async def set_contract_template(
        self,
        contract_type: str,
        node_type: str,
        template: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """Cache contract template.

        Args:
            contract_type: Contract type
            node_type: Node type
            template: Template data
            ttl: Time to live in seconds (default: 24 hours)

        Returns:
            True if cached successfully
        """
        key = self._generate_key(
            "contract", {"type": contract_type, "node_type": node_type}
        )
        return await self._set(key, template, ttl or self.TTL_CONTRACTS)

    async def get_validated_pattern(
        self, pattern: str, language: str
    ) -> dict[str, Any] | None:
        """Get cached validated pattern.

        Args:
            pattern: Pattern name
            language: Programming language

        Returns:
            Cached pattern or None if not found
        """
        key = self._generate_key("pattern", {"pattern": pattern, "language": language})
        return await self._get(key)

    async def set_validated_pattern(
        self,
        pattern: str,
        language: str,
        validation_result: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """Cache validated pattern.

        Args:
            pattern: Pattern name
            language: Programming language
            validation_result: Validation result
            ttl: Time to live in seconds (default: 12 hours)

        Returns:
            True if cached successfully
        """
        key = self._generate_key("pattern", {"pattern": pattern, "language": language})
        return await self._set(key, validation_result, ttl or self.TTL_PATTERNS)

    async def get_code_snippet(self, snippet_hash: str) -> dict[str, Any] | None:
        """Get cached code snippet.

        Args:
            snippet_hash: Snippet hash

        Returns:
            Cached snippet or None if not found
        """
        key = f"snippet:{snippet_hash}"
        return await self._get(key)

    async def set_code_snippet(
        self,
        snippet_hash: str,
        snippet: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """Cache code snippet.

        Args:
            snippet_hash: Snippet hash
            snippet: Snippet data
            ttl: Time to live in seconds (default: 30 minutes)

        Returns:
            True if cached successfully
        """
        key = f"snippet:{snippet_hash}"
        return await self._set(key, snippet, ttl or self.TTL_CODE_SNIPPETS)

    async def _get(self, key: str) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        import time

        start_time = time.perf_counter()

        try:
            if self.backend == "redis":
                value_str = await self._cache.get(key)
                value = json.loads(value_str) if value_str else None
            else:
                # Memory cache with TTL check
                cached_data = self._cache.get(key)
                if cached_data:
                    value, expires_at = cached_data
                    if datetime.now(UTC).timestamp() < expires_at:
                        pass  # Value is still valid
                    else:
                        # Expired, remove it
                        del self._cache[key]
                        value = None
                else:
                    value = None

            # Track metrics
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.total_read_time_ms += elapsed_ms

            if value is not None:
                self.metrics.hits += 1
                logger.debug(f"Cache HIT: {key} ({elapsed_ms:.2f}ms)")
            else:
                self.metrics.misses += 1
                logger.debug(f"Cache MISS: {key} ({elapsed_ms:.2f}ms)")

            return value

        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
            self.metrics.errors += 1
            self.metrics.misses += 1
            return None

    async def _set(self, key: str, value: Any, ttl: int) -> bool:
        """Set value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds

        Returns:
            True if set successfully
        """
        import time

        start_time = time.perf_counter()

        try:
            if self.backend == "redis":
                value_str = json.dumps(value, default=str)
                await self._cache.setex(key, ttl, value_str)
            else:
                # Memory cache with expiration timestamp
                expires_at = datetime.now(UTC).timestamp() + ttl
                self._cache[key] = (value, expires_at)

            # Track metrics
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.total_write_time_ms += elapsed_ms
            self.metrics.writes += 1

            logger.debug(f"Cache SET: {key} (TTL={ttl}s, {elapsed_ms:.2f}ms)")
            return True

        except Exception as e:
            logger.error(f"Error writing to cache: {e}")
            self.metrics.errors += 1
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern.

        Args:
            pattern: Pattern to match (e.g., 'intelligence:*', 'contract:*')

        Returns:
            Number of keys invalidated
        """
        try:
            count = 0

            if self.backend == "redis":
                # Use SCAN for safe pattern matching
                cursor = 0
                while True:
                    cursor, keys = await self._cache.scan(
                        cursor, match=pattern, count=100
                    )
                    if keys:
                        await self._cache.delete(*keys)
                        count += len(keys)
                    if cursor == 0:
                        break
            else:
                # Memory cache pattern matching
                keys_to_delete = [
                    k for k in self._cache if self._pattern_match(k, pattern)
                ]
                for key in keys_to_delete:
                    del self._cache[key]
                count = len(keys_to_delete)

            self.metrics.invalidations += count
            logger.info(f"Invalidated {count} cache entries matching '{pattern}'")
            return count

        except Exception as e:
            logger.error(f"Error invalidating cache pattern '{pattern}': {e}")
            self.metrics.errors += 1
            return 0

    def _pattern_match(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern (simple * wildcard support)."""
        if "*" not in pattern:
            return key == pattern

        # Simple wildcard matching
        import re

        regex_pattern = "^" + pattern.replace("*", ".*") + "$"
        return bool(re.match(regex_pattern, key))

    async def clear(self) -> None:
        """Clear all cache entries."""
        try:
            if self.backend == "redis":
                await self._cache.flushdb()
            else:
                self._cache.clear()

            logger.info("Cache cleared")

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    async def shutdown(self) -> None:
        """Shutdown cache backend."""
        if self.backend == "redis" and self._cache:
            await self._cache.close()

        self._initialized = False
        logger.info("Cache manager shutdown")

    def get_metrics(self) -> dict[str, Any]:
        """Get cache performance metrics.

        Returns:
            Dictionary with metrics:
            - hits: Cache hit count
            - misses: Cache miss count
            - hit_rate: Cache hit rate (0.0-1.0)
            - invalidations: Total invalidations
            - errors: Total errors
            - average_read_time_ms: Average read time
            - average_write_time_ms: Average write time
            - backend: Cache backend type
            - uptime_seconds: Cache uptime
        """
        uptime = (datetime.now(UTC) - self.metrics.created_at).total_seconds()

        return {
            "hits": self.metrics.hits,
            "misses": self.metrics.misses,
            "hit_rate": round(self.metrics.get_hit_rate(), 4),
            "invalidations": self.metrics.invalidations,
            "errors": self.metrics.errors,
            "average_read_time_ms": round(self.metrics.get_average_read_time_ms(), 2),
            "average_write_time_ms": round(self.metrics.get_average_write_time_ms(), 2),
            "backend": self.backend,
            "uptime_seconds": round(uptime, 2),
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()


# Global cache manager instance
_cache_manager: CacheManager | None = None


async def init_cache_manager(
    backend: str | None = None,
    redis_host: str | None = None,
    redis_port: int | None = None,
    redis_db: int | None = None,
    redis_password: str | None = None,
    default_ttl: int | None = None,
) -> CacheManager:
    """Initialize global cache manager.

    Args:
        backend: Cache backend (memory, redis) [default: memory]
        redis_host: Redis host [default: localhost]
        redis_port: Redis port [default: 6379]
        redis_db: Redis database [default: 0]
        redis_password: Redis password [optional]
        default_ttl: Default TTL in seconds [default: 3600]

    Returns:
        Initialized cache manager
    """
    global _cache_manager

    if _cache_manager is not None:
        logger.warning("Cache manager already initialized")
        return _cache_manager

    _cache_manager = CacheManager(
        backend=backend,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
        redis_password=redis_password,
        default_ttl=default_ttl,
    )
    await _cache_manager.initialize()

    logger.info("Global cache manager initialized")
    return _cache_manager


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance.

    Returns:
        Cache manager instance

    Raises:
        RuntimeError: If cache manager not initialized
    """
    if _cache_manager is None:
        raise RuntimeError(
            "Cache manager not initialized. Call init_cache_manager() first."
        )
    return _cache_manager


async def shutdown_cache_manager() -> None:
    """Shutdown global cache manager."""
    global _cache_manager

    if _cache_manager is not None:
        await _cache_manager.shutdown()
        _cache_manager = None
        logger.info("Global cache manager shutdown")
