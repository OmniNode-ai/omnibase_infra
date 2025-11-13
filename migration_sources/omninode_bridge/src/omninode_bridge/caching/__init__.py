"""Caching infrastructure for omninode_bridge.

This module provides Redis-based caching for:
- Intelligence query results (1hr TTL)
- Code templates (24hr TTL)
- Pattern analysis (12hr TTL)

Key Features:
- Multi-tier caching (memory + Redis)
- Automatic cache invalidation
- Cache metrics and monitoring
- TTL-based expiration
- Compression for large values
"""

from .cache_manager import (
    CacheManager,
    CacheMetrics,
    CacheType,
    get_cache_manager,
    init_cache_manager,
    shutdown_cache_manager,
)

__all__ = [
    "CacheManager",
    "CacheMetrics",
    "CacheType",
    "get_cache_manager",
    "init_cache_manager",
    "shutdown_cache_manager",
]
