# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 OmniNode Team

"""Cache information type for topic parser caching statistics.

This module provides a type-safe NamedTuple for representing cache statistics,
mirroring the structure of functools._CacheInfo for compatibility with LRU
cache implementations.

Example:
    >>> from omnibase_infra.types import TypeCacheInfo
    >>> info = TypeCacheInfo(hits=100, misses=10, maxsize=128, currsize=50)
    >>> print(f"Hit rate: {info.hits / (info.hits + info.misses):.2%}")
    Hit rate: 90.91%
"""

from typing import NamedTuple

__all__ = ["TypeCacheInfo"]


class TypeCacheInfo(NamedTuple):
    """Cache statistics for topic parsing operations.

    This mirrors functools._CacheInfo structure for type safety when
    exposing cache statistics from LRU-cached parsing functions.

    Attributes:
        hits: Number of cache hits (requests served from cache).
        misses: Number of cache misses (requests requiring computation).
        maxsize: Maximum cache size, or None for unlimited cache.
        currsize: Current number of entries in the cache.

    Example:
        >>> info = TypeCacheInfo(hits=50, misses=5, maxsize=128, currsize=30)
        >>> info.hits
        50
        >>> info.maxsize
        128

    Note:
        This type is intentionally compatible with functools._CacheInfo
        to allow seamless interoperability with standard library caching.
    """

    hits: int
    misses: int
    maxsize: int | None
    currsize: int
