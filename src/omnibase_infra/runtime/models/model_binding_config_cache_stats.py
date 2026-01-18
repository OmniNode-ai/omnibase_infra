# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Model for binding configuration cache statistics.

.. versionadded:: 0.8.0
    Initial implementation for OMN-765.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelBindingConfigCacheStats(BaseModel):
    """Cache statistics for BindingConfigResolver observability.

    Provides metrics for monitoring cache performance and health
    of the binding configuration resolution system.

    Attributes:
        total_entries: Current number of cached configurations.
        hits: Total number of cache hits.
        misses: Total number of cache misses.
        refreshes: Total number of cache refreshes (TTL expiration or manual).
        expired_evictions: Total number of entries evicted due to expiration.
        lru_evictions: Total number of entries evicted due to LRU policy.
        file_loads: Total number of configurations loaded from files.
        env_loads: Total number of configurations loaded from environment variables.
        vault_loads: Total number of configurations loaded from Vault.
        async_key_lock_count: Current number of async key locks held.
        async_key_lock_cleanups: Total number of stale lock cleanup operations.

    Example:
        >>> stats = ModelBindingConfigCacheStats(hits=100, misses=10)
        >>> stats.hit_rate
        0.909090909090909
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    total_entries: int = Field(
        default=0,
        ge=0,
        description="Current number of configurations in the cache.",
    )
    hits: int = Field(
        default=0,
        ge=0,
        description="Total number of cache hits.",
    )
    misses: int = Field(
        default=0,
        ge=0,
        description="Total number of cache misses.",
    )
    refreshes: int = Field(
        default=0,
        ge=0,
        description="Total number of cache refreshes due to TTL expiration or manual.",
    )
    expired_evictions: int = Field(
        default=0,
        ge=0,
        description="Total number of entries evicted due to expiration.",
    )
    lru_evictions: int = Field(
        default=0,
        ge=0,
        description="Total number of entries evicted due to LRU policy when max_cache_entries is reached.",
    )
    file_loads: int = Field(
        default=0,
        ge=0,
        description="Total number of configurations loaded from files.",
    )
    env_loads: int = Field(
        default=0,
        ge=0,
        description="Total number of configurations loaded from environment variables.",
    )
    vault_loads: int = Field(
        default=0,
        ge=0,
        description="Total number of configurations loaded from Vault.",
    )
    async_key_lock_count: int = Field(
        default=0,
        ge=0,
        description="Current number of async key locks held for handler type serialization.",
    )
    async_key_lock_cleanups: int = Field(
        default=0,
        ge=0,
        description="Total number of stale async key lock cleanup operations performed.",
    )

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Hit rate as a float between 0.0 and 1.0.
            Returns 0.0 if no hits or misses have occurred.

        Example:
            >>> stats = ModelBindingConfigCacheStats(hits=80, misses=20)
            >>> stats.hit_rate
            0.8
        """
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_loads(self) -> int:
        """Calculate total configuration loads from all sources.

        Returns:
            Sum of file_loads, env_loads, and vault_loads.

        Example:
            >>> stats = ModelBindingConfigCacheStats(
            ...     file_loads=10, env_loads=5, vault_loads=3
            ... )
            >>> stats.total_loads
            18
        """
        return self.file_loads + self.env_loads + self.vault_loads


__all__: list[str] = ["ModelBindingConfigCacheStats"]
