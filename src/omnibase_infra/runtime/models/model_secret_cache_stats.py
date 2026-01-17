# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Model for secret cache statistics.

.. versionadded:: 0.8.0
    Initial implementation for OMN-764.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelSecretCacheStats(BaseModel):
    """Cache statistics for observability.

    Provides metrics for monitoring cache performance and health.

    Attributes:
        total_entries: Current number of cached entries.
        hits: Total number of cache hits.
        misses: Total number of cache misses.
        refreshes: Total number of cache refreshes (TTL expiration).
        expired_evictions: Total number of entries evicted due to expiration.

    Example:
        >>> stats = ModelSecretCacheStats(hits=100, misses=10)
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
        description="Current number of secrets in the cache.",
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
        description="Total number of cache refreshes due to TTL expiration.",
    )
    expired_evictions: int = Field(
        default=0,
        ge=0,
        description="Total number of entries evicted due to expiration.",
    )

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Hit rate as a float between 0.0 and 1.0.
            Returns 0.0 if no hits or misses have occurred.

        Example:
            >>> stats = ModelSecretCacheStats(hits=80, misses=20)
            >>> stats.hit_rate
            0.8
        """
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


__all__: list[str] = ["ModelSecretCacheStats"]
