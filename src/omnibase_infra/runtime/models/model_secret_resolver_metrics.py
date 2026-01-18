# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Model for secret resolver metrics.

.. versionadded:: 0.8.0
    Initial implementation for OMN-1374.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelSecretResolverMetrics(BaseModel):
    """Resolution metrics for SecretResolver observability.

    Provides aggregated metrics for monitoring secret resolution performance
    and health across different source types (env, file, vault, cache).

    Attributes:
        success_counts: Count of successful resolutions by source type.
        failure_counts: Count of failed resolutions by source type.
        latency_samples: Number of latency samples collected.
        avg_latency_ms: Average resolution latency in milliseconds.
        cache_hits: Total number of cache hits.
        cache_misses: Total number of cache misses.

    Example:
        >>> metrics = ModelSecretResolverMetrics(
        ...     success_counts={"env": 5, "vault": 3},
        ...     failure_counts={"vault": 1},
        ...     latency_samples=9,
        ...     avg_latency_ms=12.5,
        ...     cache_hits=100,
        ...     cache_misses=10,
        ... )
        >>> metrics.success_counts["env"]
        5
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    success_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Count of successful resolutions by source type (e.g., 'env', 'vault', 'file').",
    )
    failure_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Count of failed resolutions by source type.",
    )
    latency_samples: int = Field(
        default=0,
        ge=0,
        description="Number of latency samples collected.",
    )
    avg_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Average resolution latency in milliseconds.",
    )
    cache_hits: int = Field(
        default=0,
        ge=0,
        description="Total number of cache hits.",
    )
    cache_misses: int = Field(
        default=0,
        ge=0,
        description="Total number of cache misses.",
    )

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Hit rate as a float between 0.0 and 1.0.
            Returns 0.0 if no hits or misses have occurred.

        Example:
            >>> metrics = ModelSecretResolverMetrics(cache_hits=80, cache_misses=20)
            >>> metrics.cache_hit_rate
            0.8
        """
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def total_resolutions(self) -> int:
        """Calculate total number of resolutions attempted.

        Returns:
            Sum of all success and failure counts across all source types.

        Example:
            >>> metrics = ModelSecretResolverMetrics(
            ...     success_counts={"env": 5, "vault": 3},
            ...     failure_counts={"vault": 1},
            ... )
            >>> metrics.total_resolutions
            9
        """
        return sum(self.success_counts.values()) + sum(self.failure_counts.values())


__all__: list[str] = ["ModelSecretResolverMetrics"]
