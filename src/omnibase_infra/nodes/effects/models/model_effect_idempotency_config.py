# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration model for Effect layer idempotency store.

This module provides configuration for the in-memory idempotency store used
by Effect nodes to track completed backends during dual-backend operations.

Memory Characteristics:
    The default configuration limits memory usage:
    - max_cache_size=10000 entries
    - Each entry: ~100 bytes (UUID key + set of backend strings + timestamp)
    - Max memory at default: ~1MB

    For high-volume systems, adjust based on:
    - Expected concurrent correlation IDs
    - Backend completion tracking duration (TTL)
    - Available memory headroom

Production Note:
    This in-memory store does NOT persist across restarts and does NOT
    support distributed scenarios. For production use:
    - Use StoreIdempotencyPostgres from omnibase_infra.idempotency
    - Or implement a Redis/Valkey-backed store
    - Consider the existing ProtocolIdempotencyStore for full persistence

Related:
    - NodeRegistryEffect: Uses this for dual-backend idempotency
    - InMemoryEffectIdempotencyStore: Implementation using this config
    - OMN-954: Registry effect idempotency requirements
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelEffectIdempotencyConfig(BaseModel):
    """Configuration for Effect layer in-memory idempotency store.

    This configuration controls the bounded in-memory cache used for tracking
    completed backends during dual-backend registration operations.

    Attributes:
        max_cache_size: Maximum number of correlation IDs to track.
            When exceeded, oldest entries are evicted (LRU).
            Default: 10000 entries (~1MB memory).
        cache_ttl_seconds: Time-to-live for entries in seconds.
            Entries older than TTL are eligible for cleanup.
            Default: 3600.0 (1 hour).
        cleanup_interval_seconds: How often to run TTL cleanup.
            Cleanup happens lazily on access, but this sets the minimum
            interval between full cleanup passes.
            Default: 300.0 (5 minutes).

    Memory Estimation:
        - Per-entry overhead: ~100 bytes
        - max_cache_size=10000 -> ~1MB
        - max_cache_size=100000 -> ~10MB

    Example:
        >>> config = ModelEffectIdempotencyConfig(
        ...     max_cache_size=5000,
        ...     cache_ttl_seconds=1800.0,  # 30 minutes
        ... )
        >>> store = InMemoryEffectIdempotencyStore(config)
    """

    max_cache_size: int = Field(
        default=10000,
        ge=1,
        le=1000000,
        description="Maximum entries before LRU eviction.",
    )

    cache_ttl_seconds: float = Field(
        default=3600.0,
        ge=1.0,
        le=86400.0,
        description="Entry TTL in seconds (1 hour default).",
    )

    cleanup_interval_seconds: float = Field(
        default=300.0,
        ge=1.0,
        le=3600.0,
        description="Minimum interval between TTL cleanup passes.",
    )

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)


__all__ = ["ModelEffectIdempotencyConfig"]
