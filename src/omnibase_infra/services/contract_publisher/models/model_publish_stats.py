# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Publish Statistics Model.

Statistics for contract publishing operations including counts, timing, and per-origin breakdown.

.. versionadded:: 0.3.0
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelPublishStats(BaseModel):
    """Publishing statistics for observability.

    Tracks counts, timing, and per-origin breakdown for contract publishing.
    Included in every ModelPublishResult for debugging and monitoring.

    Attributes:
        discovered_count: Total contracts found by discovery
        valid_count: Contracts that passed validation
        published_count: Contracts successfully published to Kafka
        errored_count: Contracts that failed validation
        dedup_count: Contracts deduplicated (same handler_id + hash)
        duration_ms: Total duration of publish_all()
        discover_ms: Time spent in discovery phase
        validate_ms: Time spent in validation phase
        publish_ms: Time spent in publishing phase
        environment: Resolved environment (for log clarity)
        filesystem_count: Contracts from filesystem source
        package_count: Contracts from package source

    Example:
        >>> stats = ModelPublishStats(
        ...     discovered_count=10,
        ...     valid_count=8,
        ...     published_count=8,
        ...     errored_count=2,
        ...     dedup_count=0,
        ...     duration_ms=1234.5,
        ...     discover_ms=100.0,
        ...     validate_ms=500.0,
        ...     publish_ms=634.5,
        ...     environment="dev",
        ...     filesystem_count=10,
        ...     package_count=0,
        ... )

    .. versionadded:: 0.3.0
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Counts
    discovered_count: int = Field(ge=0, description="Total contracts found")
    valid_count: int = Field(ge=0, description="Contracts that passed validation")
    published_count: int = Field(ge=0, description="Contracts successfully published")
    errored_count: int = Field(ge=0, description="Contracts that failed validation")
    dedup_count: int = Field(ge=0, description="Contracts deduplicated (same hash)")

    # Timing (milliseconds)
    duration_ms: float = Field(ge=0.0, description="Total duration")
    discover_ms: float = Field(ge=0.0, description="Discovery phase duration")
    validate_ms: float = Field(ge=0.0, description="Validation phase duration")
    publish_ms: float = Field(ge=0.0, description="Publishing phase duration")

    # Context
    environment: str = Field(description="Resolved environment for topics")

    # Per-origin breakdown
    filesystem_count: int = Field(
        default=0, ge=0, description="Contracts from filesystem"
    )
    package_count: int = Field(default=0, ge=0, description="Contracts from package")


__all__ = ["ModelPublishStats"]
