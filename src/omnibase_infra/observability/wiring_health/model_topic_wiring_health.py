# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Wiring health metrics model for a single topic.

This module provides the data model for wiring health status of a single topic,
including emission count, consumption count, and computed mismatch ratio.

See Also:
    - OMN-1895: Wiring health monitor implementation
    - model_wiring_health_metrics.py: Aggregate metrics across all topics
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

# Default mismatch threshold: 5%
# Accounts for at-least-once delivery, retries, and idempotency windows
DEFAULT_MISMATCH_THRESHOLD: Final[float] = 0.05


class ModelTopicWiringHealth(BaseModel):
    """Wiring health metrics for a single topic.

    Contains emission count, consumption count, and computed mismatch ratio
    for a single topic being monitored.

    Attributes:
        topic: The topic name being monitored.
        emit_count: Number of messages emitted to this topic.
        consume_count: Number of messages successfully consumed from this topic.
        mismatch_ratio: Ratio of abs(emit - consume) / max(emit, 1).
        is_healthy: Whether the topic is within acceptable mismatch threshold.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str = Field(..., description="Topic name")
    emit_count: int = Field(..., ge=0, description="Messages emitted")
    consume_count: int = Field(..., ge=0, description="Messages consumed")
    mismatch_ratio: float = Field(..., ge=0.0, description="Mismatch ratio")
    is_healthy: bool = Field(..., description="Within threshold")

    @classmethod
    def from_counts(
        cls,
        topic: str,
        emit_count: int,
        consume_count: int,
        threshold: float = DEFAULT_MISMATCH_THRESHOLD,
    ) -> ModelTopicWiringHealth:
        """Create topic health from emission and consumption counts.

        Args:
            topic: The topic name.
            emit_count: Number of emissions.
            consume_count: Number of consumptions.
            threshold: Mismatch threshold for health determination.

        Returns:
            ModelTopicWiringHealth with computed mismatch ratio and health status.
        """
        # Compute mismatch ratio: abs(emit - consume) / max(emit, 1)
        # Using max(emit, 1) avoids division by zero
        mismatch_ratio = abs(emit_count - consume_count) / max(emit_count, 1)
        is_healthy = mismatch_ratio <= threshold

        return cls(
            topic=topic,
            emit_count=emit_count,
            consume_count=consume_count,
            mismatch_ratio=mismatch_ratio,
            is_healthy=is_healthy,
        )


__all__ = ["DEFAULT_MISMATCH_THRESHOLD", "ModelTopicWiringHealth"]
