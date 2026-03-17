# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka event payload for wiring health snapshot.

Published by ``WiringHealthChecker`` on the
``onex.evt.omnibase-infra.wiring-health-snapshot.v1`` topic after each
health computation.  Downstream consumers (omnidash /wiring-health)
subscribe to this topic to display emission/consumption health.

Ticket: OMN-5292
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.observability.wiring_health.model_wiring_health_metrics import (
    ModelWiringHealthMetrics,
)


class ModelWiringHealthSnapshotEvent(BaseModel):
    """Payload emitted as a Kafka event after each wiring health computation.

    Attributes:
        timestamp: When the health check ran.
        overall_healthy: True if ALL monitored topics are healthy.
        unhealthy_count: Number of topics exceeding mismatch threshold.
        threshold: Mismatch threshold used.
        topics: Per-topic emit/consume counts and health flags.
        correlation_id: Correlation ID for tracing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    timestamp: datetime = Field(..., description="When the health check ran")
    overall_healthy: bool = Field(..., description="All monitored topics are healthy")
    unhealthy_count: int = Field(..., ge=0, description="Number of unhealthy topics")
    threshold: float = Field(..., ge=0.0, description="Mismatch threshold")
    topics: tuple[dict[str, object], ...] = Field(
        ..., description="Per-topic health records"
    )
    correlation_id: UUID = Field(..., description="Correlation ID for tracing")

    @classmethod
    def from_metrics(
        cls,
        metrics: ModelWiringHealthMetrics,
        correlation_id: UUID,
    ) -> ModelWiringHealthSnapshotEvent:
        """Build event payload from computed wiring health metrics.

        Args:
            metrics: Computed wiring health metrics.
            correlation_id: Correlation ID for tracing.

        Returns:
            ModelWiringHealthSnapshotEvent ready for emission.
        """
        topic_records: tuple[dict[str, object], ...] = tuple(
            {
                "topic": t.topic,
                "emit_count": t.emit_count,
                "consume_count": t.consume_count,
                "mismatch_ratio": t.mismatch_ratio,
                "is_healthy": t.is_healthy,
            }
            for t in metrics.topics
        )
        return cls(
            timestamp=metrics.timestamp,
            overall_healthy=metrics.overall_healthy,
            unhealthy_count=metrics.unhealthy_count,
            threshold=metrics.threshold,
            topics=topic_records,
            correlation_id=correlation_id,
        )


__all__: list[str] = ["ModelWiringHealthSnapshotEvent"]
