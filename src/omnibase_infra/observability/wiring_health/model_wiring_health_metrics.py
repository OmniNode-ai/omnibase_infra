# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Wiring health metrics model for emission/consumption comparison.

This module provides the data model for aggregate wiring health status,
including per-topic health and overall system status.

Design Rationale:
    - Immutable (frozen=True) for thread-safety
    - Copy-on-write updates via factory methods
    - Threshold-based alert triggering
    - Prometheus metric exposure

See Also:
    - OMN-1895: Wiring health monitor implementation
    - WiringHealthChecker: Checker that computes and returns this model
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.observability.wiring_health.model_topic_wiring_health import (
    DEFAULT_MISMATCH_THRESHOLD,
    ModelTopicWiringHealth,
)


class ModelWiringHealthMetrics(BaseModel):
    """Aggregate wiring health metrics for all monitored topics.

    Contains per-topic health metrics and overall system health status.
    Used by the /health/wiring endpoint and Prometheus metrics exposure.

    Attributes:
        topics: Per-topic wiring health metrics.
        overall_healthy: True if ALL topics are healthy.
        unhealthy_count: Number of topics exceeding mismatch threshold.
        threshold: The mismatch threshold used for health determination.
        timestamp: When these metrics were computed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    topics: tuple[ModelTopicWiringHealth, ...] = Field(
        default_factory=tuple, description="Per-topic health metrics"
    )
    overall_healthy: bool = Field(True, description="All topics healthy")
    unhealthy_count: int = Field(0, ge=0, description="Count of unhealthy topics")
    threshold: float = Field(
        DEFAULT_MISMATCH_THRESHOLD, ge=0.0, description="Mismatch threshold"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Metrics timestamp"
    )

    @classmethod
    def from_counts(
        cls,
        emit_counts: dict[str, int],
        consume_counts: dict[str, int],
        monitored_topics: frozenset[str],
        threshold: float = DEFAULT_MISMATCH_THRESHOLD,
    ) -> ModelWiringHealthMetrics:
        """Create aggregate metrics from emission and consumption counts.

        Args:
            emit_counts: Per-topic emission counts.
            consume_counts: Per-topic consumption counts.
            monitored_topics: Set of topics being monitored.
            threshold: Mismatch threshold for health determination.

        Returns:
            ModelWiringHealthMetrics with per-topic and aggregate health.
        """
        topic_metrics: list[ModelTopicWiringHealth] = []

        for topic in sorted(monitored_topics):
            emit = emit_counts.get(topic, 0)
            consume = consume_counts.get(topic, 0)
            topic_health = ModelTopicWiringHealth.from_counts(
                topic=topic,
                emit_count=emit,
                consume_count=consume,
                threshold=threshold,
            )
            topic_metrics.append(topic_health)

        unhealthy_count = sum(1 for t in topic_metrics if not t.is_healthy)
        overall_healthy = unhealthy_count == 0

        return cls(
            topics=tuple(topic_metrics),
            overall_healthy=overall_healthy,
            unhealthy_count=unhealthy_count,
            threshold=threshold,
        )

    def to_prometheus_metrics(self) -> dict[str, float]:
        """Export metrics in Prometheus-compatible format.

        Returns:
            Dictionary of metric name -> value pairs suitable for
            Prometheus gauge/counter registration.

        Format:
            - wiring_health_emit_total{topic="..."}: Emission count
            - wiring_health_consume_total{topic="..."}: Consumption count
            - wiring_health_mismatch_ratio{topic="..."}: Mismatch ratio
        """
        metrics: dict[str, float] = {}

        for topic_health in self.topics:
            safe_topic = topic_health.topic.replace(".", "_")
            metrics[f"wiring_health_emit_total_{safe_topic}"] = float(
                topic_health.emit_count
            )
            metrics[f"wiring_health_consume_total_{safe_topic}"] = float(
                topic_health.consume_count
            )
            metrics[f"wiring_health_mismatch_ratio_{safe_topic}"] = (
                topic_health.mismatch_ratio
            )

        # Aggregate metrics
        metrics["wiring_health_overall_healthy"] = 1.0 if self.overall_healthy else 0.0
        metrics["wiring_health_unhealthy_count"] = float(self.unhealthy_count)

        return metrics


__all__ = [
    "ModelWiringHealthMetrics",
]
