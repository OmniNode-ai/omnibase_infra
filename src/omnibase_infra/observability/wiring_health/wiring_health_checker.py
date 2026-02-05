# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Wiring health checker for computing emission/consumption comparisons.

This module provides the checker that computes wiring health metrics by
comparing emission counts (from EventBusKafka) with consumption counts
(from EventBusSubcontractWiring). It exposes metrics for Prometheus and
can emit alerts when thresholds are exceeded.

Architecture:
    The checker is triggered by health endpoint requests (/health/wiring).
    Prometheus scraping becomes the scheduler - no background loop needed.

    ┌─────────────────┐    request    ┌─────────────────────┐
    │   Prometheus    │──────────────▶│  /health/wiring     │
    │   (scheduler)   │               │     endpoint        │
    └─────────────────┘               └──────────┬──────────┘
                                                 │
                                                 ▼
                            ┌─────────────────────────────────────┐
                            │      WiringHealthChecker            │
                            │  1. Get emission counts             │
                            │  2. Get consumption counts          │
                            │  3. Compute mismatch ratios         │
                            │  4. Update Prometheus metrics       │
                            │  5. Return health status JSON       │
                            │  6. Emit alert intent if unhealthy  │
                            └─────────────────────────────────────┘

Design Decisions:
    - No background loop - Prometheus scrape triggers computation
    - Direct Prometheus metrics via prometheus_client (not event-based)
    - Alert intents emitted but not blocking (best-effort)
    - Health endpoint returns JSON for Kubernetes probes

See Also:
    - OMN-1895: Wiring health monitor implementation
    - MixinEmissionCounter: Emission counting on EventBusKafka
    - MixinConsumptionCounter: Consumption counting on EventBusSubcontractWiring
"""

from __future__ import annotations

import logging
from uuid import UUID, uuid4

from omnibase_infra.event_bus.topic_constants import WIRING_HEALTH_MONITORED_TOPICS
from omnibase_infra.observability.wiring_health.model_topic_wiring_health import (
    DEFAULT_MISMATCH_THRESHOLD,
)
from omnibase_infra.observability.wiring_health.model_wiring_health_alert import (
    ModelWiringHealthAlert,
)
from omnibase_infra.observability.wiring_health.model_wiring_health_metrics import (
    ModelWiringHealthMetrics,
)
from omnibase_infra.observability.wiring_health.protocol_consumption_count_source import (
    ProtocolConsumptionCountSource,
)
from omnibase_infra.observability.wiring_health.protocol_emission_count_source import (
    ProtocolEmissionCountSource,
)

_logger = logging.getLogger(__name__)


class WiringHealthChecker:
    """Checker for computing and exposing wiring health metrics.

    Computes wiring health by comparing emission counts from the event bus
    with consumption counts from the subcontract wiring. Triggered by health
    endpoint requests (Prometheus scrape is the scheduler).

    Attributes:
        emission_source: Source of emission counts (EventBusKafka).
        consumption_source: Source of consumption counts (EventBusSubcontractWiring).
        environment: Environment identifier for alerts.
        threshold: Mismatch threshold for health determination.

    Example:
        >>> checker = WiringHealthChecker(
        ...     emission_source=event_bus,
        ...     consumption_source=wiring,
        ...     environment="prod",
        ... )
        >>> metrics = checker.compute_health()
        >>> print(f"Overall healthy: {metrics.overall_healthy}")
    """

    def __init__(
        self,
        emission_source: ProtocolEmissionCountSource,
        consumption_source: ProtocolConsumptionCountSource,
        environment: str,
        threshold: float = DEFAULT_MISMATCH_THRESHOLD,
        prometheus_sink: object | None = None,
    ) -> None:
        """Initialize the wiring health checker.

        Args:
            emission_source: Source of emission counts (e.g., EventBusKafka).
            consumption_source: Source of consumption counts (e.g., Wiring).
            environment: Environment identifier for alert context.
            threshold: Mismatch threshold (default 5%).
            prometheus_sink: Optional SinkMetricsPrometheus for metric export.
        """
        self._emission_source = emission_source
        self._consumption_source = consumption_source
        self._environment = environment
        self._threshold = threshold
        self._prometheus_sink = prometheus_sink
        self._monitored_topics = frozenset(WIRING_HEALTH_MONITORED_TOPICS)

        _logger.info(
            "WiringHealthChecker initialized",
            extra={
                "environment": environment,
                "threshold": threshold,
                "monitored_topics": list(self._monitored_topics),
            },
        )

    def compute_health(
        self,
        correlation_id: UUID | None = None,
    ) -> ModelWiringHealthMetrics:
        """Compute current wiring health metrics.

        Gathers emission and consumption counts, computes mismatch ratios,
        and returns the health status. This is the primary method called
        by the health endpoint.

        Args:
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelWiringHealthMetrics with per-topic and aggregate health.
        """
        correlation_id = correlation_id or uuid4()

        # Gather counts
        emit_counts = self._emission_source.get_emission_counts()
        consume_counts = self._consumption_source.get_consumption_counts()

        # Compute metrics
        metrics = ModelWiringHealthMetrics.from_counts(
            emit_counts=emit_counts,
            consume_counts=consume_counts,
            monitored_topics=self._monitored_topics,
            threshold=self._threshold,
        )

        _logger.debug(
            "Computed wiring health",
            extra={
                "correlation_id": str(correlation_id),
                "overall_healthy": metrics.overall_healthy,
                "unhealthy_count": metrics.unhealthy_count,
            },
        )

        # Update Prometheus metrics if sink available
        if self._prometheus_sink is not None:
            self._update_prometheus_metrics(metrics, correlation_id)

        return metrics

    def compute_health_with_alert(
        self,
        correlation_id: UUID | None = None,
    ) -> tuple[ModelWiringHealthMetrics, ModelWiringHealthAlert | None]:
        """Compute health and generate alert if unhealthy.

        Combines compute_health() with alert generation for use cases
        that need both the metrics and the alert payload.

        Args:
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Tuple of (metrics, alert) where alert is None if healthy.
        """
        correlation_id = correlation_id or uuid4()

        metrics = self.compute_health(correlation_id)

        alert = ModelWiringHealthAlert.from_metrics(
            metrics=metrics,
            environment=self._environment,
            correlation_id=correlation_id,
        )

        if alert is not None:
            _logger.warning(
                "Wiring health alert generated",
                extra={
                    "correlation_id": str(correlation_id),
                    "unhealthy_topics": list(alert.unhealthy_topics),
                    "summary": alert.summary,
                },
            )

        return metrics, alert

    def _update_prometheus_metrics(
        self, metrics: ModelWiringHealthMetrics, correlation_id: UUID
    ) -> None:
        """Update Prometheus metrics from health metrics.

        Args:
            metrics: The computed wiring health metrics.
            correlation_id: Correlation ID for error tracing.
        """
        try:
            # Update gauges for each topic
            for topic_health in metrics.topics:
                labels = {"topic": topic_health.topic}

                self._prometheus_sink.set_gauge(  # type: ignore[union-attr]
                    "wiring_health_emit_total",
                    labels,
                    float(topic_health.emit_count),
                )
                self._prometheus_sink.set_gauge(  # type: ignore[union-attr]
                    "wiring_health_consume_total",
                    labels,
                    float(topic_health.consume_count),
                )
                self._prometheus_sink.set_gauge(  # type: ignore[union-attr]
                    "wiring_health_mismatch_ratio",
                    labels,
                    topic_health.mismatch_ratio,
                )

            # Aggregate metrics
            self._prometheus_sink.set_gauge(  # type: ignore[union-attr]
                "wiring_health_overall_healthy",
                {},
                1.0 if metrics.overall_healthy else 0.0,
            )
            self._prometheus_sink.set_gauge(  # type: ignore[union-attr]
                "wiring_health_unhealthy_count",
                {},
                float(metrics.unhealthy_count),
            )

            _logger.debug("Updated Prometheus wiring health metrics")

        except Exception as e:
            _logger.warning(
                "Failed to update Prometheus metrics",
                extra={"error": str(e), "correlation_id": str(correlation_id)},
            )

    def to_health_response(
        self,
        metrics: ModelWiringHealthMetrics,
    ) -> dict[str, object]:
        """Convert metrics to health endpoint JSON response.

        Args:
            metrics: The computed wiring health metrics.

        Returns:
            Dictionary suitable for JSON serialization in health endpoint.
        """
        return {
            "status": "healthy" if metrics.overall_healthy else "degraded",
            "overall_healthy": metrics.overall_healthy,
            "unhealthy_count": metrics.unhealthy_count,
            "threshold": metrics.threshold,
            "timestamp": metrics.timestamp.isoformat(),
            "topics": [
                {
                    "topic": t.topic,
                    "emit_count": t.emit_count,
                    "consume_count": t.consume_count,
                    "mismatch_ratio": round(t.mismatch_ratio, 4),
                    "is_healthy": t.is_healthy,
                }
                for t in metrics.topics
            ],
        }


__all__ = [
    "WiringHealthChecker",
]
