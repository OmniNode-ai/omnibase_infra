# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Eval Metric Collector.

Subscribes to Kafka topics during an eval run and collects metrics
scoped to a time window and correlation ID. Used by the eval
orchestrator to gather runtime metrics during A/B testing.

Related:
    - OMN-6777: Build metric collector from Kafka events
    - OMN-6773: Eval runner service
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricEvent:
    """A single metric event captured from the Kafka bus.

    Attributes:
        topic: Kafka topic the event was consumed from.
        correlation_id: Correlation ID linking the event to an eval run.
        timestamp: When the event was produced.
        metric_name: Name of the metric (e.g., 'latency_ms', 'token_count').
        metric_value: Numeric value of the metric.
        metadata: Additional event metadata.
    """

    topic: str
    correlation_id: str
    timestamp: datetime
    metric_name: str
    metric_value: float
    metadata: dict[str, str] = field(default_factory=dict)


class MetricCollector:
    """Collects metrics from Kafka events scoped to a time window.

    The collector buffers events matching a given correlation ID within
    a time window, then provides aggregated metrics for reporting.

    Args:
        correlation_id: The eval run correlation ID to filter events.
        window_start: Start of the collection window.
        window_end: End of the collection window (None = still collecting).
        topics: Kafka topics to subscribe to for metric collection.
    """

    def __init__(
        self,
        correlation_id: str,
        window_start: datetime | None = None,
        topics: list[str] | None = None,
    ) -> None:
        self._correlation_id = correlation_id
        self._window_start = window_start or datetime.now(UTC)
        self._window_end: datetime | None = None
        self._topics = topics or []
        self._events: list[MetricEvent] = []

    @property
    def correlation_id(self) -> str:
        return self._correlation_id

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def is_collecting(self) -> bool:
        return self._window_end is None

    def record_event(self, event: MetricEvent) -> bool:
        """Record a metric event if it matches the correlation ID, topic, and window.

        Returns True if the event was accepted, False if filtered out.
        """
        if event.correlation_id != self._correlation_id:
            return False
        if self._topics and event.topic not in self._topics:
            return False
        if event.timestamp < self._window_start:
            return False
        if self._window_end is not None and event.timestamp > self._window_end:
            return False
        self._events.append(event)
        return True

    def stop_collecting(self) -> None:
        """Close the collection window."""
        self._window_end = datetime.now(UTC)

    def get_metrics_by_name(self, metric_name: str) -> list[float]:
        """Get all values for a specific metric name."""
        return [e.metric_value for e in self._events if e.metric_name == metric_name]

    def get_average(self, metric_name: str) -> float | None:
        """Get the average value for a metric, or None if no data."""
        values = self.get_metrics_by_name(metric_name)
        if not values:
            return None
        return sum(values) / len(values)

    def get_summary(self) -> dict[str, float | int | str]:
        """Get a summary of all collected metrics.

        Returns a dict with:
            - correlation_id: The run's correlation ID
            - event_count: Total events collected
            - window_start: ISO timestamp of collection start
            - window_end: ISO timestamp of collection end (or 'active')
            - Per-metric averages as metric_name_avg keys
        """
        summary: dict[str, float | int | str] = {
            "correlation_id": self._correlation_id,
            "event_count": len(self._events),
            "window_start": self._window_start.isoformat(),
            "window_end": self._window_end.isoformat()
            if self._window_end
            else "active",
        }

        metric_names = {e.metric_name for e in self._events}
        for name in sorted(metric_names):
            avg = self.get_average(name)
            if avg is not None:
                summary[f"{name}_avg"] = avg

        return summary


__all__: list[str] = ["MetricCollector", "MetricEvent"]
