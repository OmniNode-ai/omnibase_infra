# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for MetricCollector [OMN-6777]."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from omnibase_infra.services.eval.metric_collector import MetricCollector, MetricEvent


@pytest.fixture
def collector() -> MetricCollector:
    return MetricCollector(
        correlation_id="test-run-123",
        window_start=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _make_event(
    correlation_id: str = "test-run-123",
    metric_name: str = "latency_ms",
    metric_value: float = 100.0,
    timestamp: datetime | None = None,
) -> MetricEvent:
    return MetricEvent(
        topic="onex.evt.test.v1",
        correlation_id=correlation_id,
        timestamp=timestamp or datetime(2026, 1, 1, 0, 30, tzinfo=UTC),
        metric_name=metric_name,
        metric_value=metric_value,
    )


@pytest.mark.unit
class TestMetricCollector:
    def test_record_matching_event(self, collector: MetricCollector) -> None:
        event = _make_event()
        assert collector.record_event(event) is True
        assert collector.event_count == 1

    def test_reject_wrong_correlation_id(self, collector: MetricCollector) -> None:
        event = _make_event(correlation_id="wrong-id")
        assert collector.record_event(event) is False
        assert collector.event_count == 0

    def test_reject_event_before_window(self, collector: MetricCollector) -> None:
        event = _make_event(timestamp=datetime(2025, 12, 31, tzinfo=UTC))
        assert collector.record_event(event) is False

    def test_reject_event_after_window_closed(self, collector: MetricCollector) -> None:
        collector.stop_collecting()
        future = datetime(2099, 1, 1, tzinfo=UTC)
        event = _make_event(timestamp=future)
        assert collector.record_event(event) is False

    def test_get_metrics_by_name(self, collector: MetricCollector) -> None:
        collector.record_event(_make_event(metric_name="latency_ms", metric_value=100))
        collector.record_event(_make_event(metric_name="latency_ms", metric_value=200))
        collector.record_event(_make_event(metric_name="token_count", metric_value=50))
        assert collector.get_metrics_by_name("latency_ms") == [100, 200]

    def test_get_average(self, collector: MetricCollector) -> None:
        collector.record_event(_make_event(metric_name="latency_ms", metric_value=100))
        collector.record_event(_make_event(metric_name="latency_ms", metric_value=200))
        assert collector.get_average("latency_ms") == 150.0
        assert collector.get_average("nonexistent") is None

    def test_get_summary(self, collector: MetricCollector) -> None:
        collector.record_event(_make_event(metric_name="latency_ms", metric_value=100))
        collector.record_event(_make_event(metric_name="latency_ms", metric_value=200))
        summary = collector.get_summary()
        assert summary["correlation_id"] == "test-run-123"
        assert summary["event_count"] == 2
        assert summary["latency_ms_avg"] == 150.0

    def test_is_collecting(self, collector: MetricCollector) -> None:
        assert collector.is_collecting is True
        collector.stop_collecting()
        assert collector.is_collecting is False
