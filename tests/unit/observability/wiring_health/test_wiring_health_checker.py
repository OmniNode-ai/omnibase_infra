# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for WiringHealthChecker and wiring health models."""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.event_bus.topic_constants import WIRING_HEALTH_MONITORED_TOPICS
from omnibase_infra.observability.wiring_health import (
    DEFAULT_MISMATCH_THRESHOLD,
    ModelTopicWiringHealth,
    ModelWiringHealthAlert,
    ModelWiringHealthMetrics,
    WiringHealthChecker,
)

pytestmark = pytest.mark.unit


class MockEmissionSource:
    """Mock emission count source for testing."""

    def __init__(self, counts: dict[str, int] | None = None) -> None:
        self._counts = counts or {}

    def get_emission_counts(self) -> dict[str, int]:
        return dict(self._counts)


class MockConsumptionSource:
    """Mock consumption count source for testing."""

    def __init__(self, counts: dict[str, int] | None = None) -> None:
        self._counts = counts or {}

    def get_consumption_counts(self) -> dict[str, int]:
        return dict(self._counts)


class TestModelTopicWiringHealth:
    """Tests for ModelTopicWiringHealth."""

    def test_from_counts_healthy(self) -> None:
        """Topic should be healthy when mismatch within threshold."""
        health = ModelTopicWiringHealth.from_counts(
            topic="test.topic.v1",
            emit_count=100,
            consume_count=98,  # 2% mismatch
            threshold=0.05,
        )

        assert health.emit_count == 100
        assert health.consume_count == 98
        assert health.mismatch_ratio == pytest.approx(0.02, abs=0.001)
        assert health.is_healthy is True

    def test_from_counts_unhealthy(self) -> None:
        """Topic should be unhealthy when mismatch exceeds threshold."""
        health = ModelTopicWiringHealth.from_counts(
            topic="test.topic.v1",
            emit_count=100,
            consume_count=85,  # 15% mismatch
            threshold=0.05,
        )

        assert health.mismatch_ratio == pytest.approx(0.15, abs=0.001)
        assert health.is_healthy is False

    def test_from_counts_zero_emit(self) -> None:
        """Should handle zero emissions without division by zero."""
        health = ModelTopicWiringHealth.from_counts(
            topic="test.topic.v1",
            emit_count=0,
            consume_count=0,
            threshold=0.05,
        )

        # mismatch = abs(0 - 0) / max(0, 1) = 0 / 1 = 0
        assert health.mismatch_ratio == 0.0
        assert health.is_healthy is True

    def test_from_counts_consume_exceeds_emit(self) -> None:
        """Should handle consumption exceeding emission (at-least-once)."""
        health = ModelTopicWiringHealth.from_counts(
            topic="test.topic.v1",
            emit_count=100,
            consume_count=110,  # 10% over-consumption (redelivery)
            threshold=0.05,
        )

        # mismatch = abs(100 - 110) / max(100, 1) = 10 / 100 = 0.1
        assert health.mismatch_ratio == pytest.approx(0.10, abs=0.001)
        assert health.is_healthy is False


class TestModelWiringHealthMetrics:
    """Tests for ModelWiringHealthMetrics."""

    def test_from_counts_all_healthy(self) -> None:
        """Should report overall healthy when all topics healthy."""
        emit_counts = {"topic1": 100, "topic2": 200}
        consume_counts = {"topic1": 98, "topic2": 195}
        monitored = frozenset(["topic1", "topic2"])

        metrics = ModelWiringHealthMetrics.from_counts(
            emit_counts=emit_counts,
            consume_counts=consume_counts,
            monitored_topics=monitored,
            threshold=0.05,
        )

        assert metrics.overall_healthy is True
        assert metrics.unhealthy_count == 0
        assert len(metrics.topics) == 2

    def test_from_counts_some_unhealthy(self) -> None:
        """Should report overall unhealthy when any topic unhealthy."""
        emit_counts = {"topic1": 100, "topic2": 200}
        consume_counts = {"topic1": 50, "topic2": 195}  # topic1 50% mismatch
        monitored = frozenset(["topic1", "topic2"])

        metrics = ModelWiringHealthMetrics.from_counts(
            emit_counts=emit_counts,
            consume_counts=consume_counts,
            monitored_topics=monitored,
            threshold=0.05,
        )

        assert metrics.overall_healthy is False
        assert metrics.unhealthy_count == 1

    def test_from_counts_missing_topic_data(self) -> None:
        """Should handle topics with no emission/consumption data."""
        emit_counts = {"topic1": 100}  # topic2 not emitted
        consume_counts = {"topic2": 50}  # topic1 not consumed
        monitored = frozenset(["topic1", "topic2"])

        metrics = ModelWiringHealthMetrics.from_counts(
            emit_counts=emit_counts,
            consume_counts=consume_counts,
            monitored_topics=monitored,
            threshold=0.05,
        )

        assert len(metrics.topics) == 2
        # topic1: 100 emit, 0 consume = 100% mismatch
        # topic2: 0 emit, 50 consume = 50/1 = 5000% mismatch (!)
        assert metrics.overall_healthy is False

    def test_to_prometheus_metrics(self) -> None:
        """Should export metrics in Prometheus-compatible format."""
        emit_counts = {"test.topic.v1": 100}
        consume_counts = {"test.topic.v1": 95}
        monitored = frozenset(["test.topic.v1"])

        metrics = ModelWiringHealthMetrics.from_counts(
            emit_counts=emit_counts,
            consume_counts=consume_counts,
            monitored_topics=monitored,
        )

        prom_metrics = metrics.to_prometheus_metrics()

        # Topic metrics should use underscores instead of dots
        assert "wiring_health_emit_total_test_topic_v1" in prom_metrics
        assert prom_metrics["wiring_health_emit_total_test_topic_v1"] == 100.0
        assert prom_metrics["wiring_health_overall_healthy"] == 1.0


class TestModelWiringHealthAlert:
    """Tests for ModelWiringHealthAlert."""

    def test_from_metrics_returns_none_when_healthy(self) -> None:
        """Should return None when all topics are healthy."""
        emit_counts = {"topic1": 100}
        consume_counts = {"topic1": 98}
        monitored = frozenset(["topic1"])

        metrics = ModelWiringHealthMetrics.from_counts(
            emit_counts=emit_counts,
            consume_counts=consume_counts,
            monitored_topics=monitored,
        )

        alert = ModelWiringHealthAlert.from_metrics(metrics, "dev")

        assert alert is None

    def test_from_metrics_returns_alert_when_unhealthy(self) -> None:
        """Should return alert when any topic is unhealthy."""
        emit_counts = {"topic1": 100}
        consume_counts = {"topic1": 50}  # 50% mismatch
        monitored = frozenset(["topic1"])

        metrics = ModelWiringHealthMetrics.from_counts(
            emit_counts=emit_counts,
            consume_counts=consume_counts,
            monitored_topics=monitored,
        )

        alert = ModelWiringHealthAlert.from_metrics(metrics, "prod")

        assert alert is not None
        assert alert.environment == "prod"
        assert "topic1" in alert.unhealthy_topics
        assert "1 topic exceeds" in alert.summary

    def test_to_slack_message(self) -> None:
        """Should format alert as Slack message payload."""
        emit_counts = {"topic1": 100, "topic2": 200}
        consume_counts = {"topic1": 50, "topic2": 100}  # Both 50% mismatch
        monitored = frozenset(["topic1", "topic2"])

        metrics = ModelWiringHealthMetrics.from_counts(
            emit_counts=emit_counts,
            consume_counts=consume_counts,
            monitored_topics=monitored,
        )

        alert = ModelWiringHealthAlert.from_metrics(metrics, "prod")
        assert alert is not None

        slack_msg = alert.to_slack_message()

        assert "text" in slack_msg
        assert "blocks" in slack_msg
        assert "prod" in slack_msg["text"]


class TestWiringHealthChecker:
    """Tests for WiringHealthChecker."""

    @pytest.fixture
    def topic(self) -> str:
        """Get first monitored topic."""
        return WIRING_HEALTH_MONITORED_TOPICS[0]

    def test_compute_health_all_healthy(self, topic: str) -> None:
        """Should return healthy when emit/consume match within threshold."""
        emission_source = MockEmissionSource({topic: 100})
        consumption_source = MockConsumptionSource({topic: 98})

        handler = WiringHealthChecker(
            emission_source=emission_source,
            consumption_source=consumption_source,
            environment="dev",
        )

        metrics = handler.compute_health()

        assert metrics.overall_healthy is True

    def test_compute_health_unhealthy(self, topic: str) -> None:
        """Should return unhealthy when mismatch exceeds threshold."""
        emission_source = MockEmissionSource({topic: 100})
        consumption_source = MockConsumptionSource({topic: 50})

        handler = WiringHealthChecker(
            emission_source=emission_source,
            consumption_source=consumption_source,
            environment="dev",
        )

        metrics = handler.compute_health()

        assert metrics.overall_healthy is False

    def test_compute_health_with_alert(self, topic: str) -> None:
        """Should generate alert when unhealthy."""
        emission_source = MockEmissionSource({topic: 100})
        consumption_source = MockConsumptionSource({topic: 50})

        handler = WiringHealthChecker(
            emission_source=emission_source,
            consumption_source=consumption_source,
            environment="prod",
        )

        metrics, alert = handler.compute_health_with_alert()

        assert metrics.overall_healthy is False
        assert alert is not None
        assert alert.environment == "prod"

    def test_compute_health_with_alert_healthy(self, topic: str) -> None:
        """Should not generate alert when healthy."""
        emission_source = MockEmissionSource({topic: 100})
        consumption_source = MockConsumptionSource({topic: 98})

        handler = WiringHealthChecker(
            emission_source=emission_source,
            consumption_source=consumption_source,
            environment="dev",
        )

        metrics, alert = handler.compute_health_with_alert()

        assert metrics.overall_healthy is True
        assert alert is None

    def test_to_health_response_healthy(self, topic: str) -> None:
        """Should format healthy response correctly."""
        emission_source = MockEmissionSource({topic: 100})
        consumption_source = MockConsumptionSource({topic: 98})

        handler = WiringHealthChecker(
            emission_source=emission_source,
            consumption_source=consumption_source,
            environment="dev",
        )

        metrics = handler.compute_health()
        response = handler.to_health_response(metrics)

        assert response["status"] == "healthy"
        assert response["overall_healthy"] is True
        assert "topics" in response

    def test_to_health_response_degraded(self, topic: str) -> None:
        """Should format unhealthy response as degraded."""
        emission_source = MockEmissionSource({topic: 100})
        consumption_source = MockConsumptionSource({topic: 50})

        handler = WiringHealthChecker(
            emission_source=emission_source,
            consumption_source=consumption_source,
            environment="dev",
        )

        metrics = handler.compute_health()
        response = handler.to_health_response(metrics)

        assert response["status"] == "degraded"
        assert response["overall_healthy"] is False

    def test_custom_threshold(self, topic: str) -> None:
        """Should respect custom mismatch threshold."""
        emission_source = MockEmissionSource({topic: 100})
        consumption_source = MockConsumptionSource({topic: 85})  # 15% mismatch

        # With default threshold (5%), should be unhealthy
        handler_strict = WiringHealthChecker(
            emission_source=emission_source,
            consumption_source=consumption_source,
            environment="dev",
            threshold=0.05,
        )
        assert handler_strict.compute_health().overall_healthy is False

        # With relaxed threshold (20%), should be healthy
        handler_relaxed = WiringHealthChecker(
            emission_source=emission_source,
            consumption_source=consumption_source,
            environment="dev",
            threshold=0.20,
        )
        assert handler_relaxed.compute_health().overall_healthy is True

    def test_correlation_id_propagation(self, topic: str) -> None:
        """Should propagate correlation ID through alert."""
        emission_source = MockEmissionSource({topic: 100})
        consumption_source = MockConsumptionSource({topic: 50})

        handler = WiringHealthChecker(
            emission_source=emission_source,
            consumption_source=consumption_source,
            environment="dev",
        )

        correlation_id = uuid4()
        _metrics, alert = handler.compute_health_with_alert(correlation_id)

        assert alert is not None
        assert alert.correlation_id == correlation_id
