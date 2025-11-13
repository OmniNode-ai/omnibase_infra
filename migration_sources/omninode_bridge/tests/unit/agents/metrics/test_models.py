"""
Unit tests for metrics models.

Tests Pydantic v2 models for type safety and validation.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from omninode_bridge.agents.metrics.models import (
    Alert,
    AlertEvent,
    AlertRule,
    AlertSeverity,
    Metric,
    MetricEvent,
    MetricType,
)


class TestMetric:
    """Tests for Metric model."""

    def test_metric_creation_valid(self):
        """Test creating valid metric."""
        metric = Metric(
            metric_name="test_metric",
            metric_type=MetricType.TIMING,
            value=42.5,
            unit="ms",
            tags={"env": "test"},
        )

        assert metric.metric_name == "test_metric"
        assert metric.metric_type == MetricType.TIMING
        assert metric.value == 42.5
        assert metric.unit == "ms"
        assert metric.tags == {"env": "test"}
        assert isinstance(metric.timestamp, datetime)
        assert len(metric.metric_id) > 0

    def test_metric_creation_minimal(self):
        """Test creating metric with minimal required fields."""
        metric = Metric(
            metric_name="minimal",
            metric_type=MetricType.COUNTER,
            value=1.0,
            unit="count",
        )

        assert metric.metric_name == "minimal"
        assert metric.tags == {}
        assert metric.correlation_id is None

    def test_metric_creation_invalid_name(self):
        """Test metric creation with invalid name fails."""
        with pytest.raises(ValidationError):
            Metric(
                metric_name="",  # Empty name not allowed
                metric_type=MetricType.TIMING,
                value=1.0,
                unit="ms",
            )

    def test_metric_type_enum(self):
        """Test metric type enumeration."""
        assert MetricType.TIMING == "timing"
        assert MetricType.COUNTER == "counter"
        assert MetricType.GAUGE == "gauge"
        assert MetricType.RATE == "rate"


class TestAlert:
    """Tests for Alert model."""

    def test_alert_creation_valid(self):
        """Test creating valid alert."""
        alert = Alert(
            severity=AlertSeverity.CRITICAL,
            metric_name="test_metric",
            threshold=100.0,
            actual_value=150.0,
            message="Threshold exceeded",
        )

        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.metric_name == "test_metric"
        assert alert.threshold == 100.0
        assert alert.actual_value == 150.0
        assert alert.message == "Threshold exceeded"
        assert len(alert.alert_id) > 0

    def test_alert_immutable(self):
        """Test alert is immutable."""
        alert = Alert(
            severity=AlertSeverity.WARNING,
            metric_name="test",
            threshold=50.0,
            actual_value=60.0,
            message="Test",
        )

        with pytest.raises(ValidationError):
            alert.severity = AlertSeverity.INFO


class TestAlertRule:
    """Tests for AlertRule model."""

    def test_alert_rule_creation_valid(self):
        """Test creating valid alert rule."""
        rule = AlertRule(
            metric_name="test_metric",
            threshold=100.0,
            operator="gt",
            severity=AlertSeverity.WARNING,
        )

        assert rule.metric_name == "test_metric"
        assert rule.threshold == 100.0
        assert rule.operator == "gt"
        assert rule.severity == AlertSeverity.WARNING

    def test_alert_rule_evaluate_gt_triggered(self):
        """Test evaluate() triggers alert for gt operator."""
        rule = AlertRule(
            metric_name="test_metric",
            threshold=100.0,
            operator="gt",
            severity=AlertSeverity.CRITICAL,
        )

        metric = Metric(
            metric_name="test_metric",
            metric_type=MetricType.TIMING,
            value=150.0,
            unit="ms",
        )

        alert = rule.evaluate(metric)
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.actual_value == 150.0

    def test_alert_rule_evaluate_gt_not_triggered(self):
        """Test evaluate() does not trigger alert when threshold not exceeded."""
        rule = AlertRule(
            metric_name="test_metric",
            threshold=100.0,
            operator="gt",
            severity=AlertSeverity.WARNING,
        )

        metric = Metric(
            metric_name="test_metric",
            metric_type=MetricType.TIMING,
            value=50.0,
            unit="ms",
        )

        alert = rule.evaluate(metric)
        assert alert is None

    def test_alert_rule_evaluate_lt_triggered(self):
        """Test evaluate() triggers alert for lt operator."""
        rule = AlertRule(
            metric_name="cache_hit_rate",
            threshold=80.0,
            operator="lt",
            severity=AlertSeverity.WARNING,
        )

        metric = Metric(
            metric_name="cache_hit_rate",
            metric_type=MetricType.RATE,
            value=60.0,
            unit="%",
        )

        alert = rule.evaluate(metric)
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING

    def test_alert_rule_evaluate_wrong_metric(self):
        """Test evaluate() returns None for wrong metric."""
        rule = AlertRule(
            metric_name="test_metric",
            threshold=100.0,
            operator="gt",
            severity=AlertSeverity.WARNING,
        )

        metric = Metric(
            metric_name="other_metric",
            metric_type=MetricType.TIMING,
            value=150.0,
            unit="ms",
        )

        alert = rule.evaluate(metric)
        assert alert is None

    def test_alert_rule_invalid_operator(self):
        """Test creating rule with invalid operator fails."""
        with pytest.raises(ValidationError):
            AlertRule(
                metric_name="test",
                threshold=100.0,
                operator="invalid",  # Not in (gt, lt, gte, lte)
                severity=AlertSeverity.WARNING,
            )


class TestMetricEvent:
    """Tests for Kafka metric event."""

    def test_metric_event_creation(self):
        """Test creating metric event in OnexEnvelopeV1 format."""
        from omninode_bridge.agents.metrics.models import MetricEventPayload

        payload = MetricEventPayload(
            metric_id="test-id",
            metric_name="test_metric",
            metric_type="timing",
            value=42.5,
            unit="ms",
            tags={"env": "test"},
        )

        event = MetricEvent(payload=payload)

        assert event.event_type == "metric.recorded"
        assert event.event_version == "v1"
        assert event.source_service == "omninode-bridge"
        assert len(event.event_id) > 0
        assert event.payload == payload

    def test_metric_event_immutable(self):
        """Test metric event is immutable."""
        from omninode_bridge.agents.metrics.models import MetricEventPayload

        payload = MetricEventPayload(
            metric_id="test-id",
            metric_name="test",
            metric_type="timing",
            value=1.0,
            unit="ms",
            tags={},
        )

        event = MetricEvent(payload=payload)

        with pytest.raises(ValidationError):
            event.event_type = "other.type"


class TestAlertEvent:
    """Tests for Kafka alert event."""

    def test_alert_event_creation(self):
        """Test creating alert event in OnexEnvelopeV1 format."""
        from omninode_bridge.agents.metrics.models import AlertEventPayload

        payload = AlertEventPayload(
            alert_id="test-id",
            severity=AlertSeverity.CRITICAL,
            metric_name="test_metric",
            threshold=100.0,
            actual_value=150.0,
            message="Threshold exceeded",
            tags={"env": "test"},
        )

        event = AlertEvent(payload=payload)

        assert event.event_type == "alert.triggered"
        assert event.event_version == "v1"
        assert event.source_service == "omninode-bridge"
        assert len(event.event_id) > 0
        assert event.payload == payload
