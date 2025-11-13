"""
Unit tests for alerting system.

Tests AlertRuleEngine and alert notifiers.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from omninode_bridge.agents.metrics.alerting.notifiers import (
    AlertNotifier,
    LogAlertNotifier,
)
from omninode_bridge.agents.metrics.alerting.rules import AlertRuleEngine
from omninode_bridge.agents.metrics.models import (
    Alert,
    AlertRule,
    AlertSeverity,
    Metric,
    MetricType,
)


class TestAlertRuleEngine:
    """Tests for AlertRuleEngine."""

    def test_engine_creation_empty(self):
        """Test creating engine with no rules."""
        engine = AlertRuleEngine()

        assert len(engine.get_rules()) == 0

    def test_engine_creation_with_rules(self):
        """Test creating engine with rules."""
        rules = [
            AlertRule(
                metric_name="test1",
                threshold=100.0,
                operator="gt",
                severity=AlertSeverity.WARNING,
            ),
            AlertRule(
                metric_name="test2",
                threshold=50.0,
                operator="lt",
                severity=AlertSeverity.CRITICAL,
            ),
        ]

        engine = AlertRuleEngine(rules=rules)

        assert len(engine.get_rules()) == 2

    def test_add_rule(self):
        """Test adding rule to engine."""
        engine = AlertRuleEngine()

        rule = AlertRule(
            metric_name="test",
            threshold=100.0,
            operator="gt",
            severity=AlertSeverity.WARNING,
        )

        engine.add_rule(rule)

        assert len(engine.get_rules()) == 1

    @pytest.mark.asyncio
    async def test_evaluate_metric_no_rules(self):
        """Test evaluating metric with no rules."""
        engine = AlertRuleEngine()

        metric = Metric(
            metric_name="test",
            metric_type=MetricType.TIMING,
            value=100.0,
            unit="ms",
        )

        alerts = await engine.evaluate_metric(metric)

        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_evaluate_metric_triggered(self):
        """Test evaluating metric that triggers alert."""
        rule = AlertRule(
            metric_name="test_metric",
            threshold=100.0,
            operator="gt",
            severity=AlertSeverity.CRITICAL,
        )

        engine = AlertRuleEngine(rules=[rule])

        metric = Metric(
            metric_name="test_metric",
            metric_type=MetricType.TIMING,
            value=150.0,
            unit="ms",
        )

        alerts = await engine.evaluate_metric(metric)

        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL
        assert alerts[0].metric_name == "test_metric"
        assert alerts[0].actual_value == 150.0
        assert alerts[0].threshold == 100.0

    @pytest.mark.asyncio
    async def test_evaluate_metric_not_triggered(self):
        """Test evaluating metric that does not trigger alert."""
        rule = AlertRule(
            metric_name="test_metric",
            threshold=100.0,
            operator="gt",
            severity=AlertSeverity.WARNING,
        )

        engine = AlertRuleEngine(rules=[rule])

        metric = Metric(
            metric_name="test_metric",
            metric_type=MetricType.TIMING,
            value=50.0,
            unit="ms",
        )

        alerts = await engine.evaluate_metric(metric)

        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_evaluate_metric_multiple_rules(self):
        """Test evaluating metric against multiple rules."""
        rules = [
            AlertRule(
                metric_name="test_metric",
                threshold=100.0,
                operator="gt",
                severity=AlertSeverity.WARNING,
            ),
            AlertRule(
                metric_name="test_metric",
                threshold=150.0,
                operator="gt",
                severity=AlertSeverity.CRITICAL,
            ),
        ]

        engine = AlertRuleEngine(rules=rules)

        # Metric that triggers both rules
        metric = Metric(
            metric_name="test_metric",
            metric_type=MetricType.TIMING,
            value=160.0,
            unit="ms",
        )

        alerts = await engine.evaluate_metric(metric)

        assert len(alerts) == 2
        severities = {alert.severity for alert in alerts}
        assert AlertSeverity.WARNING in severities
        assert AlertSeverity.CRITICAL in severities

    @pytest.mark.asyncio
    async def test_evaluate_metric_with_notifier(self):
        """Test alert notification is sent."""
        # Create mock notifier
        mock_notifier = AsyncMock(spec=AlertNotifier)

        rule = AlertRule(
            metric_name="test_metric",
            threshold=100.0,
            operator="gt",
            severity=AlertSeverity.CRITICAL,
        )

        engine = AlertRuleEngine(rules=[rule], notifiers=[mock_notifier])

        metric = Metric(
            metric_name="test_metric",
            metric_type=MetricType.TIMING,
            value=150.0,
            unit="ms",
        )

        alerts = await engine.evaluate_metric(metric)

        # Give notification task time to execute
        await asyncio.sleep(0.1)

        # Notifier should have been called
        mock_notifier.notify.assert_called_once()

    def test_get_rules_for_metric(self):
        """Test getting rules for specific metric."""
        rules = [
            AlertRule(
                metric_name="metric1",
                threshold=100.0,
                operator="gt",
                severity=AlertSeverity.WARNING,
            ),
            AlertRule(
                metric_name="metric2",
                threshold=50.0,
                operator="lt",
                severity=AlertSeverity.CRITICAL,
            ),
            AlertRule(
                metric_name="metric1",
                threshold=200.0,
                operator="gt",
                severity=AlertSeverity.CRITICAL,
            ),
        ]

        engine = AlertRuleEngine(rules=rules)

        metric1_rules = engine.get_rules_for_metric("metric1")
        assert len(metric1_rules) == 2

        metric2_rules = engine.get_rules_for_metric("metric2")
        assert len(metric2_rules) == 1

        metric3_rules = engine.get_rules_for_metric("metric3")
        assert len(metric3_rules) == 0


class TestLogAlertNotifier:
    """Tests for LogAlertNotifier."""

    @pytest.mark.asyncio
    async def test_log_notifier_notify(self):
        """Test log notifier sends alert."""
        notifier = LogAlertNotifier()

        alert = Alert(
            severity=AlertSeverity.WARNING,
            metric_name="test_metric",
            threshold=100.0,
            actual_value=150.0,
            message="Test alert",
        )

        # Should not raise error
        await notifier.notify(alert)

    @pytest.mark.asyncio
    async def test_log_notifier_multiple_severities(self):
        """Test log notifier handles all severity levels."""
        notifier = LogAlertNotifier()

        for severity in [
            AlertSeverity.INFO,
            AlertSeverity.WARNING,
            AlertSeverity.CRITICAL,
        ]:
            alert = Alert(
                severity=severity,
                metric_name="test_metric",
                threshold=100.0,
                actual_value=150.0,
                message=f"Test {severity} alert",
            )

            await notifier.notify(alert)
