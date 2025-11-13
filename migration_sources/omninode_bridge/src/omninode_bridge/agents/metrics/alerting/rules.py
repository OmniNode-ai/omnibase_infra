"""
Alert rule engine for threshold-based alerting.

Evaluates metrics against configured rules and triggers notifications.
"""

import asyncio
import logging
from typing import Optional

from omninode_bridge.agents.metrics.alerting.notifiers import AlertNotifier
from omninode_bridge.agents.metrics.models import Alert, AlertRule, Metric

logger = logging.getLogger(__name__)


class AlertRuleEngine:
    """
    Alert rule engine with threshold checking.

    Features:
    - Multiple alert rules per metric
    - Configurable thresholds (CRITICAL, WARNING, INFO)
    - Multiple notification channels
    - Async notification (non-blocking)

    Usage:
        engine = AlertRuleEngine(rules=[...], notifiers=[...])
        await engine.evaluate_metric(metric)
    """

    def __init__(
        self,
        rules: Optional[list[AlertRule]] = None,
        notifiers: Optional[list[AlertNotifier]] = None,
    ):
        """
        Initialize alert rule engine.

        Args:
            rules: List of alert rules
            notifiers: List of alert notifiers
        """
        self._rules = rules or []
        self._notifiers = notifiers or []
        logger.info(
            f"AlertRuleEngine initialized: {len(self._rules)} rules, "
            f"{len(self._notifiers)} notifiers"
        )

    def add_rule(self, rule: AlertRule) -> None:
        """
        Add alert rule.

        Args:
            rule: Alert rule to add
        """
        self._rules.append(rule)
        logger.debug(
            f"Added alert rule: {rule.metric_name} {rule.operator} {rule.threshold}"
        )

    def add_notifier(self, notifier: AlertNotifier) -> None:
        """
        Add alert notifier.

        Args:
            notifier: Alert notifier to add
        """
        self._notifiers.append(notifier)
        logger.debug(f"Added alert notifier: {notifier.__class__.__name__}")

    async def evaluate_metric(self, metric: Metric) -> list[Alert]:
        """
        Evaluate metric against all rules.

        Args:
            metric: Metric to evaluate

        Returns:
            List of triggered alerts
        """
        alerts: list[Alert] = []

        # Evaluate all rules
        for rule in self._rules:
            alert = rule.evaluate(metric)
            if alert:
                alerts.append(alert)
                logger.info(f"Alert triggered: {alert.severity} - {alert.message}")

                # Notify (async, non-blocking)
                asyncio.create_task(self._notify(alert))

        return alerts

    async def _notify(self, alert: Alert) -> None:
        """
        Send alert to all notifiers.

        Args:
            alert: Alert to send
        """
        if not self._notifiers:
            return

        # Send to all notifiers in parallel
        tasks = [notifier.notify(alert) for notifier in self._notifiers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Notifier {self._notifiers[i].__class__.__name__} "
                    f"failed: {result}"
                )

    def get_rules(self) -> list[AlertRule]:
        """Get all alert rules."""
        return self._rules.copy()

    def get_rules_for_metric(self, metric_name: str) -> list[AlertRule]:
        """
        Get alert rules for specific metric.

        Args:
            metric_name: Metric name

        Returns:
            List of alert rules for metric
        """
        return [rule for rule in self._rules if rule.metric_name == metric_name]
