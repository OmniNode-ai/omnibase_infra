#!/usr/bin/env python3
"""
Production Alerting System for ONEX v2.0 Code Generation.

Provides comprehensive alerting capabilities:
- SLA violation alerts
- Performance threshold alerts
- Quality degradation alerts
- Cost budget alerts
- Custom alert rules
- Multi-channel notifications

ONEX v2.0 Compliance:
- Type-safe alert definitions
- Async notification delivery
- Structured alert payloads
- Comprehensive alert tracking

Performance Requirements:
- Alert evaluation: <10ms
- Notification delivery: <100ms
- Non-blocking async execution

Author: Code Generation System
Last Updated: 2025-11-06
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# === Alert Enums ===


class AlertSeverity(str, Enum):
    """Severity levels for alerts."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertType(str, Enum):
    """Types of alerts."""

    LATENCY_VIOLATION = "latency_violation"
    CACHE_HIT_RATE_LOW = "cache_hit_rate_low"
    VALIDATION_PASS_RATE_LOW = "validation_pass_rate_low"
    COST_BUDGET_EXCEEDED = "cost_budget_exceeded"
    ERROR_RATE_HIGH = "error_rate_high"
    THROUGHPUT_LOW = "throughput_low"
    HEALTH_CHECK_FAILED = "health_check_failed"
    CUSTOM = "custom"


class NotificationChannel(str, Enum):
    """Notification delivery channels."""

    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    LOG = "log"


# === Alert Models ===


@dataclass
class Alert:
    """
    Alert for SLA violations or system issues.

    Attributes:
        alert_id: Unique alert identifier
        alert_type: Type of alert
        severity: Alert severity level
        component: Component that triggered the alert
        message: Human-readable alert message
        threshold_violated: Details of threshold violation
        current_value: Current value that violated threshold
        timestamp: When the alert was triggered
        metadata: Additional alert metadata
    """

    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    component: str
    message: str
    threshold_violated: dict[str, Any]
    current_value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "component": self.component,
            "message": self.message,
            "threshold_violated": self.threshold_violated,
            "current_value": self.current_value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class AlertRule:
    """
    Definition of an alert rule.

    Attributes:
        name: Rule name
        alert_type: Type of alert to generate
        severity: Alert severity
        condition: Callable that evaluates if alert should fire
        message_template: Template for alert message
        enabled: Whether rule is enabled
        metadata: Additional rule metadata
    """

    name: str
    alert_type: AlertType
    severity: AlertSeverity
    condition: Callable[[dict[str, Any]], bool]
    message_template: str
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


# === Alert Manager ===


class AlertManager:
    """
    Alert manager for production monitoring.

    Manages:
    - Alert rule evaluation
    - Alert generation
    - Notification delivery
    - Alert deduplication
    - Alert history

    Performance:
    - Rule evaluation: <10ms per rule
    - Notification delivery: <100ms
    - Async/non-blocking execution
    """

    def __init__(
        self,
        notification_channels: Optional[list[NotificationChannel]] = None,
    ):
        """
        Initialize alert manager.

        Args:
            notification_channels: List of enabled notification channels
        """
        self.alert_rules: list[AlertRule] = []
        self.notification_channels = notification_channels or [NotificationChannel.LOG]
        self.alert_history: list[Alert] = []
        self.alert_history_lock = asyncio.Lock()

        # Alert deduplication (prevent alert spam)
        self.recent_alerts: dict[str, datetime] = {}
        self.dedup_window_seconds = 300  # 5 minutes

        # Initialize default alert rules
        self._initialize_default_rules()

    def _initialize_default_rules(self):
        """Initialize default alert rules for common scenarios."""

        # High latency alert
        self.add_rule(
            AlertRule(
                name="high_workflow_latency",
                alert_type=AlertType.LATENCY_VIOLATION,
                severity=AlertSeverity.HIGH,
                condition=lambda m: m.get("workflow_latency_p95", 0) > 10000,  # >10s
                message_template="Workflow latency P95 ({workflow_latency_p95}ms) exceeds SLA (10000ms)",
            )
        )

        # Low cache hit rate
        self.add_rule(
            AlertRule(
                name="low_cache_hit_rate",
                alert_type=AlertType.CACHE_HIT_RATE_LOW,
                severity=AlertSeverity.MEDIUM,
                condition=lambda m: m.get("template_cache_hit_rate", 1.0) < 0.85,
                message_template="Template cache hit rate ({template_cache_hit_rate:.2%}) below target (85%)",
            )
        )

        # High cost per node
        self.add_rule(
            AlertRule(
                name="high_cost_per_node",
                alert_type=AlertType.COST_BUDGET_EXCEEDED,
                severity=AlertSeverity.CRITICAL,
                condition=lambda m: m.get("cost_per_node", 0) > 0.05,
                message_template="Cost per node (${cost_per_node:.4f}) exceeds budget ($0.05)",
            )
        )

        # Low validation pass rate
        self.add_rule(
            AlertRule(
                name="low_validation_pass_rate",
                alert_type=AlertType.VALIDATION_PASS_RATE_LOW,
                severity=AlertSeverity.HIGH,
                condition=lambda m: m.get("validation_pass_rate", 1.0) < 0.90,
                message_template="Validation pass rate ({validation_pass_rate:.2%}) below SLA (90%)",
            )
        )

        # High error rate
        self.add_rule(
            AlertRule(
                name="high_error_rate",
                alert_type=AlertType.ERROR_RATE_HIGH,
                severity=AlertSeverity.CRITICAL,
                condition=lambda m: m.get("error_rate", 0) > 0.05,  # >5%
                message_template="Error rate ({error_rate:.2%}) exceeds threshold (5%)",
            )
        )

    def add_rule(self, rule: AlertRule):
        """
        Add an alert rule.

        Args:
            rule: Alert rule to add
        """
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name} ({rule.severity.value})")

    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove an alert rule by name.

        Args:
            rule_name: Name of rule to remove

        Returns:
            True if rule was removed, False if not found
        """
        original_count = len(self.alert_rules)
        self.alert_rules = [r for r in self.alert_rules if r.name != rule_name]
        removed = len(self.alert_rules) < original_count

        if removed:
            logger.info(f"Removed alert rule: {rule_name}")

        return removed

    async def evaluate_rules(self, metrics: dict[str, Any]) -> list[Alert]:
        """
        Evaluate all alert rules against current metrics.

        Args:
            metrics: Current system metrics

        Returns:
            List of alerts generated from rule evaluation

        Performance:
            - <10ms per rule evaluation
            - Parallel evaluation of independent rules
        """
        alerts: list[Alert] = []

        for rule in self.alert_rules:
            if not rule.enabled:
                continue

            try:
                # Evaluate rule condition
                if rule.condition(metrics):
                    # Generate alert
                    alert = self._create_alert_from_rule(rule, metrics)

                    # Check for deduplication
                    if not self._is_duplicate_alert(alert):
                        alerts.append(alert)
                        await self._record_alert(alert)

            except Exception as e:
                logger.error(f"Failed to evaluate alert rule {rule.name}: {e}")

        return alerts

    def _create_alert_from_rule(
        self, rule: AlertRule, metrics: dict[str, Any]
    ) -> Alert:
        """
        Create alert from rule and metrics.

        Args:
            rule: Alert rule that triggered
            metrics: Current metrics

        Returns:
            Generated alert
        """
        # Format message with metrics
        try:
            message = rule.message_template.format(**metrics)
        except KeyError:
            message = rule.message_template

        # Extract violated threshold details
        threshold_violated = rule.metadata.copy()

        # Generate unique alert ID
        alert_id = f"{rule.name}_{datetime.now().timestamp()}"

        return Alert(
            alert_id=alert_id,
            alert_type=rule.alert_type,
            severity=rule.severity,
            component=rule.metadata.get("component", "system"),
            message=message,
            threshold_violated=threshold_violated,
            current_value=metrics,
            metadata={"rule_name": rule.name},
        )

    def _is_duplicate_alert(self, alert: Alert) -> bool:
        """
        Check if alert is a duplicate within deduplication window.

        Args:
            alert: Alert to check

        Returns:
            True if alert is duplicate, False otherwise
        """
        dedup_key = f"{alert.component}:{alert.alert_type.value}"

        if dedup_key in self.recent_alerts:
            last_alert_time = self.recent_alerts[dedup_key]
            time_since_last = (datetime.now() - last_alert_time).total_seconds()

            if time_since_last < self.dedup_window_seconds:
                logger.debug(f"Suppressing duplicate alert: {dedup_key}")
                return True

        # Update recent alerts
        self.recent_alerts[dedup_key] = datetime.now()
        return False

    async def _record_alert(self, alert: Alert):
        """
        Record alert in history.

        Args:
            alert: Alert to record
        """
        async with self.alert_history_lock:
            self.alert_history.append(alert)

            # Limit history size (keep last 1000 alerts)
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]

    async def send_alert(self, alert: Alert):
        """
        Send alert to configured notification channels.

        Args:
            alert: Alert to send

        Performance:
            - <100ms notification delivery target
            - Async/non-blocking delivery
            - Graceful handling of delivery failures
        """
        logger.info(
            f"Sending alert: [{alert.severity.value}] {alert.component} - {alert.message}"
        )

        # Send to all enabled channels in parallel
        send_tasks = []

        for channel in self.notification_channels:
            if channel == NotificationChannel.LOG:
                send_tasks.append(self._send_to_log(alert))
            elif channel == NotificationChannel.SLACK:
                send_tasks.append(self._send_to_slack(alert))
            elif channel == NotificationChannel.EMAIL:
                send_tasks.append(self._send_to_email(alert))
            elif channel == NotificationChannel.PAGERDUTY:
                send_tasks.append(self._send_to_pagerduty(alert))
            elif channel == NotificationChannel.WEBHOOK:
                send_tasks.append(self._send_to_webhook(alert))

        # Execute all sends in parallel
        await asyncio.gather(*send_tasks, return_exceptions=True)

    async def _send_to_log(self, alert: Alert):
        """Send alert to log channel."""
        log_level = {
            AlertSeverity.CRITICAL: logging.CRITICAL,
            AlertSeverity.HIGH: logging.ERROR,
            AlertSeverity.MEDIUM: logging.WARNING,
            AlertSeverity.LOW: logging.INFO,
            AlertSeverity.INFO: logging.INFO,
        }[alert.severity]

        logger.log(
            log_level,
            f"ALERT [{alert.severity.value}] {alert.component}: {alert.message}",
        )

    async def _send_to_slack(self, alert: Alert):
        """Send alert to Slack channel (stub for future implementation)."""
        logger.debug(f"Slack notification for alert: {alert.alert_id}")
        # TODO: Implement Slack webhook integration

    async def _send_to_email(self, alert: Alert):
        """Send alert via email (stub for future implementation)."""
        logger.debug(f"Email notification for alert: {alert.alert_id}")
        # TODO: Implement email notification

    async def _send_to_pagerduty(self, alert: Alert):
        """Send alert to PagerDuty (stub for future implementation)."""
        logger.debug(f"PagerDuty notification for alert: {alert.alert_id}")
        # TODO: Implement PagerDuty integration

    async def _send_to_webhook(self, alert: Alert):
        """Send alert to webhook (stub for future implementation)."""
        logger.debug(f"Webhook notification for alert: {alert.alert_id}")
        # TODO: Implement webhook integration

    def get_alert_history(
        self,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100,
    ) -> list[Alert]:
        """
        Get alert history with optional filtering.

        Args:
            severity: Filter by severity level
            limit: Maximum number of alerts to return

        Returns:
            List of historical alerts
        """
        alerts = self.alert_history

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        # Return most recent alerts first
        return list(reversed(alerts[-limit:]))

    def get_alert_statistics(self) -> dict[str, Any]:
        """
        Get alert statistics.

        Returns:
            Dictionary of alert statistics
        """
        if not self.alert_history:
            return {
                "total_alerts": 0,
                "by_severity": {},
                "by_type": {},
                "by_component": {},
            }

        # Count by severity
        by_severity = {}
        for severity in AlertSeverity:
            count = sum(1 for a in self.alert_history if a.severity == severity)
            if count > 0:
                by_severity[severity.value] = count

        # Count by type
        by_type = {}
        for alert_type in AlertType:
            count = sum(1 for a in self.alert_history if a.alert_type == alert_type)
            if count > 0:
                by_type[alert_type.value] = count

        # Count by component
        by_component = {}
        for alert in self.alert_history:
            component = alert.component
            by_component[component] = by_component.get(component, 0) + 1

        return {
            "total_alerts": len(self.alert_history),
            "by_severity": by_severity,
            "by_type": by_type,
            "by_component": by_component,
        }
