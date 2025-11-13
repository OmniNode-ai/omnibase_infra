"""Alert management for performance and resource monitoring.

Handles alert generation, acknowledgment, escalation, and notification
for performance threshold violations and resource constraints.
"""

import asyncio
import logging
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status states."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


@dataclass
class Alert:
    """Individual alert instance."""

    id: str
    alert_type: str
    severity: AlertSeverity
    title: str
    message: str
    source: str
    timestamp: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    escalation_count: int = 0
    last_seen: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary for API responses.

        Returns:
            Alert as dictionary
        """
        return {
            "id": self.id,
            "type": self.alert_type,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": (
                self.acknowledged_at.isoformat() if self.acknowledged_at else None
            ),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata,
            "escalation_count": self.escalation_count,
            "last_seen": self.last_seen.isoformat(),
            "age_seconds": (datetime.now() - self.timestamp).total_seconds(),
        }


@dataclass
class AlertRule:
    """Alert rule configuration."""

    rule_id: str
    alert_type: str
    severity: AlertSeverity
    threshold_value: float
    comparison: str  # "gt", "lt", "eq", "gte", "lte"
    window_minutes: int = 5
    min_occurrences: int = 1
    escalation_minutes: int = 30
    auto_resolve_minutes: int = 60
    enabled: bool = True


class AlertManager:
    """Comprehensive alert management system."""

    def __init__(
        self,
        max_alerts: int = 1000,
        auto_cleanup_hours: int = 24,
        escalation_enabled: bool = True,
    ):
        """Initialize alert manager.

        Args:
            max_alerts: Maximum alerts to keep in memory
            auto_cleanup_hours: Hours before auto-cleaning resolved alerts
            escalation_enabled: Enable alert escalation
        """
        self.max_alerts = max_alerts
        self.auto_cleanup_period = timedelta(hours=auto_cleanup_hours)
        self.escalation_enabled = escalation_enabled

        # Alert storage
        self.alerts: dict[str, Alert] = {}
        self.alert_history: deque[Alert] = deque(maxlen=max_alerts)

        # Alert rules
        self.alert_rules: dict[str, AlertRule] = {}

        # Alert counters for rate limiting
        self.alert_counters: dict[str, deque[datetime]] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # Setup default alert rules
        self._setup_default_rules()

        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.escalation_task: Optional[asyncio.Task] = None
        self.is_running = False

    def _setup_default_rules(self):
        """Setup default alert rules based on omnibase_3 requirements."""
        default_rules = [
            # Database performance rules
            AlertRule(
                rule_id="db_query_latency_warning",
                alert_type="database_latency",
                severity=AlertSeverity.WARNING,
                threshold_value=100.0,  # 100ms
                comparison="gt",
                window_minutes=5,
                min_occurrences=3,
            ),
            AlertRule(
                rule_id="db_query_latency_critical",
                alert_type="database_latency",
                severity=AlertSeverity.CRITICAL,
                threshold_value=200.0,  # 200ms
                comparison="gt",
                window_minutes=5,
                min_occurrences=2,
            ),
            # Hash generation performance rules
            AlertRule(
                rule_id="hash_generation_warning",
                alert_type="hash_performance",
                severity=AlertSeverity.WARNING,
                threshold_value=2.0,  # 2ms
                comparison="gt",
                window_minutes=5,
                min_occurrences=5,
            ),
            AlertRule(
                rule_id="hash_generation_critical",
                alert_type="hash_performance",
                severity=AlertSeverity.CRITICAL,
                threshold_value=5.0,  # 5ms
                comparison="gt",
                window_minutes=5,
                min_occurrences=3,
            ),
            # Batch throughput rules
            AlertRule(
                rule_id="batch_throughput_warning",
                alert_type="batch_throughput",
                severity=AlertSeverity.WARNING,
                threshold_value=50.0,  # 50 items/second
                comparison="lt",
                window_minutes=10,
                min_occurrences=3,
            ),
            # Event publishing rules
            AlertRule(
                rule_id="event_publishing_warning",
                alert_type="event_latency",
                severity=AlertSeverity.WARNING,
                threshold_value=50.0,  # 50ms
                comparison="gt",
                window_minutes=5,
                min_occurrences=3,
            ),
            # Resource utilization rules
            AlertRule(
                rule_id="cpu_usage_warning",
                alert_type="resource_cpu",
                severity=AlertSeverity.WARNING,
                threshold_value=70.0,  # 70%
                comparison="gt",
                window_minutes=10,
                min_occurrences=2,
            ),
            AlertRule(
                rule_id="cpu_usage_critical",
                alert_type="resource_cpu",
                severity=AlertSeverity.CRITICAL,
                threshold_value=90.0,  # 90%
                comparison="gt",
                window_minutes=5,
                min_occurrences=1,
            ),
            AlertRule(
                rule_id="memory_usage_warning",
                alert_type="resource_memory",
                severity=AlertSeverity.WARNING,
                threshold_value=70.0,  # 70%
                comparison="gt",
                window_minutes=10,
                min_occurrences=2,
            ),
            AlertRule(
                rule_id="memory_usage_critical",
                alert_type="resource_memory",
                severity=AlertSeverity.CRITICAL,
                threshold_value=90.0,  # 90%
                comparison="gt",
                window_minutes=5,
                min_occurrences=1,
            ),
            # Success rate rules
            AlertRule(
                rule_id="success_rate_warning",
                alert_type="success_rate",
                severity=AlertSeverity.WARNING,
                threshold_value=0.95,  # 95%
                comparison="lt",
                window_minutes=15,
                min_occurrences=2,
            ),
            AlertRule(
                rule_id="success_rate_critical",
                alert_type="success_rate",
                severity=AlertSeverity.CRITICAL,
                threshold_value=0.90,  # 90%
                comparison="lt",
                window_minutes=10,
                min_occurrences=1,
            ),
        ]

        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule

    async def start(self):
        """Start alert manager background tasks."""
        if self.is_running:
            return

        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        if self.escalation_enabled:
            self.escalation_task = asyncio.create_task(self._escalation_loop())

        logger.info("Alert manager started")

    async def stop(self):
        """Stop alert manager background tasks."""
        if not self.is_running:
            return

        self.is_running = False

        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        if self.escalation_task:
            self.escalation_task.cancel()
            try:
                await self.escalation_task
            except asyncio.CancelledError:
                pass

        logger.info("Alert manager stopped")

    async def _cleanup_loop(self):
        """Background task for cleaning up old resolved alerts."""
        try:
            while self.is_running:
                await self._cleanup_old_alerts()
                await asyncio.sleep(3600)  # Run every hour
        except asyncio.CancelledError:
            logger.info("Alert cleanup loop cancelled")
        except Exception as e:
            logger.error(f"Error in alert cleanup loop: {e}")

    async def _escalation_loop(self):
        """Background task for escalating unacknowledged alerts."""
        try:
            while self.is_running:
                await self._check_escalations()
                await asyncio.sleep(300)  # Check every 5 minutes
        except asyncio.CancelledError:
            logger.info("Alert escalation loop cancelled")
        except Exception as e:
            logger.error(f"Error in alert escalation loop: {e}")

    async def create_alert(
        self,
        alert_type: str,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str = "monitoring",
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Create a new alert.

        Args:
            alert_type: Type of alert
            severity: Alert severity
            title: Alert title
            message: Alert message
            source: Alert source system
            metadata: Additional metadata

        Returns:
            Alert ID
        """
        # Check for rate limiting
        if self._is_rate_limited(alert_type):
            logger.debug(f"Alert {alert_type} rate limited, skipping")
            return ""

        alert_id = str(uuid.uuid4())

        alert = Alert(
            id=alert_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            source=source,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )

        # Check if similar alert already exists
        existing_alert = self._find_similar_alert(alert)
        if existing_alert:
            # Update existing alert instead of creating new one
            existing_alert.last_seen = datetime.now()
            existing_alert.escalation_count += 1
            logger.debug(f"Updated existing alert {existing_alert.id}")
            return existing_alert.id

        # Store new alert
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Update rate limiting counter
        self.alert_counters[alert_type].append(datetime.now())

        logger.info(f"Created {severity.value} alert: {title}")

        return alert_id

    def _is_rate_limited(self, alert_type: str) -> bool:
        """Check if alert type is rate limited.

        Args:
            alert_type: Type of alert

        Returns:
            True if rate limited
        """
        now = datetime.now()
        window = timedelta(minutes=5)

        # Clean old entries
        while (
            self.alert_counters[alert_type]
            and now - self.alert_counters[alert_type][0] > window
        ):
            self.alert_counters[alert_type].popleft()

        # Check rate limit (max 10 alerts per 5 minutes per type)
        return len(self.alert_counters[alert_type]) >= 10

    def _find_similar_alert(self, new_alert: Alert) -> Optional[Alert]:
        """Find existing similar alert.

        Args:
            new_alert: New alert to check

        Returns:
            Existing similar alert or None
        """
        for alert in self.alerts.values():
            if (
                alert.alert_type == new_alert.alert_type
                and alert.severity == new_alert.severity
                and alert.status == AlertStatus.ACTIVE
                and alert.source == new_alert.source
            ):
                # Check if within time window (5 minutes)
                if (new_alert.timestamp - alert.timestamp).total_seconds() < 300:
                    return alert

        return None

    async def acknowledge_alert(
        self, alert_id: str, acknowledged_by: str = "system"
    ) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: Alert ID
            acknowledged_by: Who acknowledged the alert

        Returns:
            True if successful
        """
        if alert_id not in self.alerts:
            return False

        alert = self.alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.now()

        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return True

    async def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an alert.

        Args:
            alert_id: Alert ID
            resolved_by: Who resolved the alert

        Returns:
            True if successful
        """
        if alert_id not in self.alerts:
            return False

        alert = self.alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()

        logger.info(f"Alert {alert_id} resolved by {resolved_by}")
        return True

    async def get_active_alerts(
        self,
        severity_filter: Optional[AlertSeverity] = None,
        alert_type_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get all active alerts.

        Args:
            severity_filter: Filter by severity
            alert_type_filter: Filter by alert type

        Returns:
            List of active alerts
        """
        active_alerts = []

        for alert in self.alerts.values():
            if alert.status != AlertStatus.ACTIVE:
                continue

            if severity_filter and alert.severity != severity_filter:
                continue

            if alert_type_filter and alert.alert_type != alert_type_filter:
                continue

            active_alerts.append(alert.to_dict())

        # Sort by severity and timestamp
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.INFO: 2,
        }

        active_alerts.sort(
            key=lambda a: (
                severity_order.get(AlertSeverity(a["severity"]), 3),
                a["timestamp"],
            )
        )

        return active_alerts

    async def get_alert_statistics(self) -> dict[str, Any]:
        """Get alert statistics.

        Returns:
            Alert statistics
        """
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        last_1h = now - timedelta(hours=1)

        stats = {
            "total_alerts": len(self.alerts),
            "active_alerts": 0,
            "acknowledged_alerts": 0,
            "resolved_alerts": 0,
            "critical_alerts": 0,
            "warning_alerts": 0,
            "info_alerts": 0,
            "alerts_last_24h": 0,
            "alerts_last_1h": 0,
            "by_type": defaultdict(int),
            "by_source": defaultdict(int),
            "escalated_alerts": 0,
        }

        for alert in self.alerts.values():
            # Status counts
            if alert.status == AlertStatus.ACTIVE:
                stats["active_alerts"] += 1
            elif alert.status == AlertStatus.ACKNOWLEDGED:
                stats["acknowledged_alerts"] += 1
            elif alert.status == AlertStatus.RESOLVED:
                stats["resolved_alerts"] += 1

            # Severity counts
            if alert.severity == AlertSeverity.CRITICAL:
                stats["critical_alerts"] += 1
            elif alert.severity == AlertSeverity.WARNING:
                stats["warning_alerts"] += 1
            elif alert.severity == AlertSeverity.INFO:
                stats["info_alerts"] += 1

            # Time-based counts
            if alert.timestamp >= last_24h:
                stats["alerts_last_24h"] += 1
            if alert.timestamp >= last_1h:
                stats["alerts_last_1h"] += 1

            # Type and source counts
            stats["by_type"][alert.alert_type] += 1
            stats["by_source"][alert.source] += 1

            # Escalation counts
            if alert.escalation_count > 0:
                stats["escalated_alerts"] += 1

        # Convert defaultdicts to regular dicts
        stats["by_type"] = dict(stats["by_type"])
        stats["by_source"] = dict(stats["by_source"])

        return stats

    async def _cleanup_old_alerts(self):
        """Clean up old resolved alerts."""
        cutoff_time = datetime.now() - self.auto_cleanup_period
        alerts_to_remove = []

        for alert_id, alert in self.alerts.items():
            if (
                alert.status == AlertStatus.RESOLVED
                and alert.resolved_at
                and alert.resolved_at < cutoff_time
            ):
                alerts_to_remove.append(alert_id)

        for alert_id in alerts_to_remove:
            del self.alerts[alert_id]

        if alerts_to_remove:
            logger.info(f"Cleaned up {len(alerts_to_remove)} old resolved alerts")

    async def _check_escalations(self):
        """Check for alerts that need escalation."""
        now = datetime.now()

        for alert in self.alerts.values():
            if (
                alert.status == AlertStatus.ACTIVE
                and alert.severity == AlertSeverity.CRITICAL
            ):

                # Check if alert should be escalated
                alert_age = (now - alert.timestamp).total_seconds() / 60  # minutes

                # Get escalation threshold from rule
                rule = self.alert_rules.get(f"{alert.alert_type}_critical")
                escalation_threshold = rule.escalation_minutes if rule else 30

                if alert_age >= escalation_threshold and alert.escalation_count == 0:
                    await self._escalate_alert(alert)

    async def _escalate_alert(self, alert: Alert):
        """Escalate an alert.

        Args:
            alert: Alert to escalate
        """
        alert.status = AlertStatus.ESCALATED
        alert.escalation_count += 1

        logger.warning(
            f"Escalating alert {alert.id}: {alert.title} "
            f"(age: {(datetime.now() - alert.timestamp).total_seconds() / 60:.1f} minutes)"
        )

        # Here you could integrate with external notification systems
        # like PagerDuty, Slack, email, etc.

    def add_alert_rule(self, rule: AlertRule):
        """Add a custom alert rule.

        Args:
            rule: Alert rule to add
        """
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.rule_id}")

    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule.

        Args:
            rule_id: Rule ID to remove

        Returns:
            True if rule was removed
        """
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
            return True
        return False

    def get_alert_rules(self) -> list[dict[str, Any]]:
        """Get all alert rules.

        Returns:
            List of alert rules
        """
        return [
            {
                "rule_id": rule.rule_id,
                "alert_type": rule.alert_type,
                "severity": rule.severity.value,
                "threshold_value": rule.threshold_value,
                "comparison": rule.comparison,
                "window_minutes": rule.window_minutes,
                "min_occurrences": rule.min_occurrences,
                "escalation_minutes": rule.escalation_minutes,
                "auto_resolve_minutes": rule.auto_resolve_minutes,
                "enabled": rule.enabled,
            }
            for rule in self.alert_rules.values()
        ]
