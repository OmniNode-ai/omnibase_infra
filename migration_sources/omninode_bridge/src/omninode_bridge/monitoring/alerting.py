"""Production monitoring and alerting configuration for OmniNode Bridge."""

import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional

import structlog
from prometheus_client import Counter, Gauge, Histogram, generate_latest

from ..middleware.request_correlation import get_correlation_context

logger = structlog.get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


class AlertChannel(Enum):
    """Alert delivery channels."""

    LOG = "log"
    WEBHOOK = "webhook"
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"


@dataclass
class AlertRule:
    """Configuration for an alert rule."""

    name: str
    condition: Callable[[], bool]
    severity: AlertSeverity
    message_template: str
    channels: list[AlertChannel]
    cooldown_minutes: int = 15
    enabled: bool = True


@dataclass
class AlertInstance:
    """An active alert instance."""

    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    correlation_context: dict[str, str]
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


class ProductionMonitoringConfig:
    """Production monitoring and alerting configuration."""

    def __init__(
        self,
        service_name: str,
        environment: str = "production",
        enable_custom_metrics: bool = True,
    ):
        """Initialize production monitoring configuration.

        Args:
            service_name: Name of the service
            environment: Environment name (production, staging, etc.)
            enable_custom_metrics: Enable custom Prometheus metrics
        """
        self.service_name = service_name
        self.environment = environment
        self.enable_custom_metrics = enable_custom_metrics

        # Configuration from environment variables
        self.metrics_port = int(os.getenv("METRICS_PORT", "9090"))
        self.alert_webhook_url = os.getenv("ALERT_WEBHOOK_URL")
        self.enable_alerting = os.getenv("ENABLE_ALERTING", "true").lower() == "true"

        # Alert state management
        self.active_alerts: dict[str, AlertInstance] = {}
        self.alert_history: list[AlertInstance] = []
        self.last_alert_check = datetime.now(UTC)

        # Alert rules
        self.alert_rules: dict[str, AlertRule] = {}

        # Prometheus registry and metrics
        if enable_custom_metrics:
            self.setup_custom_metrics()

        # Default alert rules
        self.setup_default_alert_rules()

    def setup_custom_metrics(self) -> None:
        """Setup custom Prometheus metrics for production monitoring."""
        # Service-level metrics
        self.service_availability_gauge = Gauge(
            "omninode_service_availability",
            "Service availability (1=available, 0=unavailable)",
            ["service", "environment"],
        )

        self.service_response_time = Histogram(
            "omninode_service_response_time_seconds",
            "Service response time in seconds",
            ["service", "environment", "endpoint", "method"],
        )

        self.service_error_rate = Counter(
            "omninode_service_errors_total",
            "Total service errors",
            ["service", "environment", "error_type", "endpoint"],
        )

        # Infrastructure metrics
        self.database_connections_gauge = Gauge(
            "omninode_database_connections_active",
            "Active database connections",
            ["service", "database_type"],
        )

        self.message_queue_lag = Gauge(
            "omninode_message_queue_lag_messages",
            "Message queue consumer lag in messages",
            ["service", "topic", "consumer_group"],
        )

        # Business metrics
        self.workflow_executions_total = Counter(
            "omninode_workflow_executions_total",
            "Total workflow executions",
            ["service", "workflow_type", "status"],
        )

        self.workflow_duration = Histogram(
            "omninode_workflow_duration_seconds",
            "Workflow execution duration in seconds",
            ["service", "workflow_type"],
        )

        # Alert metrics
        self.alerts_fired_total = Counter(
            "omninode_alerts_fired_total",
            "Total alerts fired",
            ["service", "rule_name", "severity"],
        )

        self.alerts_active_gauge = Gauge(
            "omninode_alerts_active", "Number of active alerts", ["service", "severity"]
        )

        logger.info("Custom Prometheus metrics initialized", service=self.service_name)

    def setup_default_alert_rules(self) -> None:
        """Setup default alert rules for production monitoring."""

        # High error rate alert
        self.add_alert_rule(
            AlertRule(
                name="high_error_rate",
                condition=lambda: self._check_error_rate_threshold(
                    5.0
                ),  # 5% error rate
                severity=AlertSeverity.CRITICAL,
                message_template="High error rate detected: {error_rate:.2f}% (threshold: 5%)",
                channels=[AlertChannel.LOG, AlertChannel.WEBHOOK],
                cooldown_minutes=10,
            )
        )

        # Slow response time alert
        self.add_alert_rule(
            AlertRule(
                name="slow_response_time",
                condition=lambda: self._check_response_time_threshold(5.0),  # 5 seconds
                severity=AlertSeverity.WARNING,
                message_template="Slow response time detected: {avg_response_time:.2f}s (threshold: 5s)",
                channels=[AlertChannel.LOG, AlertChannel.WEBHOOK],
                cooldown_minutes=15,
            )
        )

        # Database connection alert
        self.add_alert_rule(
            AlertRule(
                name="database_connection_failure",
                condition=lambda: self._check_database_health(),
                severity=AlertSeverity.CRITICAL,
                message_template="Database connection failure detected",
                channels=[AlertChannel.LOG, AlertChannel.WEBHOOK],
                cooldown_minutes=5,
            )
        )

        # High memory usage alert
        self.add_alert_rule(
            AlertRule(
                name="high_memory_usage",
                condition=lambda: self._check_memory_usage_threshold(85.0),  # 85%
                severity=AlertSeverity.WARNING,
                message_template="High memory usage detected: {memory_usage:.1f}% (threshold: 85%)",
                channels=[AlertChannel.LOG],
                cooldown_minutes=20,
            )
        )

        # Message queue lag alert
        self.add_alert_rule(
            AlertRule(
                name="message_queue_lag",
                condition=lambda: self._check_message_queue_lag_threshold(
                    1000
                ),  # 1000 messages
                severity=AlertSeverity.WARNING,
                message_template="Message queue lag detected: {lag} messages (threshold: 1000)",
                channels=[AlertChannel.LOG, AlertChannel.WEBHOOK],
                cooldown_minutes=10,
            )
        )

        logger.info("Default alert rules configured", service=self.service_name)

    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule to the monitoring system.

        Args:
            rule: Alert rule configuration
        """
        self.alert_rules[rule.name] = rule
        logger.debug("Alert rule added", rule_name=rule.name, service=self.service_name)

    def remove_alert_rule(self, rule_name: str) -> None:
        """Remove an alert rule.

        Args:
            rule_name: Name of the rule to remove
        """
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.debug(
                "Alert rule removed", rule_name=rule_name, service=self.service_name
            )

    async def check_alert_conditions(self) -> list[AlertInstance]:
        """Check all alert conditions and fire alerts as needed.

        Returns:
            List of newly fired alerts
        """
        if not self.enable_alerting:
            return []

        new_alerts = []
        current_time = datetime.now(UTC)

        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue

            try:
                # Check if rule is in cooldown
                if rule_name in self.active_alerts:
                    last_alert = self.active_alerts[rule_name]
                    cooldown_end = last_alert.timestamp.replace(
                        tzinfo=UTC
                    ) + datetime.timedelta(minutes=rule.cooldown_minutes)

                    if current_time < cooldown_end:
                        continue  # Still in cooldown

                # Check alert condition
                if rule.condition():
                    # Fire alert
                    alert = AlertInstance(
                        rule_name=rule_name,
                        severity=rule.severity,
                        message=rule.message_template,
                        timestamp=current_time,
                        correlation_context=get_correlation_context(),
                    )

                    # Add to active alerts
                    self.active_alerts[rule_name] = alert
                    self.alert_history.append(alert)
                    new_alerts.append(alert)

                    # Update metrics
                    if self.enable_custom_metrics:
                        self.alerts_fired_total.labels(
                            service=self.service_name,
                            rule_name=rule_name,
                            severity=rule.severity.value,
                        ).inc()

                    # Send alert through configured channels
                    await self._send_alert(alert, rule.channels)

                    logger.warning(
                        "Alert fired",
                        rule_name=rule_name,
                        severity=rule.severity.value,
                        message=alert.message,
                        service=self.service_name,
                        **alert.correlation_context,
                    )

                else:
                    # Check if we should resolve an active alert
                    if rule_name in self.active_alerts:
                        active_alert = self.active_alerts[rule_name]
                        if not active_alert.resolved:
                            active_alert.resolved = True
                            active_alert.resolution_timestamp = current_time

                            logger.info(
                                "Alert resolved",
                                rule_name=rule_name,
                                service=self.service_name,
                                **active_alert.correlation_context,
                            )

            except Exception as e:
                logger.error(
                    "Error checking alert condition",
                    rule_name=rule_name,
                    error=str(e),
                    service=self.service_name,
                )

        # Update active alerts gauge
        if self.enable_custom_metrics:
            for severity in AlertSeverity:
                active_count = len(
                    [
                        alert
                        for alert in self.active_alerts.values()
                        if alert.severity == severity and not alert.resolved
                    ]
                )
                self.alerts_active_gauge.labels(
                    service=self.service_name, severity=severity.value
                ).set(active_count)

        return new_alerts

    async def _send_alert(
        self, alert: AlertInstance, channels: list[AlertChannel]
    ) -> None:
        """Send alert through configured channels.

        Args:
            alert: Alert instance to send
            channels: List of channels to send alert through
        """
        for channel in channels:
            try:
                if channel == AlertChannel.LOG:
                    # Already logged in check_alert_conditions
                    pass
                elif channel == AlertChannel.WEBHOOK and self.alert_webhook_url:
                    await self._send_webhook_alert(alert)
                elif channel == AlertChannel.EMAIL:
                    await self._send_email_alert(alert)
                elif channel == AlertChannel.SLACK:
                    await self._send_slack_alert(alert)
                elif channel == AlertChannel.PAGERDUTY:
                    await self._send_pagerduty_alert(alert)

            except Exception as e:
                logger.error(
                    "Failed to send alert",
                    rule_name=alert.rule_name,
                    channel=channel.value,
                    error=str(e),
                    service=self.service_name,
                )

    async def _send_webhook_alert(self, alert: AlertInstance) -> None:
        """Send alert via webhook.

        Args:
            alert: Alert instance to send
        """
        import aiohttp

        payload = {
            "service": self.service_name,
            "environment": self.environment,
            "rule_name": alert.rule_name,
            "severity": alert.severity.value,
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat(),
            "correlation_context": alert.correlation_context,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.alert_webhook_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status != 200:
                    logger.warning(
                        "Webhook alert failed",
                        status_code=response.status,
                        rule_name=alert.rule_name,
                    )

    async def _send_email_alert(self, alert: AlertInstance) -> None:
        """Send alert via email (placeholder for email integration)."""
        # Placeholder for email integration
        logger.debug("Email alert placeholder", rule_name=alert.rule_name)

    async def _send_slack_alert(self, alert: AlertInstance) -> None:
        """Send alert via Slack (placeholder for Slack integration)."""
        # Placeholder for Slack integration
        logger.debug("Slack alert placeholder", rule_name=alert.rule_name)

    async def _send_pagerduty_alert(self, alert: AlertInstance) -> None:
        """Send alert via PagerDuty (placeholder for PagerDuty integration)."""
        # Placeholder for PagerDuty integration
        logger.debug("PagerDuty alert placeholder", rule_name=alert.rule_name)

    # Alert condition check methods (to be implemented based on actual metrics)
    def _check_error_rate_threshold(self, threshold: float) -> bool:
        """Check if error rate exceeds threshold."""
        # Placeholder - implement based on actual metrics collection
        return False

    def _check_response_time_threshold(self, threshold: float) -> bool:
        """Check if average response time exceeds threshold."""
        # Placeholder - implement based on actual metrics collection
        return False

    def _check_database_health(self) -> bool:
        """Check database health."""
        # Placeholder - implement based on actual database health checks
        return False

    def _check_memory_usage_threshold(self, threshold: float) -> bool:
        """Check if memory usage exceeds threshold."""
        # Placeholder - implement based on system monitoring
        return False

    def _check_message_queue_lag_threshold(self, threshold: int) -> bool:
        """Check if message queue lag exceeds threshold."""
        # Placeholder - implement based on Kafka metrics
        return False

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format.

        Returns:
            Prometheus metrics string
        """
        if self.enable_custom_metrics:
            return generate_latest()
        return ""

    def get_alert_status(self) -> dict[str, Any]:
        """Get current alert status.

        Returns:
            Dictionary with alert status information
        """
        active_alerts = [
            {
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved,
            }
            for alert in self.active_alerts.values()
            if not alert.resolved
        ]

        return {
            "service": self.service_name,
            "environment": self.environment,
            "alerting_enabled": self.enable_alerting,
            "active_alerts": active_alerts,
            "total_rules": len(self.alert_rules),
            "enabled_rules": len([r for r in self.alert_rules.values() if r.enabled]),
            "alert_history_count": len(self.alert_history),
        }


# Global monitoring configuration
_monitoring_config: Optional[ProductionMonitoringConfig] = None


def initialize_production_monitoring(
    service_name: str,
    environment: str = "production",
    enable_custom_metrics: bool = True,
) -> ProductionMonitoringConfig:
    """Initialize production monitoring configuration.

    Args:
        service_name: Name of the service
        environment: Environment name
        enable_custom_metrics: Enable custom Prometheus metrics

    Returns:
        Monitoring configuration instance
    """
    global _monitoring_config

    _monitoring_config = ProductionMonitoringConfig(
        service_name=service_name,
        environment=environment,
        enable_custom_metrics=enable_custom_metrics,
    )

    logger.info(
        "Production monitoring initialized",
        service=service_name,
        environment=environment,
        custom_metrics=enable_custom_metrics,
    )

    return _monitoring_config


def get_monitoring_config() -> Optional[ProductionMonitoringConfig]:
    """Get the global monitoring configuration."""
    return _monitoring_config


async def start_alert_monitoring_loop(interval_seconds: int = 60) -> None:
    """Start the alert monitoring background loop.

    Args:
        interval_seconds: How often to check alert conditions
    """
    if not _monitoring_config:
        logger.warning("Monitoring configuration not initialized")
        return

    logger.info("Starting alert monitoring loop", interval=interval_seconds)

    while True:
        try:
            await _monitoring_config.check_alert_conditions()
            await asyncio.sleep(interval_seconds)
        except Exception as e:
            logger.error("Error in alert monitoring loop", error=str(e))
            await asyncio.sleep(interval_seconds)


def record_workflow_execution(
    workflow_type: str,
    status: str,
    duration_seconds: float,
) -> None:
    """Record workflow execution metrics.

    Args:
        workflow_type: Type of workflow
        status: Execution status (success, failed, timeout, etc.)
        duration_seconds: Execution duration in seconds
    """
    if _monitoring_config and _monitoring_config.enable_custom_metrics:
        _monitoring_config.workflow_executions_total.labels(
            service=_monitoring_config.service_name,
            workflow_type=workflow_type,
            status=status,
        ).inc()

        _monitoring_config.workflow_duration.labels(
            service=_monitoring_config.service_name, workflow_type=workflow_type
        ).observe(duration_seconds)


def record_service_response_time(
    endpoint: str,
    method: str,
    response_time_seconds: float,
) -> None:
    """Record service response time metrics.

    Args:
        endpoint: API endpoint
        method: HTTP method
        response_time_seconds: Response time in seconds
    """
    if _monitoring_config and _monitoring_config.enable_custom_metrics:
        _monitoring_config.service_response_time.labels(
            service=_monitoring_config.service_name,
            environment=_monitoring_config.environment,
            endpoint=endpoint,
            method=method,
        ).observe(response_time_seconds)


def record_service_error(
    endpoint: str,
    error_type: str,
) -> None:
    """Record service error metrics.

    Args:
        endpoint: API endpoint where error occurred
        error_type: Type of error
    """
    if _monitoring_config and _monitoring_config.enable_custom_metrics:
        _monitoring_config.service_error_rate.labels(
            service=_monitoring_config.service_name,
            environment=_monitoring_config.environment,
            error_type=error_type,
            endpoint=endpoint,
        ).inc()


def update_database_connections(database_type: str, active_connections: int) -> None:
    """Update database connection metrics.

    Args:
        database_type: Type of database (postgresql, redis, etc.)
        active_connections: Number of active connections
    """
    if _monitoring_config and _monitoring_config.enable_custom_metrics:
        _monitoring_config.database_connections_gauge.labels(
            service=_monitoring_config.service_name, database_type=database_type
        ).set(active_connections)
