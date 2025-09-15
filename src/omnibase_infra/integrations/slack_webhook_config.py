"""
Production Slack Webhook Configuration for ONEX Infrastructure.

This module provides production-ready Slack webhook configurations for various
infrastructure alerts and notifications in the ONEX system.
"""

from typing import Dict, Any, List
from datetime import datetime
from enum import Enum

from omnibase_core.enums.enum_notification_method import EnumNotificationMethod
from omnibase_core.enums.enum_auth_type import EnumAuthType
from omnibase_infra.models.notification.model_notification_request import ModelNotificationRequest
from omnibase_infra.models.notification.model_notification_auth import ModelNotificationAuth
from omnibase_infra.models.notification.model_notification_retry_policy import ModelNotificationRetryPolicy
from omnibase_core.enums.enum_backoff_strategy import EnumBackoffStrategy


class SlackChannel(str, Enum):
    """Production Slack channels for different notification types."""
    ALERTS = "#infrastructure-alerts"
    GENERAL = "#dev-general"
    CRITICAL = "#critical-alerts"
    MONITORING = "#infrastructure-monitoring"
    DEPLOYMENTS = "#deployments"
    SECURITY = "#security-alerts"


class SlackPriority(str, Enum):
    """Alert priority levels with corresponding Slack formatting."""
    CRITICAL = "danger"  # Red
    HIGH = "warning"     # Yellow
    MEDIUM = "good"      # Green
    INFO = "#36a64f"     # Custom green


class ProductionSlackWebhook:
    """Production-ready Slack webhook configuration and message formatting."""

    def __init__(self, webhook_url: str, default_channel: SlackChannel = SlackChannel.ALERTS):
        """
        Initialize production Slack webhook.

        Args:
            webhook_url: Your production Slack webhook URL
            default_channel: Default channel for notifications
        """
        self.webhook_url = webhook_url
        self.default_channel = default_channel

    def create_infrastructure_alert(
        self,
        title: str,
        message: str,
        service: str,
        severity: SlackPriority = SlackPriority.MEDIUM,
        channel: SlackChannel = None,
        additional_fields: List[Dict[str, str]] = None
    ) -> ModelNotificationRequest:
        """
        Create infrastructure alert notification for Slack.

        Args:
            title: Alert title
            message: Detailed alert message
            service: Service name that triggered the alert
            severity: Alert severity level
            channel: Target Slack channel (optional)
            additional_fields: Additional fields to include

        Returns:
            ModelNotificationRequest configured for Slack
        """
        fields = [
            {
                "title": "Service",
                "value": service,
                "short": True
            },
            {
                "title": "Timestamp",
                "value": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "short": True
            },
            {
                "title": "Severity",
                "value": severity.name,
                "short": True
            }
        ]

        if additional_fields:
            fields.extend(additional_fields)

        payload = {
            "text": f"🚨 {title}",
            "channel": channel.value if channel else self.default_channel.value,
            "username": "ONEX Infrastructure",
            "icon_emoji": ":warning:" if severity in [SlackPriority.CRITICAL, SlackPriority.HIGH] else ":information_source:",
            "attachments": [
                {
                    "color": severity.value,
                    "title": title,
                    "text": message,
                    "fields": fields,
                    "footer": "ONEX Infrastructure Monitoring",
                    "footer_icon": "https://github.com/favicon.ico",
                    "ts": int(datetime.utcnow().timestamp())
                }
            ]
        }

        return ModelNotificationRequest(
            url=self.webhook_url,
            method=EnumNotificationMethod.POST,
            payload=payload,
            retry_policy=self._get_retry_policy(severity)
        )

    def create_circuit_breaker_alert(
        self,
        service: str,
        state: str,
        destination: str,
        failure_count: int = None,
        last_error: str = None
    ) -> ModelNotificationRequest:
        """Create circuit breaker state change alert."""

        severity = SlackPriority.CRITICAL if state == "OPEN" else SlackPriority.MEDIUM
        emoji = "🔴" if state == "OPEN" else "🟢" if state == "CLOSED" else "🟡"

        fields = [
            {
                "title": "Service",
                "value": service,
                "short": True
            },
            {
                "title": "New State",
                "value": f"{emoji} {state}",
                "short": True
            },
            {
                "title": "Destination",
                "value": destination,
                "short": False
            }
        ]

        if failure_count is not None:
            fields.append({
                "title": "Failure Count",
                "value": str(failure_count),
                "short": True
            })

        if last_error:
            fields.append({
                "title": "Last Error",
                "value": last_error[:200] + ("..." if len(last_error) > 200 else ""),
                "short": False
            })

        message = f"Circuit breaker for {service} → {destination} changed to {state}"

        return self.create_infrastructure_alert(
            title=f"Circuit Breaker State Change: {service}",
            message=message,
            service=service,
            severity=severity,
            channel=SlackChannel.CRITICAL if state == "OPEN" else SlackChannel.MONITORING,
            additional_fields=fields[2:]  # Skip service and state since they're in base fields
        )

    def create_deployment_notification(
        self,
        service: str,
        version: str,
        environment: str,
        status: str,
        deployer: str = None
    ) -> ModelNotificationRequest:
        """Create deployment status notification."""

        severity = SlackPriority.CRITICAL if status == "FAILED" else SlackPriority.INFO
        emoji = "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "🚀"

        fields = [
            {
                "title": "Environment",
                "value": environment,
                "short": True
            },
            {
                "title": "Version",
                "value": version,
                "short": True
            },
            {
                "title": "Status",
                "value": f"{emoji} {status}",
                "short": True
            }
        ]

        if deployer:
            fields.append({
                "title": "Deployed By",
                "value": deployer,
                "short": True
            })

        return self.create_infrastructure_alert(
            title=f"Deployment: {service}",
            message=f"{service} deployment to {environment} has {status.lower()}",
            service=service,
            severity=severity,
            channel=SlackChannel.DEPLOYMENTS,
            additional_fields=fields
        )

    def create_performance_alert(
        self,
        service: str,
        metric: str,
        current_value: str,
        threshold: str,
        duration: str = None
    ) -> ModelNotificationRequest:
        """Create performance threshold alert."""

        fields = [
            {
                "title": "Metric",
                "value": metric,
                "short": True
            },
            {
                "title": "Current Value",
                "value": current_value,
                "short": True
            },
            {
                "title": "Threshold",
                "value": threshold,
                "short": True
            }
        ]

        if duration:
            fields.append({
                "title": "Duration",
                "value": duration,
                "short": True
            })

        return self.create_infrastructure_alert(
            title=f"Performance Alert: {service}",
            message=f"{metric} has exceeded threshold: {current_value} > {threshold}",
            service=service,
            severity=SlackPriority.HIGH,
            channel=SlackChannel.MONITORING,
            additional_fields=fields
        )

    def create_security_alert(
        self,
        service: str,
        alert_type: str,
        details: str,
        source_ip: str = None,
        user: str = None
    ) -> ModelNotificationRequest:
        """Create security alert notification."""

        fields = [
            {
                "title": "Alert Type",
                "value": alert_type,
                "short": True
            }
        ]

        if source_ip:
            fields.append({
                "title": "Source IP",
                "value": source_ip,
                "short": True
            })

        if user:
            fields.append({
                "title": "User",
                "value": user,
                "short": True
            })

        return self.create_infrastructure_alert(
            title=f"🔒 Security Alert: {alert_type}",
            message=details,
            service=service,
            severity=SlackPriority.CRITICAL,
            channel=SlackChannel.SECURITY,
            additional_fields=fields
        )

    def _get_retry_policy(self, severity: SlackPriority) -> ModelNotificationRetryPolicy:
        """Get retry policy based on alert severity."""

        if severity == SlackPriority.CRITICAL:
            # Critical alerts: More aggressive retry
            return ModelNotificationRetryPolicy(
                max_attempts=5,
                backoff_strategy=EnumBackoffStrategy.EXPONENTIAL,
                delay_seconds=2.0
            )
        elif severity == SlackPriority.HIGH:
            # High priority: Standard retry
            return ModelNotificationRetryPolicy(
                max_attempts=3,
                backoff_strategy=EnumBackoffStrategy.EXPONENTIAL,
                delay_seconds=5.0
            )
        else:
            # Medium/Info: Light retry
            return ModelNotificationRetryPolicy(
                max_attempts=2,
                backoff_strategy=EnumBackoffStrategy.LINEAR,
                delay_seconds=10.0
            )


# Production webhook factory
def create_production_slack_webhook(webhook_url: str) -> ProductionSlackWebhook:
    """
    Factory function to create production Slack webhook integration.

    Args:
        webhook_url: Your Slack webhook URL from https://api.slack.com/apps

    Returns:
        Configured ProductionSlackWebhook instance

    Example:
        webhook = create_production_slack_webhook("https://hooks.slack.com/services/...")

        # Create circuit breaker alert
        alert = webhook.create_circuit_breaker_alert(
            service="postgresql_adapter",
            state="OPEN",
            destination="postgres://production-db",
            failure_count=5,
            last_error="Connection timeout after 30s"
        )
    """
    return ProductionSlackWebhook(webhook_url)