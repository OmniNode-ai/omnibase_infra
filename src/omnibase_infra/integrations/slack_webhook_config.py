"""
Production Slack Webhook Configuration for ONEX Infrastructure.

This module provides production-ready Slack webhook configurations for various
infrastructure alerts and notifications in the ONEX system using injectable
configuration models and proper Pydantic data structures.
"""

from typing import List, Optional
from datetime import datetime

from omnibase_core.enums.enum_notification_method import EnumNotificationMethod
from omnibase_core.enums.enum_backoff_strategy import EnumBackoffStrategy
from omnibase_core.core.onex_container import ONEXContainer
from omnibase_infra.models.notification.model_notification_request import ModelNotificationRequest
from omnibase_infra.models.notification.model_notification_retry_policy import ModelNotificationRetryPolicy
from omnibase_infra.models.slack.model_slack_webhook_config import ModelSlackWebhookConfig
from omnibase_infra.models.slack.model_slack_payload import ModelSlackPayload
from omnibase_infra.models.slack.model_slack_attachment import ModelSlackAttachment
from omnibase_infra.models.slack.model_slack_field import ModelSlackField
from omnibase_infra.enums.enum_slack_channel import EnumSlackChannel
from omnibase_infra.enums.enum_slack_priority import EnumSlackPriority


class ProductionSlackWebhook:
    """
    Production-ready Slack webhook service with injectable configuration.

    This service uses contract-driven configuration and proper Pydantic models
    to ensure type safety and ONEX compliance.
    """

    def __init__(self, container: ONEXContainer):
        """
        Initialize production Slack webhook with dependency injection.

        Args:
            container: ONEX container with injected dependencies
        """
        self.container = container
        # Configuration will be injected via container in actual implementation
        # For now, we maintain backward compatibility while showing the pattern
        self._config: Optional[ModelSlackWebhookConfig] = None

    def configure(self, config: ModelSlackWebhookConfig) -> None:
        """
        Configure the webhook with injectable configuration model.

        Args:
            config: Validated Slack webhook configuration
        """
        self._config = config

    def create_infrastructure_alert(
        self,
        title: str,
        message: str,
        service: str,
        severity: EnumSlackPriority = EnumSlackPriority.MEDIUM,
        channel: Optional[EnumSlackChannel] = None,
        additional_fields: Optional[List[ModelSlackField]] = None
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
        if not self._config:
            raise ValueError("Slack webhook not configured. Call configure() first.")

        # Create structured fields using proper Pydantic models
        fields = [
            ModelSlackField(
                title="Service",
                value=service,
                short=True
            ),
            ModelSlackField(
                title="Timestamp",
                value=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                short=True
            ),
            ModelSlackField(
                title="Severity",
                value=severity.name,
                short=True
            )
        ]

        if additional_fields:
            fields.extend(additional_fields)

        # Create attachment using proper Pydantic model
        attachment = ModelSlackAttachment(
            color=severity.value,
            title=title,
            text=message,
            fields=fields,
            footer=self._config.footer_text,
            footer_icon=self._config.footer_icon_url,
            ts=int(datetime.utcnow().timestamp())
        )

        # Determine icon based on severity
        icon_emoji = (
            self._config.critical_icon_emoji
            if severity in [EnumSlackPriority.CRITICAL, EnumSlackPriority.HIGH]
            else self._config.info_icon_emoji
        )

        # Create complete payload using proper Pydantic model
        payload = ModelSlackPayload(
            text=f"🚨 {title}",
            channel=channel.value if channel else self._config.default_channel.value,
            username=self._config.username,
            icon_emoji=icon_emoji,
            attachments=[attachment]
        )

        return ModelNotificationRequest(
            url=str(self._config.webhook_url),
            method=EnumNotificationMethod.POST,
            payload=payload.dict(),  # Convert to dict for request
            retry_policy=self._get_retry_policy(severity)
        )

    def create_circuit_breaker_alert(
        self,
        service: str,
        state: str,
        destination: str,
        failure_count: Optional[int] = None,
        last_error: Optional[str] = None
    ) -> ModelNotificationRequest:
        """Create circuit breaker state change alert."""

        severity = EnumSlackPriority.CRITICAL if state == "OPEN" else EnumSlackPriority.MEDIUM
        emoji = "🔴" if state == "OPEN" else "🟢" if state == "CLOSED" else "🟡"

        fields = [
            ModelSlackField(
                title="New State",
                value=f"{emoji} {state}",
                short=True
            ),
            ModelSlackField(
                title="Destination",
                value=destination,
                short=False
            )
        ]

        if failure_count is not None:
            fields.append(ModelSlackField(
                title="Failure Count",
                value=str(failure_count),
                short=True
            ))

        if last_error:
            fields.append(ModelSlackField(
                title="Last Error",
                value=last_error[:200] + ("..." if len(last_error) > 200 else ""),
                short=False
            ))

        message = f"Circuit breaker for {service} → {destination} changed to {state}"

        return self.create_infrastructure_alert(
            title=f"Circuit Breaker State Change: {service}",
            message=message,
            service=service,
            severity=severity,
            channel=EnumSlackChannel.CRITICAL if state == "OPEN" else EnumSlackChannel.MONITORING,
            additional_fields=fields
        )

    def create_deployment_notification(
        self,
        service: str,
        version: str,
        environment: str,
        status: str,
        deployer: Optional[str] = None
    ) -> ModelNotificationRequest:
        """Create deployment status notification."""

        severity = EnumSlackPriority.CRITICAL if status == "FAILED" else EnumSlackPriority.INFO
        emoji = "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "🚀"

        fields = [
            ModelSlackField(
                title="Environment",
                value=environment,
                short=True
            ),
            ModelSlackField(
                title="Version",
                value=version,
                short=True
            ),
            ModelSlackField(
                title="Status",
                value=f"{emoji} {status}",
                short=True
            )
        ]

        if deployer:
            fields.append(ModelSlackField(
                title="Deployed By",
                value=deployer,
                short=True
            ))

        return self.create_infrastructure_alert(
            title=f"Deployment: {service}",
            message=f"{service} deployment to {environment} has {status.lower()}",
            service=service,
            severity=severity,
            channel=EnumSlackChannel.DEPLOYMENTS,
            additional_fields=fields
        )

    def create_performance_alert(
        self,
        service: str,
        metric: str,
        current_value: str,
        threshold: str,
        duration: Optional[str] = None
    ) -> ModelNotificationRequest:
        """Create performance threshold alert."""

        fields = [
            ModelSlackField(
                title="Metric",
                value=metric,
                short=True
            ),
            ModelSlackField(
                title="Current Value",
                value=current_value,
                short=True
            ),
            ModelSlackField(
                title="Threshold",
                value=threshold,
                short=True
            )
        ]

        if duration:
            fields.append(ModelSlackField(
                title="Duration",
                value=duration,
                short=True
            ))

        return self.create_infrastructure_alert(
            title=f"Performance Alert: {service}",
            message=f"{metric} has exceeded threshold: {current_value} > {threshold}",
            service=service,
            severity=EnumSlackPriority.HIGH,
            channel=EnumSlackChannel.MONITORING,
            additional_fields=fields
        )

    def create_security_alert(
        self,
        service: str,
        alert_type: str,
        details: str,
        source_ip: Optional[str] = None,
        user: Optional[str] = None
    ) -> ModelNotificationRequest:
        """Create security alert notification."""

        fields = [
            ModelSlackField(
                title="Alert Type",
                value=alert_type,
                short=True
            )
        ]

        if source_ip:
            fields.append(ModelSlackField(
                title="Source IP",
                value=source_ip,
                short=True
            ))

        if user:
            fields.append(ModelSlackField(
                title="User",
                value=user,
                short=True
            ))

        return self.create_infrastructure_alert(
            title=f"🔒 Security Alert: {alert_type}",
            message=details,
            service=service,
            severity=EnumSlackPriority.CRITICAL,
            channel=EnumSlackChannel.SECURITY,
            additional_fields=fields
        )

    def _get_retry_policy(self, severity: EnumSlackPriority) -> ModelNotificationRetryPolicy:
        """Get retry policy based on alert severity."""

        if severity == EnumSlackPriority.CRITICAL:
            # Critical alerts: More aggressive retry
            return ModelNotificationRetryPolicy(
                max_attempts=5,
                backoff_strategy=EnumBackoffStrategy.EXPONENTIAL,
                delay_seconds=2.0
            )
        elif severity == EnumSlackPriority.HIGH:
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


# Factory function for backward compatibility
def create_production_slack_webhook(webhook_url: str) -> ProductionSlackWebhook:
    """
    Factory function to create production Slack webhook integration.

    NOTE: This function provides backward compatibility but should be replaced
    with proper dependency injection via ONEXContainer in production.

    Args:
        webhook_url: Your Slack webhook URL from https://api.slack.com/apps

    Returns:
        Configured ProductionSlackWebhook instance

    Example:
        webhook = create_production_slack_webhook("https://hooks.slack.com/services/...")

        # Configure with proper model
        config = ModelSlackWebhookConfig(webhook_url=webhook_url)
        webhook.configure(config)

        # Create circuit breaker alert
        alert = webhook.create_circuit_breaker_alert(
            service="postgresql_adapter",
            state="OPEN",
            destination="postgres://production-db",
            failure_count=5,
            last_error="Connection timeout after 30s"
        )
    """
    from omnibase_core.core.onex_container import ONEXContainer

    # Create minimal container for backward compatibility
    container = ONEXContainer()
    webhook = ProductionSlackWebhook(container)

    # Configure with provided webhook URL
    config = ModelSlackWebhookConfig(webhook_url=webhook_url)
    webhook.configure(config)

    return webhook