"""
Slack Webhook Configuration Model for ONEX Infrastructure.

This module provides injectable configuration models for Slack webhook
integration following ONEX contract-driven configuration standards.
"""


from pydantic import BaseModel, Field, HttpUrl

from omnibase_infra.enums.enum_slack_channel import EnumSlackChannel


class ModelSlackWebhookConfig(BaseModel):
    """
    Injectable Slack webhook configuration model.

    This model replaces hardcoded configuration values with
    contract-driven configuration following ONEX standards.
    """

    webhook_url: HttpUrl = Field(
        description="Slack webhook URL for sending notifications",
    )

    default_channel: EnumSlackChannel = Field(
        default=EnumSlackChannel.ALERTS,
        description="Default channel for notifications when none specified",
    )

    username: str = Field(
        default="ONEX Infrastructure",
        description="Display name for webhook messages",
        min_length=1,
        max_length=80,
    )

    footer_text: str = Field(
        default="ONEX Infrastructure Monitoring",
        description="Footer text displayed in message attachments",
        min_length=1,
        max_length=300,
    )

    footer_icon_url: HttpUrl | None = Field(
        default="https://github.com/favicon.ico",
        description="Icon URL displayed next to footer text",
    )

    critical_icon_emoji: str = Field(
        default=":warning:",
        description="Emoji for critical/high priority alerts",
    )

    info_icon_emoji: str = Field(
        default=":information_source:",
        description="Emoji for medium/info priority alerts",
    )
