"""
Webhook Payload Models - ONEX Agent-Safe Design

Strongly-typed webhook payload models that eliminate Dict usage and provide
agent-safe interfaces for automated webhook generation and delivery.

ONEX Principle: Zero flexibility - maximum predictability for agent execution.
"""

from datetime import datetime
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class ModelWebhookAttachment(BaseModel):
    """Structured webhook attachment model."""

    title: str = Field(..., description="Attachment title")
    text: str = Field(..., description="Attachment content")
    color: str | None = Field(
        default=None, description="Color indicator (hex or semantic)",
    )
    timestamp: datetime | None = Field(default=None, description="Attachment timestamp")

    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelSlackWebhookPayload(BaseModel):
    """Slack-specific webhook payload with strict typing."""

    webhook_type: Literal["slack"] = Field(
        default="slack", description="Webhook type discriminator",
    )
    text: str = Field(..., description="Primary message text")
    channel: str | None = Field(default=None, description="Target Slack channel")
    username: str | None = Field(default=None, description="Bot username override")
    icon_emoji: str | None = Field(default=None, description="Bot emoji icon")
    attachments: list[ModelWebhookAttachment] | None = Field(
        default=None, description="Message attachments",
    )

    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelDiscordWebhookPayload(BaseModel):
    """Discord-specific webhook payload with strict typing."""

    webhook_type: Literal["discord"] = Field(
        default="discord", description="Webhook type discriminator",
    )
    content: str = Field(..., description="Primary message content")
    username: str | None = Field(default=None, description="Bot username override")
    avatar_url: str | None = Field(default=None, description="Bot avatar URL")
    embeds: list[ModelWebhookAttachment] | None = Field(
        default=None, description="Discord embeds",
    )

    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelTeamsWebhookPayload(BaseModel):
    """Microsoft Teams webhook payload with strict typing."""

    webhook_type: Literal["teams"] = Field(
        default="teams", description="Webhook type discriminator",
    )
    summary: str = Field(..., description="Message summary")
    text: str = Field(..., description="Message content")
    title: str | None = Field(default=None, description="Message title")
    theme_color: str | None = Field(default=None, description="Theme color (hex)")

    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelInfrastructureAlertPayload(BaseModel):
    """Infrastructure alert payload for ONEX systems."""

    webhook_type: Literal["infrastructure_alert"] = Field(
        default="infrastructure_alert", description="Webhook type discriminator",
    )

    # Required alert fields
    alert_level: Literal["info", "warning", "critical"] = Field(
        ..., description="Alert severity level",
    )
    service_name: str = Field(..., description="Service generating the alert")
    alert_message: str = Field(..., description="Primary alert message")

    # Optional context
    node_id: str | None = Field(default=None, description="ONEX node identifier")
    correlation_id: str | None = Field(
        default=None, description="Request correlation ID",
    )
    timestamp: datetime | None = Field(default=None, description="Alert timestamp")
    metrics: list[str] | None = Field(default=None, description="Related metrics")

    model_config = ConfigDict(frozen=True, extra="forbid")


# Agent-Safe Union Type for Webhook Payloads
ModelWebhookPayloadUnion = Union[
    ModelSlackWebhookPayload,
    ModelDiscordWebhookPayload,
    ModelTeamsWebhookPayload,
    ModelInfrastructureAlertPayload,
]


class ModelWebhookPayloadWrapper(BaseModel):
    """
    Wrapper for webhook payloads with agent-safe validation.

    This eliminates the need for Dict types while providing
    compile-time safety for agent-generated webhooks.
    """

    payload: ModelWebhookPayloadUnion = Field(
        ..., description="Strongly-typed webhook payload",
    )
    target_platform: Literal["slack", "discord", "teams", "infrastructure_alert"] = (
        Field(
            ...,
            description="Target webhook platform",
        )
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    @property
    def is_slack(self) -> bool:
        """Check if this is a Slack webhook payload."""
        return self.target_platform == "slack"

    @property
    def is_discord(self) -> bool:
        """Check if this is a Discord webhook payload."""
        return self.target_platform == "discord"

    @property
    def is_teams(self) -> bool:
        """Check if this is a Teams webhook payload."""
        return self.target_platform == "teams"

    @property
    def is_infrastructure_alert(self) -> bool:
        """Check if this is an infrastructure alert payload."""
        return self.target_platform == "infrastructure_alert"
