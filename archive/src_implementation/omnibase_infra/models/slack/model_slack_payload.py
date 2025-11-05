"""
Slack Payload Model for ONEX Infrastructure Notifications.

This module provides a Pydantic model for complete Slack webhook payloads
following ONEX standards for strong typing and data validation.
"""


from pydantic import BaseModel, Field

from omnibase_infra.models.slack.model_slack_attachment import ModelSlackAttachment


class ModelSlackPayload(BaseModel):
    """
    Complete Slack webhook payload model with proper validation.

    Represents the full payload sent to Slack webhook endpoints
    following the Slack API specification and ONEX standards.
    """

    text: str = Field(
        description="Main message text (fallback if attachments not supported)",
        min_length=1,
        max_length=4000,
    )

    channel: str | None = Field(
        default=None,
        description="Target channel (overrides webhook default)",
        max_length=80,
    )

    username: str = Field(
        default="ONEX Infrastructure",
        description="Display name for the webhook bot",
        min_length=1,
        max_length=80,
    )

    icon_emoji: str = Field(
        default=":information_source:",
        description="Emoji icon for the webhook bot",
        min_length=1,
        max_length=100,
    )

    attachments: list[ModelSlackAttachment] = Field(
        default_factory=list,
        description="Rich attachments for structured content display",
        max_items=20,
    )
