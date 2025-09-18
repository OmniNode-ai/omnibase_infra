"""
Slack Attachment Model for ONEX Infrastructure Notifications.

This module provides a Pydantic model for Slack message attachments
following ONEX standards for strong typing and data validation.
"""

from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl

from omnibase_infra.models.integration.slack.model_slack_field import ModelSlackField


class ModelSlackAttachment(BaseModel):
    """
    Slack message attachment model with proper validation.

    Represents a rich attachment in a Slack message following
    the Slack API specification for structured content display.
    """

    color: str = Field(
        description="Attachment color bar (danger, warning, good, or hex code)",
        min_length=1,
        max_length=20,
    )

    title: str = Field(
        description="Attachment title displayed prominently",
        min_length=1,
        max_length=200,
    )

    text: str = Field(
        description="Main attachment content text",
        min_length=1,
        max_length=8000,
    )

    fields: list[ModelSlackField] = Field(
        default_factory=list,
        description="List of structured fields for data display",
        max_items=20,
    )

    footer: str = Field(
        default="ONEX Infrastructure Monitoring",
        description="Footer text displayed at bottom of attachment",
        max_length=300,
    )

    footer_icon: HttpUrl | None = Field(
        default="https://github.com/favicon.ico",
        description="Small icon displayed next to footer text",
    )

    ts: int = Field(
        default_factory=lambda: int(datetime.utcnow().timestamp()),
        description="Unix timestamp for attachment display",
    )
