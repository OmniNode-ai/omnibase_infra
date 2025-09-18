"""
Slack Field Model for ONEX Infrastructure Notifications.

This module provides a Pydantic model for Slack attachment fields
following ONEX standards for strong typing and data validation.
"""

from pydantic import BaseModel, Field


class ModelSlackField(BaseModel):
    """
    Slack attachment field model with proper validation.

    Represents a single field in a Slack message attachment following
    the Slack API specification for structured data display.
    """

    title: str = Field(
        description="Field title displayed in bold",
        min_length=1,
        max_length=100,
    )

    value: str = Field(
        description="Field value content",
        min_length=1,
        max_length=2000,
    )

    short: bool = Field(
        default=True,
        description="Whether field should be displayed side-by-side with other short fields",
    )
