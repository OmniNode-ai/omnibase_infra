# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Configuration for the Waitlist Signup Slack Notifier consumer.

Related Tickets:
    - OMN-7199: Slack notification on new waitlist signup
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from omnibase_infra.services.waitlist_signup_notifier.topics import (
    TOPIC_WAITLIST_SIGNUP,
)


class ConfigWaitlistSignupNotifier(BaseSettings):
    """Configuration sourced from environment variables.

    Prefix: WAITLIST_NOTIFIER_

    The ``slack_bot_token`` and ``slack_channel_id`` fields accept both
    prefixed (``WAITLIST_NOTIFIER_SLACK_BOT_TOKEN``) and unprefixed
    (``SLACK_BOT_TOKEN``) env var names via ``AliasChoices``, matching the
    pattern in ``session/config_store.py``.
    """

    model_config = SettingsConfigDict(
        env_prefix="WAITLIST_NOTIFIER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    kafka_bootstrap_servers: str = Field(
        ...,
        description="Kafka bootstrap servers",
    )
    kafka_topic: str = Field(
        default=TOPIC_WAITLIST_SIGNUP,
        description="Topic to subscribe to",
    )
    kafka_group_id: str = Field(
        default="waitlist-signup-slack-notifier",
        description="Consumer group ID",
    )
    health_check_port: int = Field(
        default=8095,
        description="Health check HTTP port",
    )
    health_check_host: str = Field(
        default="0.0.0.0",  # noqa: S104
        description="Health check bind address",
    )
    slack_bot_token: str = Field(
        default="",
        description="Slack Bot Token (accepts WAITLIST_NOTIFIER_SLACK_BOT_TOKEN or SLACK_BOT_TOKEN)",
        validation_alias=AliasChoices(
            "WAITLIST_NOTIFIER_SLACK_BOT_TOKEN",
            "SLACK_BOT_TOKEN",
        ),
    )
    slack_channel_id: str = Field(
        default="",
        description="Slack channel ID (accepts WAITLIST_NOTIFIER_SLACK_CHANNEL_ID or SLACK_CHANNEL_ID)",
        validation_alias=AliasChoices(
            "WAITLIST_NOTIFIER_SLACK_CHANNEL_ID",
            "SLACK_CHANNEL_ID",
        ),
    )
