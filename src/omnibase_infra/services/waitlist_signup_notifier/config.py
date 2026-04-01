# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Configuration for the Waitlist Signup Slack Notifier consumer.

Related Tickets:
    - OMN-7199: Slack notification on new waitlist signup
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings

from omnibase_infra.services.waitlist_signup_notifier.topics import (
    TOPIC_WAITLIST_SIGNUP,
)


class ConfigWaitlistSignupNotifier(BaseSettings):
    """Configuration sourced from environment variables.

    Prefix: WAITLIST_NOTIFIER_
    """

    model_config = {"env_prefix": "WAITLIST_NOTIFIER_"}

    kafka_bootstrap_servers: str = Field(
        default="localhost:19092",
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
        description="Slack Bot Token (falls back to SLACK_BOT_TOKEN env var)",
    )
    slack_channel_id: str = Field(
        default="",
        description="Slack channel ID (falls back to SLACK_CHANNEL_ID env var)",
    )
