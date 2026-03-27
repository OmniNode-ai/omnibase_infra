# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Configuration for the post-merge consumer chain.

Loads from environment variables with ``POST_MERGE_`` prefix.

Related Tickets:
    - OMN-6727: post-merge consumer chain
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from omnibase_infra.topics.platform_topic_suffixes import SUFFIX_GITHUB_PR_MERGED


class ConfigPostMergeConsumer(BaseSettings):
    """Configuration for the post-merge consumer.

    Environment variables use the ``POST_MERGE_`` prefix.
    Example: ``POST_MERGE_KAFKA_BOOTSTRAP_SERVERS=localhost:19092``

    Related Tickets:
        - OMN-6727: post-merge consumer chain
    """

    model_config = SettingsConfigDict(
        env_prefix="POST_MERGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Kafka connection
    kafka_bootstrap_servers: str = Field(
        default="localhost:19092",
        description="Kafka bootstrap servers",
    )
    kafka_group_id: str = Field(
        default="post-merge-consumer",
        description="Consumer group ID for offset tracking",
    )

    # Input topic
    input_topic: str = Field(
        default=SUFFIX_GITHUB_PR_MERGED,
        description="Kafka topic to consume PR merged events from",
    )

    # Consumer behaviour
    auto_offset_reset: str = Field(
        default="earliest",
        description="Where to start consuming if no committed offset exists",
    )

    # GitHub access for diff retrieval
    github_token: str = Field(
        default="",
        description=(
            "GitHub personal access token for fetching PR diffs. "
            "Set via POST_MERGE_GITHUB_TOKEN env var."
        ),
    )

    # Linear access for auto-ticket creation
    linear_api_key: str = Field(
        default="",
        description=(
            "Linear API key for auto-creating tickets from findings. "
            "Set via POST_MERGE_LINEAR_API_KEY env var."
        ),
    )
    linear_team_id: str = Field(
        default="",
        description="Linear team ID to create tickets under",
    )

    # Hostile review settings
    hostile_review_enabled: bool = Field(
        default=True,
        description="Enable hostile review stage",
    )

    # Contract sweep settings
    contract_sweep_enabled: bool = Field(
        default=True,
        description="Enable contract sweep (check-drift) stage",
    )
    contract_sweep_contracts_dir: str = Field(
        default="src/omnibase_infra/nodes",
        description="Relative path to contracts directory for check-drift",
    )

    # Integration check settings
    integration_check_enabled: bool = Field(
        default=True,
        description="Enable integration check stage",
    )

    # Ticket creation
    auto_ticket_min_severity: str = Field(
        default="medium",
        description=(
            "Minimum finding severity to auto-create a Linear ticket. "
            "One of: critical, high, medium, low, info"
        ),
    )
    dry_run: bool = Field(
        default=False,
        description=(
            "Dry-run mode: process events but skip Linear ticket creation. "
            "Useful for testing the consumer chain without side effects."
        ),
    )

    # Health check
    health_check_port: int = Field(
        default=8088,
        ge=1024,
        le=65535,
        description="Port for HTTP health check endpoint",
    )
    health_check_host: str = Field(
        default="127.0.0.1",
        description="Host/IP for health check server binding",
    )


__all__ = ["ConfigPostMergeConsumer"]
