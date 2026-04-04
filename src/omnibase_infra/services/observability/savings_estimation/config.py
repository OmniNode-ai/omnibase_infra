# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Configuration for savings estimation consumer.

Related Tickets:
    - OMN-5550: Create ServiceSavingsEstimator Kafka consumer
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from omnibase_infra.topics import topic_keys


def _default_consumed_topics() -> list[str]:
    """Resolve consumed topics lazily via ServiceTopicRegistry."""
    from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry

    reg = ServiceTopicRegistry.from_defaults()
    return [
        reg.resolve(topic_keys.LLM_CALL_COMPLETED),
        reg.resolve(topic_keys.SESSION_OUTCOME_CANONICAL),
        reg.resolve(topic_keys.HOOK_CONTEXT_INJECTED),
        reg.resolve(topic_keys.VALIDATOR_CATCH),
        reg.resolve(topic_keys.PATTERN_ENFORCEMENT),
    ]


def _default_produce_topic() -> str:
    """Resolve produce topic lazily via ServiceTopicRegistry."""
    from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry

    return ServiceTopicRegistry.from_defaults().resolve(topic_keys.SAVINGS_ESTIMATED)


class ConfigSavingsEstimation(BaseSettings):
    """Configuration for the savings estimation Kafka consumer."""

    model_config = SettingsConfigDict(
        env_prefix="OMNIBASE_INFRA_SAVINGS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    kafka_bootstrap_servers: str = Field(
        ...,
        description="Kafka bootstrap servers.",
    )
    kafka_group_id: str = Field(
        default="savings-estimation",
        description="Consumer group ID for offset tracking.",
    )

    consumed_topics: list[str] = Field(
        default_factory=_default_consumed_topics,
        description="Kafka topics to consume.",
    )

    produce_topic: str = Field(
        default_factory=_default_produce_topic,
        description="Kafka topic to produce savings estimates.",
    )

    auto_offset_reset: Literal["earliest", "latest"] = Field(
        default="earliest",
        description="Where to start consuming if no offset exists.",
    )

    batch_size: int = Field(default=100, ge=1, le=1000)
    batch_timeout_ms: int = Field(default=1000, ge=100, le=60000)

    max_sessions: int = Field(
        default=1000,
        ge=100,
        le=100000,
        description="Maximum sessions in the LRU correlation buffer.",
    )

    grace_window_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Grace window after session-outcome before finalizing.",
    )

    session_timeout_seconds: float = Field(
        default=3600.0,
        ge=60.0,
        le=86400.0,
        description="Maximum age of a session in the buffer before timeout.",
    )

    finalized_session_cache_size: int = Field(
        default=10000,
        ge=1000,
        le=1000000,
        description="Size of in-memory finalized session set for dedup optimization.",
    )

    schema_version: str = Field(
        default="1.0",
        description="Schema version for produced savings events.",
    )


__all__: list[str] = ["ConfigSavingsEstimation"]
