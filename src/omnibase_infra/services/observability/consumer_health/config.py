# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Configuration for consumer health read-model projection (OMN-6757).

Loads from environment variables with OMNIBASE_INFRA_CONSUMER_HEALTH_ prefix.
"""

from __future__ import annotations

import logging
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from omnibase_infra.topics.platform_topic_suffixes import SUFFIX_CONSUMER_HEALTH

logger = logging.getLogger(__name__)


class ConfigConsumerHealthProjection(BaseSettings):
    """Configuration for the consumer health read-model Kafka consumer.

    Environment variables use the OMNIBASE_INFRA_CONSUMER_HEALTH_ prefix.
    Example: OMNIBASE_INFRA_CONSUMER_HEALTH_KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    """

    model_config = SettingsConfigDict(
        env_prefix="OMNIBASE_INFRA_CONSUMER_HEALTH_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Kafka connection
    kafka_bootstrap_servers: str = Field(
        default="localhost:19092",
        description="Kafka bootstrap servers.",
    )
    kafka_group_id: str = Field(
        default="consumer-health-projection-postgres",
        description="Consumer group ID for offset tracking",
    )

    # Topics
    topics: list[str] = Field(
        default_factory=lambda: [SUFFIX_CONSUMER_HEALTH],
        description="Kafka topics to consume for consumer health projection",
    )

    # Consumer behavior
    auto_offset_reset: str = Field(
        default="earliest",
        description="Where to start consuming if no offset exists",
    )
    enable_auto_commit: bool = Field(
        default=False,
        description="Disable auto-commit for at-least-once delivery",
    )

    # Session timeout tuning
    session_timeout_ms: int = Field(
        default=45000,
        ge=6000,
        le=300000,
        description="Kafka session timeout in ms.",
    )
    heartbeat_interval_ms: int = Field(
        default=15000,
        ge=1000,
        le=60000,
        description="Kafka heartbeat interval in ms.",
    )
    max_poll_interval_ms: int = Field(
        default=300000,
        ge=10000,
        le=600000,
        description="Max time between poll() calls in ms.",
    )

    # PostgreSQL
    postgres_dsn: str = Field(
        description="PostgreSQL connection string.",
    )

    # Batch processing
    batch_size: int = Field(
        default=100, ge=1, le=1000, description="Maximum records per batch write"
    )
    batch_timeout_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Timeout for batch accumulation in milliseconds",
    )
    poll_timeout_buffer_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=30.0,
        description="Additional buffer time added to batch_timeout_ms for asyncio.wait_for.",
    )

    # Circuit breaker
    circuit_breaker_threshold: int = Field(
        default=5, ge=1, le=100, description="Failures before circuit opens"
    )
    circuit_breaker_reset_timeout: float = Field(
        default=60.0,
        ge=1.0,
        le=3600.0,
        description="Seconds before circuit half-opens",
    )
    circuit_breaker_half_open_successes: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Successes required to close circuit from half-open",
    )

    # Health check
    health_check_port: int = Field(
        default=8094, ge=1024, le=65535, description="Port for HTTP health check"
    )
    health_check_host: str = Field(
        default="127.0.0.1", description="Host for health check server"
    )
    health_check_staleness_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Max age in seconds for last write before DEGRADED status.",
    )
    health_check_poll_staleness_seconds: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Max age in seconds for last poll before DEGRADED status.",
    )

    @model_validator(mode="after")
    def validate_session_timeout_ratio(self) -> Self:
        """Validate heartbeat < session_timeout and max_poll >= session_timeout."""
        if self.heartbeat_interval_ms >= self.session_timeout_ms:
            raise ValueError(
                f"heartbeat_interval_ms ({self.heartbeat_interval_ms}) must be "
                f"< session_timeout_ms ({self.session_timeout_ms})"
            )
        if self.max_poll_interval_ms < self.session_timeout_ms:
            raise ValueError(
                f"max_poll_interval_ms ({self.max_poll_interval_ms}) must be "
                f">= session_timeout_ms ({self.session_timeout_ms})"
            )
        return self

    @model_validator(mode="after")
    def validate_topic_configuration(self) -> Self:
        """Ensure topics are configured."""
        if not self.topics:
            from omnibase_infra.errors import ProtocolConfigurationError

            raise ProtocolConfigurationError(
                "No topics configured for consumer health projection consumer."
            )
        return self
