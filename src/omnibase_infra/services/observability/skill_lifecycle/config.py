# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration for skill lifecycle observability consumer (OMN-2934).

Loads from environment variables with OMNIBASE_INFRA_SKILL_LIFECYCLE_ prefix.
"""

from __future__ import annotations

import logging
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class ConfigSkillLifecycleConsumer(BaseSettings):
    """Configuration for the skill lifecycle observability Kafka consumer.

    Environment variables use the OMNIBASE_INFRA_SKILL_LIFECYCLE_ prefix.
    Example: OMNIBASE_INFRA_SKILL_LIFECYCLE_KAFKA_BOOTSTRAP_SERVERS=kafka:9092

    This consumer subscribes to skill lifecycle topics and persists events
    to PostgreSQL for omnidash skill monitoring.
    """

    model_config = SettingsConfigDict(
        env_prefix="OMNIBASE_INFRA_SKILL_LIFECYCLE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Kafka connection
    kafka_bootstrap_servers: str = Field(
        default="localhost:9092",
        description=(
            "Kafka bootstrap servers. Set via "
            "OMNIBASE_INFRA_SKILL_LIFECYCLE_KAFKA_BOOTSTRAP_SERVERS env var for production."
        ),
    )
    kafka_group_id: str = Field(
        default="skill-lifecycle-postgres",
        description="Consumer group ID for offset tracking",
    )

    # Topics to subscribe (OMN-2934: skill lifecycle observability)
    topics: list[str] = Field(
        default_factory=lambda: [
            "onex.evt.omniclaude.skill-started.v1",
            "onex.evt.omniclaude.skill-completed.v1",
        ],
        description="Kafka topics to consume for skill lifecycle observability",
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

    # PostgreSQL connection
    postgres_dsn: str = Field(
        description=(
            "PostgreSQL connection string. Set via "
            "OMNIBASE_INFRA_SKILL_LIFECYCLE_POSTGRES_DSN env var."
        ),
    )

    # Batch processing
    batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum records per batch write",
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
        description=(
            "Additional buffer time in seconds added to batch_timeout_ms for "
            "the asyncio.wait_for timeout when polling Kafka. This buffer accounts "
            "for Kafka client internal processing overhead beyond the poll timeout. "
            "Configure via OMNIBASE_INFRA_SKILL_LIFECYCLE_POLL_TIMEOUT_BUFFER_SECONDS env var."
        ),
    )

    # Circuit breaker
    circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Failures before circuit opens",
    )
    circuit_breaker_reset_timeout: float = Field(
        default=60.0,
        ge=1.0,
        le=3600.0,
        description="Seconds before circuit half-opens for retry",
    )
    circuit_breaker_half_open_successes: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Successful requests required to close circuit from half-open state",
    )

    # Dead Letter Queue
    dlq_topic: str = Field(
        default="onex.evt.omniclaude.skill-lifecycle-dlq.v1",
        description=(
            "Dead letter topic for permanently failed skill lifecycle messages. "
            "Configure via OMNIBASE_INFRA_SKILL_LIFECYCLE_DLQ_TOPIC env var."
        ),
    )
    dlq_enabled: bool = Field(
        default=True,
        description=(
            "Enable dead letter queue for permanently failed messages. "
            "Configure via OMNIBASE_INFRA_SKILL_LIFECYCLE_DLQ_ENABLED env var."
        ),
    )
    max_retry_count: int = Field(
        default=3,
        ge=0,
        le=10,
        description=(
            "Maximum number of retries before sending a message to the DLQ. "
            "Configure via OMNIBASE_INFRA_SKILL_LIFECYCLE_MAX_RETRY_COUNT env var."
        ),
    )

    # Health check
    health_check_port: int = Field(
        default=8092,
        ge=1024,
        le=65535,
        description="Port for HTTP health check endpoint",
    )
    health_check_host: str = Field(
        default="127.0.0.1",
        description=(
            "Host/IP for health check server binding. Default '127.0.0.1' restricts "
            "to localhost-only access for security. For container/Kubernetes deployments, "
            "override to '0.0.0.0' via OMNIBASE_INFRA_SKILL_LIFECYCLE_HEALTH_CHECK_HOST "
            "env var to allow external probe access."
        ),
    )
    health_check_staleness_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description=(
            "Maximum age in seconds for the last successful write before "
            "the health check reports DEGRADED status. "
            "Configure via OMNIBASE_INFRA_SKILL_LIFECYCLE_HEALTH_CHECK_STALENESS_SECONDS env var."
        ),
    )
    health_check_poll_staleness_seconds: int = Field(
        default=60,
        ge=10,
        le=300,
        description=(
            "Maximum age in seconds for the last poll before the health check "
            "reports DEGRADED status. "
            "Configure via OMNIBASE_INFRA_SKILL_LIFECYCLE_HEALTH_CHECK_POLL_STALENESS_SECONDS env var."
        ),
    )

    @model_validator(mode="after")
    def validate_topic_configuration(self) -> Self:
        """Ensure topics are configured.

        Fails fast if no topics provided, preventing silent misconfiguration.

        Returns:
            Self if validation passes.

        Raises:
            ProtocolConfigurationError: If no topics are configured.
        """
        if not self.topics:
            from omnibase_infra.errors import ProtocolConfigurationError

            raise ProtocolConfigurationError(
                "No topics configured for skill lifecycle consumer. "
                "Provide explicit 'topics' via configuration or environment variable."
            )
        return self
