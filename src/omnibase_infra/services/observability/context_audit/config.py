# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Configuration for context audit observability consumer (OMN-5240).

Loads from environment variables with OMNIBASE_INFRA_CONTEXT_AUDIT_ prefix.
"""

from __future__ import annotations

import logging
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from omnibase_infra.topics import (
    SUFFIX_OMNICLAUDE_AUDIT_COMPRESSION_TRIGGERED,
    SUFFIX_OMNICLAUDE_AUDIT_CONTEXT_BUDGET_EXCEEDED,
    SUFFIX_OMNICLAUDE_AUDIT_DISPATCH_VALIDATED,
    SUFFIX_OMNICLAUDE_AUDIT_RETURN_BOUNDED,
    SUFFIX_OMNICLAUDE_AUDIT_SCOPE_VIOLATION,
    SUFFIX_OMNICLAUDE_CONTEXT_AUDIT_DLQ,
)

logger = logging.getLogger(__name__)


class ConfigContextAuditConsumer(BaseSettings):
    """Configuration for the context audit observability Kafka consumer.

    Environment variables use the OMNIBASE_INFRA_CONTEXT_AUDIT_ prefix.
    Example: OMNIBASE_INFRA_CONTEXT_AUDIT_KAFKA_BOOTSTRAP_SERVERS=kafka:9092

    This consumer subscribes to context integrity audit topics and persists
    events to PostgreSQL for enforcement tracking and CLI queries.
    """

    model_config = SettingsConfigDict(
        env_prefix="OMNIBASE_INFRA_CONTEXT_AUDIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Kafka connection
    kafka_bootstrap_servers: str = Field(
        ...,
        description=(
            "Kafka bootstrap servers. Set via "
            "OMNIBASE_INFRA_CONTEXT_AUDIT_KAFKA_BOOTSTRAP_SERVERS env var."
        ),
    )
    kafka_group_id: str = Field(
        default="context-audit-postgres",
        description="Consumer group ID for offset tracking",
    )

    # Topics to subscribe — all omniclaude audit event topics (OMN-5234)
    topics: list[str] = Field(
        default_factory=lambda: [
            SUFFIX_OMNICLAUDE_AUDIT_DISPATCH_VALIDATED,
            SUFFIX_OMNICLAUDE_AUDIT_SCOPE_VIOLATION,
            SUFFIX_OMNICLAUDE_AUDIT_CONTEXT_BUDGET_EXCEEDED,
            SUFFIX_OMNICLAUDE_AUDIT_RETURN_BOUNDED,
            SUFFIX_OMNICLAUDE_AUDIT_COMPRESSION_TRIGGERED,
        ],
        description="Kafka topics to consume for context audit observability",
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

    # Session timeout tuning — prevents rebalance storms during brief processing delays
    session_timeout_ms: int = Field(
        default=45000,
        ge=6000,
        le=300000,
        description="Kafka session timeout in ms. Default 45s prevents rebalance storms.",
    )
    heartbeat_interval_ms: int = Field(
        default=15000,
        ge=1000,
        le=60000,
        description="Kafka heartbeat interval in ms. Should be ~1/3 of session_timeout_ms.",
    )
    max_poll_interval_ms: int = Field(
        default=300000,
        ge=10000,
        le=600000,
        description="Max time between poll() calls in ms before consumer eviction. Default 5 min.",
    )

    # PostgreSQL connection
    postgres_dsn: str = Field(
        description=(
            "PostgreSQL connection string. Set via "
            "OMNIBASE_INFRA_CONTEXT_AUDIT_POSTGRES_DSN env var."
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
            "the asyncio.wait_for timeout when polling Kafka. "
            "Configure via OMNIBASE_INFRA_CONTEXT_AUDIT_POLL_TIMEOUT_BUFFER_SECONDS env var."
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
        default=SUFFIX_OMNICLAUDE_CONTEXT_AUDIT_DLQ,
        description=(
            "Dead letter topic for permanently failed audit messages. "
            "Configure via OMNIBASE_INFRA_CONTEXT_AUDIT_DLQ_TOPIC env var."
        ),
    )
    dlq_enabled: bool = Field(
        default=True,
        description=(
            "Enable dead letter queue for permanently failed messages. "
            "Configure via OMNIBASE_INFRA_CONTEXT_AUDIT_DLQ_ENABLED env var."
        ),
    )
    max_retry_count: int = Field(
        default=3,
        ge=0,
        le=10,
        description=(
            "Maximum number of retries before sending a message to the DLQ. "
            "Configure via OMNIBASE_INFRA_CONTEXT_AUDIT_MAX_RETRY_COUNT env var."
        ),
    )

    # Health check
    health_check_port: int = Field(
        default=8093,
        ge=1024,
        le=65535,
        description="Port for HTTP health check endpoint",
    )
    health_check_host: str = Field(
        default="127.0.0.1",
        description=(
            "Host/IP for health check server binding. Default '127.0.0.1' restricts "
            "to localhost-only access for security. For container/Kubernetes deployments, "
            "override to '0.0.0.0' via OMNIBASE_INFRA_CONTEXT_AUDIT_HEALTH_CHECK_HOST "
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
            "Configure via OMNIBASE_INFRA_CONTEXT_AUDIT_HEALTH_CHECK_STALENESS_SECONDS env var."
        ),
    )
    health_check_poll_staleness_seconds: int = Field(
        default=60,
        ge=10,
        le=300,
        description=(
            "Maximum age in seconds for the last poll before the health check "
            "reports DEGRADED status. "
            "Configure via OMNIBASE_INFRA_CONTEXT_AUDIT_HEALTH_CHECK_POLL_STALENESS_SECONDS env var."
        ),
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
                "No topics configured for context audit consumer. "
                "Provide explicit 'topics' via configuration or environment variable."
            )
        return self
