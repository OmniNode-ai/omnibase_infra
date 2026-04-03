# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Configuration for session event consumers.

Loads from environment variables with OMNIBASE_INFRA_SESSION_CONSUMER_ prefix.

Moved from omniclaude as part of OMN-1526 architectural cleanup.
"""

from __future__ import annotations

import logging
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class ConfigSessionConsumer(BaseSettings):
    """Configuration for the Claude session event Kafka consumer.

    Environment variables use the OMNIBASE_INFRA_SESSION_CONSUMER_ prefix.
    Example: OMNIBASE_INFRA_SESSION_CONSUMER_BOOTSTRAP_SERVERS=kafka.example.com:9092
    """

    model_config = SettingsConfigDict(
        env_prefix="OMNIBASE_INFRA_SESSION_CONSUMER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Kafka connection
    bootstrap_servers: str = Field(
        ...,
        description="Kafka bootstrap servers. Set via OMNIBASE_INFRA_SESSION_CONSUMER_BOOTSTRAP_SERVERS env var.",
    )
    group_id: str = Field(
        default="omnibase-infra-session-consumer",
        description="Consumer group ID",
    )

    # Topics to subscribe
    topics: list[str] = Field(
        default_factory=list,
        description="Kafka topics to consume. Must be explicitly configured via environment or discovery.",
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

    max_poll_records: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum records per poll",
    )

    # Processing
    batch_timeout_ms: int = Field(
        default=5000,
        ge=100,
        le=60000,
        description="Timeout for batch processing in milliseconds",
    )

    # Circuit breaker
    circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Failures before circuit opens",
    )
    circuit_breaker_timeout_seconds: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Time before circuit half-opens",
    )
    circuit_breaker_half_open_successes: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of successful requests required to close circuit from half-open state",
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
        """Ensure topics are explicitly configured.

        Fails fast if no topics provided, preventing silent misconfiguration.

        Returns:
            Self if validation passes.

        Raises:
            ProtocolConfigurationError: If no topics are configured.
        """
        if not self.topics:
            from omnibase_infra.errors import ProtocolConfigurationError

            raise ProtocolConfigurationError(
                "No topics configured for session consumer. "
                "Provide explicit 'topics' via configuration or environment variable."
            )
        return self

    @model_validator(mode="after")
    def validate_timing_relationships(self) -> Self:
        """Validate timing relationships between configuration values.

        Warns if circuit breaker timeout is very short relative to batch processing,
        which could cause premature circuit opens during normal batch operations.

        Returns:
            Self if validation passes.
        """
        batch_timeout_seconds = self.batch_timeout_ms / 1000
        min_recommended_circuit_timeout = batch_timeout_seconds * 2

        if self.circuit_breaker_timeout_seconds < min_recommended_circuit_timeout:
            logger.warning(
                "Circuit breaker timeout (%ds) is less than 2x batch timeout (%.1fs). "
                "This may cause premature circuit opens during normal batch processing. "
                "Recommended minimum: %ds",
                self.circuit_breaker_timeout_seconds,
                batch_timeout_seconds,
                int(min_recommended_circuit_timeout),
            )
        return self
