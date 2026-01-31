"""Configuration for agent actions observability consumer.

Loads from environment variables with OMNIBASE_INFRA_AGENT_ACTIONS_ prefix.

Moved from omniclaude as part of OMN-1743 layer-correction cleanup.
"""

from __future__ import annotations

import logging
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class ConfigAgentActionsConsumer(BaseSettings):
    """Configuration for the agent actions observability Kafka consumer.

    Environment variables use the OMNIBASE_INFRA_AGENT_ACTIONS_ prefix.
    Example: OMNIBASE_INFRA_AGENT_ACTIONS_KAFKA_BOOTSTRAP_SERVERS=kafka.example.com:9092

    This consumer subscribes to multiple agent observability topics and
    persists events to PostgreSQL for analytics and debugging.
    """

    model_config = SettingsConfigDict(
        env_prefix="OMNIBASE_INFRA_AGENT_ACTIONS_",
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
            "OMNIBASE_INFRA_AGENT_ACTIONS_KAFKA_BOOTSTRAP_SERVERS env var for production."
        ),
    )
    kafka_group_id: str = Field(
        default="agent-observability-postgres",
        description="Consumer group ID for offset tracking",
    )

    # Topics to subscribe (6 observability topics)
    topics: list[str] = Field(
        default_factory=lambda: [
            "agent-actions",
            "agent-routing-decisions",
            "agent-transformation-events",
            "router-performance-metrics",
            "agent-detection-failures",
            "agent-execution-logs",
        ],
        description="Kafka topics to consume for agent observability",
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
            "OMNIBASE_INFRA_AGENT_ACTIONS_POSTGRES_DSN env var."
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

    # Health check
    health_check_port: int = Field(
        default=8087,
        ge=1024,
        le=65535,
        description="Port for HTTP health check endpoint",
    )
    health_check_host: str = Field(
        default="0.0.0.0",  # noqa: S104 - Configurable, see security note below
        description=(
            "Host/IP for health check server binding. Default '0.0.0.0' binds to all "
            "interfaces for container/Kubernetes probe access. For security-sensitive "
            "deployments, set to '127.0.0.1' to restrict to localhost-only access. "
            "Configure via OMNIBASE_INFRA_AGENT_ACTIONS_HEALTH_CHECK_HOST env var."
        ),
    )
    health_check_staleness_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description=(
            "Maximum age in seconds for the last successful write before "
            "the health check reports DEGRADED status. Lower values detect "
            "stalled consumers faster but may cause false positives in "
            "low-traffic environments. Default is 300 (5 minutes). "
            "Configure via OMNIBASE_INFRA_AGENT_ACTIONS_HEALTH_CHECK_STALENESS_SECONDS env var."
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
                "No topics configured for agent actions consumer. "
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

        if self.circuit_breaker_reset_timeout < min_recommended_circuit_timeout:
            logger.warning(
                "Circuit breaker timeout (%.1fs) is less than 2x batch timeout (%.1fs). "
                "This may cause premature circuit opens during normal batch processing. "
                "Recommended minimum: %.1fs",
                self.circuit_breaker_reset_timeout,
                batch_timeout_seconds,
                min_recommended_circuit_timeout,
            )
        return self
