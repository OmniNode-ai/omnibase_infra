# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration for LLM cost aggregation consumer.

Loads from environment variables with OMNIBASE_INFRA_LLM_COST_ prefix.

Related Tickets:
    - OMN-2240: E1-T4 LLM cost aggregation service
"""

from __future__ import annotations

import logging
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class ConfigLlmCostAggregation(BaseSettings):
    """Configuration for the LLM cost aggregation Kafka consumer.

    Environment variables use the OMNIBASE_INFRA_LLM_COST_ prefix.
    Example: OMNIBASE_INFRA_LLM_COST_KAFKA_BOOTSTRAP_SERVERS=kafka.example.com:9092

    This consumer subscribes to the LLM call completed topic and
    aggregates costs into the llm_cost_aggregates table.
    """

    model_config = SettingsConfigDict(
        env_prefix="OMNIBASE_INFRA_LLM_COST_",
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
            "OMNIBASE_INFRA_LLM_COST_KAFKA_BOOTSTRAP_SERVERS env var for production."
        ),
    )
    kafka_group_id: str = Field(
        default="llm-cost-aggregation-postgres",
        description="Consumer group ID for offset tracking",
    )

    # Topics to subscribe
    topics: list[str] = Field(
        default_factory=lambda: [
            "onex.evt.omniintelligence.llm-call-completed.v1",
        ],
        description="Kafka topics to consume for LLM cost aggregation",
    )

    # Consumer behavior
    auto_offset_reset: Literal["earliest", "latest", "none"] = Field(
        default="earliest",
        description=(
            "Where to start consuming if no offset exists. "
            "Valid values: 'earliest', 'latest', 'none'."
        ),
    )

    # PostgreSQL connection
    postgres_dsn: str = Field(
        description=(
            "PostgreSQL connection string. Set via "
            "OMNIBASE_INFRA_LLM_COST_POSTGRES_DSN env var."
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
            "the asyncio.wait_for timeout when polling Kafka."
        ),
    )

    # Connection pool
    pool_min_size: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Minimum PostgreSQL pool connections",
    )
    pool_max_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum PostgreSQL pool connections",
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
        default=8089,
        ge=1024,
        le=65535,
        description="Port for HTTP health check endpoint",
    )
    health_check_host: str = Field(
        default="0.0.0.0",  # noqa: S104 - Configurable, see security note below
        description=(
            "Host/IP for health check server binding. Default '0.0.0.0' binds to all "
            "interfaces for container/Kubernetes probe access."
        ),
    )
    health_check_staleness_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description=(
            "Maximum age in seconds for the last successful write before "
            "the health check reports DEGRADED status."
        ),
    )
    health_check_poll_staleness_seconds: int = Field(
        default=60,
        ge=10,
        le=300,
        description=(
            "Maximum age in seconds for the last poll before the health check "
            "reports DEGRADED status."
        ),
    )
    startup_grace_period_seconds: float = Field(
        default=60.0,
        ge=0.0,
        le=600.0,
        description=(
            "Grace period in seconds after startup during which the consumer "
            "is considered healthy even without writes."
        ),
    )

    @model_validator(mode="after")
    def validate_topic_configuration(self) -> Self:
        """Ensure topics are configured.

        Returns:
            Self if validation passes.

        Raises:
            ProtocolConfigurationError: If no topics are configured.
        """
        if not self.topics:
            from omnibase_infra.errors import ProtocolConfigurationError

            raise ProtocolConfigurationError(
                "No topics configured for LLM cost aggregation consumer. "
                "Provide explicit 'topics' via configuration or environment variable."
            )
        return self

    @model_validator(mode="after")
    def validate_pool_size_relationship(self) -> Self:
        """Ensure pool_min_size does not exceed pool_max_size.

        asyncpg.create_pool() raises ValueError at runtime when min_size > max_size.
        This validator catches the misconfiguration eagerly at config load time.

        Returns:
            Self if validation passes.

        Raises:
            ProtocolConfigurationError: If pool_min_size > pool_max_size.
        """
        if self.pool_min_size > self.pool_max_size:
            from omnibase_infra.errors import ProtocolConfigurationError

            raise ProtocolConfigurationError(
                f"pool_min_size ({self.pool_min_size}) must not exceed "
                f"pool_max_size ({self.pool_max_size}). "
                "Adjust pool_min_size or pool_max_size so that min <= max."
            )
        return self

    @model_validator(mode="after")
    def validate_timing_relationships(self) -> Self:
        """Validate timing relationships between configuration values.

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
