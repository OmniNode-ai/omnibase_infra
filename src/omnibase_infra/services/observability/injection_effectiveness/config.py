# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration for injection effectiveness observability consumer.

Loads from environment variables with OMNIBASE_INFRA_INJECTION_EFFECTIVENESS_ prefix.

Created as part of OMN-1890 for injection metrics persistence.
"""

from __future__ import annotations

import logging
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

logger = logging.getLogger(__name__)


class ConfigInjectionEffectivenessConsumer(BaseSettings):
    """Configuration for the injection effectiveness Kafka consumer.

    Environment variables use the OMNIBASE_INFRA_INJECTION_EFFECTIVENESS_ prefix.
    Example: OMNIBASE_INFRA_INJECTION_EFFECTIVENESS_KAFKA_BOOTSTRAP_SERVERS=kafka.example.com:9092

    This consumer subscribes to injection effectiveness topics and
    persists events to PostgreSQL for A/B testing analytics.
    """

    model_config = SettingsConfigDict(
        env_prefix="OMNIBASE_INFRA_INJECTION_EFFECTIVENESS_",
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
            "OMNIBASE_INFRA_INJECTION_EFFECTIVENESS_KAFKA_BOOTSTRAP_SERVERS env var."
        ),
    )
    kafka_group_id: str = Field(
        default="injection-effectiveness-postgres",
        description="Consumer group ID for offset tracking",
    )

    # Topics to subscribe (3 injection effectiveness topics from OMN-1889)
    topics: list[str] = Field(
        default_factory=lambda: [
            "onex.evt.omniclaude.context-utilization.v1",
            "onex.evt.omniclaude.agent-match.v1",
            "onex.evt.omniclaude.latency-breakdown.v1",
        ],
        description="Kafka topics to consume for injection effectiveness",
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
            "OMNIBASE_INFRA_INJECTION_EFFECTIVENESS_POSTGRES_DSN env var."
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
            "Additional buffer time added to batch_timeout_ms for asyncio.wait_for."
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

    # Minimum support gating for pattern confidence (R3 requirement)
    min_pattern_support: int = Field(
        default=20,
        ge=1,
        le=1000,
        description=(
            "Minimum number of sessions required before pattern utilization "
            "metrics are considered statistically reliable (N=20 default)."
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
        default="0.0.0.0",  # noqa: S104 - Configurable for container access
        description="Host/IP for health check server binding.",
    )
    health_check_staleness_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Maximum age for last successful write before DEGRADED status.",
    )
    health_check_poll_staleness_seconds: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Maximum age for last poll before DEGRADED status.",
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
            # Auto-generate correlation_id for configuration errors
            # (no request context available during model validation)
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="validate_topic_configuration",
                target_name="ConfigInjectionEffectivenessConsumer",
            )
            raise ProtocolConfigurationError(
                "No topics configured for injection effectiveness consumer.",
                context=context,
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
                "Circuit breaker timeout (%.1fs) is less than 2x batch timeout (%.1fs).",
                self.circuit_breaker_reset_timeout,
                batch_timeout_seconds,
            )
        return self
