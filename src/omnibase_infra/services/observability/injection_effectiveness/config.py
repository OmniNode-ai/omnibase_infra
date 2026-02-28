# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration for injection effectiveness observability consumer.

This module provides Pydantic Settings configuration for the injection
effectiveness Kafka consumer service. Configuration is loaded from environment
variables with the ``OMNIBASE_INFRA_INJECTION_EFFECTIVENESS_`` prefix.

Configuration Groups:
    - **Kafka**: Bootstrap servers, consumer group, topics, auto-offset reset
    - **PostgreSQL**: DSN connection string, pool sizing
    - **Batch Processing**: Batch size, timeout, poll buffer
    - **Circuit Breaker**: Threshold, reset timeout, half-open successes
    - **Health Check**: Port, host, staleness thresholds, startup grace period
    - **Pattern Analytics**: Minimum support threshold for statistical confidence

Environment Variables:
    All configuration values can be set via environment variables with the
    ``OMNIBASE_INFRA_INJECTION_EFFECTIVENESS_`` prefix. For example:

    - ``OMNIBASE_INFRA_INJECTION_EFFECTIVENESS_KAFKA_BOOTSTRAP_SERVERS``
    - ``OMNIBASE_INFRA_INJECTION_EFFECTIVENESS_POSTGRES_DSN``
    - ``OMNIBASE_INFRA_INJECTION_EFFECTIVENESS_BATCH_SIZE``
    - ``OMNIBASE_INFRA_INJECTION_EFFECTIVENESS_CIRCUIT_BREAKER_THRESHOLD``

Validation:
    The configuration validates:
    - At least one topic must be configured
    - Pool min size must be <= pool max size
    - Timing relationships (warns if circuit breaker timeout < 2x batch timeout)

Related Tickets:
    - OMN-1890: Store injection metrics with corrected schema
    - OMN-1889: Emit injection metrics from omniclaude hooks (producer)
    - OMN-2942: Add consumer for manifest injection lifecycle events (OMN-1888 audit trail)

Example:
    >>> from omnibase_infra.services.observability.injection_effectiveness.config import (
    ...     ConfigInjectionEffectivenessConsumer,
    ... )
    >>>
    >>> # Load from environment (default)
    >>> config = ConfigInjectionEffectivenessConsumer()
    >>>
    >>> # Or with explicit values
    >>> config = ConfigInjectionEffectivenessConsumer(
    ...     kafka_bootstrap_servers="kafka.example.com:9092",
    ...     postgres_dsn="postgresql://user:pass@host:5432/db",
    ...     batch_size=200,
    ... )
    >>>
    >>> print(config.topics)
    ['onex.evt.omniclaude.context-utilization.v1', ...]
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

    # Topics to subscribe (3 injection effectiveness topics from OMN-1889 +
    # 3 manifest injection lifecycle topics from OMN-2942 for OMN-1888 audit trail)
    topics: list[str] = Field(
        default_factory=lambda: [
            "onex.evt.omniclaude.context-utilization.v1",
            "onex.evt.omniclaude.agent-match.v1",
            "onex.evt.omniclaude.latency-breakdown.v1",
            # Manifest injection lifecycle topics (OMN-1888 / OMN-2942)
            "onex.evt.omniclaude.manifest-injection-started.v1",
            "onex.evt.omniclaude.manifest-injected.v1",
            "onex.evt.omniclaude.manifest-injection-failed.v1",
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

    # PostgreSQL pool settings
    pool_min_size: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Minimum number of connections in the PostgreSQL connection pool.",
    )
    pool_max_size: int = Field(
        default=10,
        ge=2,
        le=100,
        description="Maximum number of connections in the PostgreSQL connection pool.",
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
    startup_grace_period_seconds: float = Field(
        default=60.0,
        ge=10.0,
        le=300.0,
        description=(
            "Grace period in seconds after startup during which the consumer is "
            "considered healthy even without successful writes. Allows time for "
            "initial Kafka partition assignment and first message processing."
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

    @model_validator(mode="after")
    def validate_pool_size_relationship(self) -> Self:
        """Validate pool size relationship (min <= max).

        Returns:
            Self if validation passes.

        Raises:
            ProtocolConfigurationError: If pool_min_size > pool_max_size.
        """
        if self.pool_min_size > self.pool_max_size:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="validate_pool_size_relationship",
                target_name="ConfigInjectionEffectivenessConsumer",
            )
            raise ProtocolConfigurationError(
                f"pool_min_size ({self.pool_min_size}) must be <= pool_max_size "
                f"({self.pool_max_size}).",
                context=context,
            )
        return self


__all__ = ["ConfigInjectionEffectivenessConsumer"]
