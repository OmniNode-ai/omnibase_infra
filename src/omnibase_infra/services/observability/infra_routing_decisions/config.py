# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Configuration for infra routing decisions consumer (OMN-8692).

Loads from environment variables with OMNIBASE_INFRA_ROUTING_DECISIONS_ prefix.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_TOPIC_ROUTING_DECIDED: str = "onex.evt.omnibase-infra.routing-decided.v1"
_TOPIC_DLQ: str = "onex.evt.omnibase-infra.routing-decided-dlq.v1"


class ConfigInfraRoutingDecisionsConsumer(BaseSettings):
    """Configuration for the infra routing decisions Kafka consumer.

    Environment variables use the OMNIBASE_INFRA_ROUTING_DECISIONS_ prefix.
    Example: OMNIBASE_INFRA_ROUTING_DECISIONS_KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    """

    model_config = SettingsConfigDict(
        env_prefix="OMNIBASE_INFRA_ROUTING_DECISIONS_",
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
        default="infra-routing-decisions-postgres",
        description="Consumer group ID for offset tracking",
    )
    topics: list[str] = Field(
        default_factory=lambda: [_TOPIC_ROUTING_DECIDED],
        description="Kafka topics to consume",
    )
    auto_offset_reset: str = Field(
        default="earliest",
        description="Where to start consuming if no offset exists",
    )
    enable_auto_commit: bool = Field(
        default=False,
        description="Disable auto-commit for at-least-once delivery",
    )
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
    postgres_dsn: str = Field(
        description="PostgreSQL connection string.",
    )
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
        description="Additional buffer time for asyncio.wait_for polling timeout.",
    )
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
    dlq_topic: str = Field(
        default=_TOPIC_DLQ,
        description="Dead letter topic for permanently failed messages.",
    )
    dlq_enabled: bool = Field(
        default=True,
        description="Enable dead letter queue for permanently failed messages.",
    )
    max_retry_count: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retries before sending a message to the DLQ.",
    )
    health_check_port: int = Field(
        default=8097,
        ge=1024,
        le=65535,
        description="Port for HTTP health check endpoint",
    )
    health_check_host: str = Field(
        default="127.0.0.1",  # fallback-ok: override to 0.0.0.0 in containers via env
        description="Host for health check server binding.",
    )
    health_check_staleness_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Max age in seconds for last successful write before DEGRADED.",
    )
    health_check_poll_staleness_seconds: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Max age in seconds for last poll before DEGRADED.",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level for the consumer process (DEBUG, INFO, WARNING, ERROR).",
    )

    @model_validator(mode="after")
    def validate_session_timeout_ratio(self) -> Self:
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
