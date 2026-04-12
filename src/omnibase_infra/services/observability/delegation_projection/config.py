# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Configuration for delegation projection consumer.

Loads from environment variables with OMNIBASE_INFRA_DELEGATION_ prefix.

Related Tickets:
    - OMN-8532: Add delegation projection consumer service
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal, Self
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class ConfigDelegationProjection(BaseSettings):
    """Configuration for the delegation projection Kafka consumer.

    Environment variables use the OMNIBASE_INFRA_DELEGATION_ prefix.
    Example: OMNIBASE_INFRA_DELEGATION_KAFKA_BOOTSTRAP_SERVERS=redpanda:9092
    """

    model_config = SettingsConfigDict(
        env_prefix="OMNIBASE_INFRA_DELEGATION_",
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
            "OMNIBASE_INFRA_DELEGATION_KAFKA_BOOTSTRAP_SERVERS env var."
        ),
    )
    kafka_group_id: str = Field(
        default="delegation-projection-postgres",
        description="Consumer group ID for offset tracking",
    )

    topics: list[str] = Field(
        default_factory=lambda: [
            "onex.evt.omniclaude.task-delegated.v1",
        ],
        description="Kafka topics to consume for delegation projection",
    )

    auto_offset_reset: Literal["earliest", "latest", "none"] = Field(
        default="earliest",
        description="Where to start consuming if no offset exists.",
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
        description="Kafka heartbeat interval in ms. Should be ~1/3 of session_timeout_ms.",
    )
    max_poll_interval_ms: int = Field(
        default=300000,
        ge=10000,
        le=600000,
        description="Max time between poll() calls in ms before consumer eviction.",
    )

    # PostgreSQL connection
    postgres_dsn: str = Field(
        ...,
        repr=False,
        exclude=True,
        description="PostgreSQL connection string. Excluded from serialization.",
    )

    @field_validator("postgres_dsn")
    @classmethod
    def validate_postgres_dsn_scheme(cls, v: str) -> str:
        if not v.startswith(("postgresql://", "postgres://")):
            try:
                parsed = urlparse(v)
                safe_prefix = (
                    f"{parsed.scheme}://..." if parsed.scheme else repr(v[:10])
                )
            except Exception:  # noqa: BLE001
                safe_prefix = repr(v[:10])
            raise ValueError(
                f"postgres_dsn must start with 'postgresql://' or 'postgres://', "
                f"got: {safe_prefix}."
            )
        return v

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
        ge=2.0,
        le=30.0,
        description="Additional buffer seconds for asyncio.wait_for timeout.",
    )

    # Connection pool
    pool_min_size: int = Field(default=2, ge=1, le=20)
    pool_max_size: int = Field(default=10, ge=1, le=50)

    # Circuit breaker
    circuit_breaker_threshold: int = Field(default=5, ge=1, le=100)
    circuit_breaker_reset_timeout: float = Field(default=60.0, ge=1.0, le=3600.0)
    circuit_breaker_half_open_successes: int = Field(default=1, ge=1, le=10)

    # Health check
    health_check_port: int = Field(
        default=8099,
        ge=1024,
        le=65535,
        description="Port for HTTP health check endpoint",
    )
    health_check_host: str = Field(
        default="127.0.0.1",
        description="Host for health check server binding.",
    )
    health_check_staleness_seconds: int = Field(default=300, ge=60, le=3600)
    health_check_poll_staleness_seconds: int = Field(default=60, ge=10, le=300)
    startup_grace_period_seconds: float = Field(default=60.0, ge=0.0, le=600.0)

    @model_validator(mode="before")
    @classmethod
    # ONEX_EXCLUDE: any_type - dict[str, Any] required for pydantic mode="before" validator
    def warn_unrecognized_env_vars(cls, data: dict[str, Any]) -> dict[str, Any]:
        prefix = "OMNIBASE_INFRA_DELEGATION_"
        known_fields = set(cls.model_fields.keys())
        for env_key in os.environ:
            if not env_key.upper().startswith(prefix):
                continue
            field_name = env_key[len(prefix) :].lower()
            if field_name not in known_fields:
                logger.warning(
                    "Unrecognized environment variable '%s' has prefix '%s' "
                    "but does not match any configuration field.",
                    env_key,
                    prefix,
                )
        return data

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

    @model_validator(mode="after")
    def validate_pool_size_relationship(self) -> Self:
        if self.pool_min_size > self.pool_max_size:
            from omnibase_infra.errors import ProtocolConfigurationError

            raise ProtocolConfigurationError(
                f"pool_min_size ({self.pool_min_size}) must not exceed "
                f"pool_max_size ({self.pool_max_size})."
            )
        return self

    @model_validator(mode="after")
    def validate_topics_nonempty(self) -> Self:
        if not self.topics:
            from omnibase_infra.errors import ProtocolConfigurationError

            raise ProtocolConfigurationError(
                "No topics configured for delegation projection consumer."
            )
        return self
