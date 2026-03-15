# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Configuration for the RetryWorker service.

Loads from environment variables with OMNIBASE_INFRA_RETRY_WORKER_ prefix.

The RetryWorker polls the delivery_attempts table for failed notifications
and re-invokes delivery with exponential backoff. Configuration controls
polling interval, backoff parameters, and circuit breaker thresholds.

Related Tickets:
    - OMN-1454: Implement RetryWorker for subscription notification delivery
"""

from __future__ import annotations

import logging

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class ConfigRetryWorker(BaseSettings):
    """Configuration for the subscription notification retry worker.

    Environment variables use the OMNIBASE_INFRA_RETRY_WORKER_ prefix.
    Example: OMNIBASE_INFRA_RETRY_WORKER_POLL_INTERVAL_SECONDS=30

    Attributes:
        postgres_dsn: PostgreSQL connection string.
        poll_interval_seconds: Seconds between polling cycles.
        batch_size: Maximum pending retries to fetch per poll.
        max_retry_attempts: Default max retries before moving to DLQ.
        backoff_base_seconds: Base delay for exponential backoff.
        backoff_max_seconds: Maximum delay cap for exponential backoff.
        backoff_multiplier: Multiplier for exponential backoff calculation.
        delivery_timeout_seconds: Timeout for individual delivery attempts.
        circuit_breaker_threshold: Failures before circuit opens.
        circuit_breaker_reset_timeout: Seconds before circuit half-opens.
        circuit_breaker_half_open_successes: Successes to close from half-open.
        query_timeout_seconds: Timeout for database queries.

    Example:
        >>> config = ConfigRetryWorker(
        ...     postgres_dsn="postgresql://postgres:secret@localhost:5432/omnibase_infra",
        ...     poll_interval_seconds=30,
        ...     batch_size=50,
        ... )
    """

    model_config = SettingsConfigDict(
        env_prefix="OMNIBASE_INFRA_RETRY_WORKER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # PostgreSQL connection
    postgres_dsn: str = Field(
        description=(
            "PostgreSQL connection string. Set via "
            "OMNIBASE_INFRA_RETRY_WORKER_POSTGRES_DSN env var."
        ),
    )

    # Polling configuration
    poll_interval_seconds: int = Field(
        default=30,
        ge=5,
        le=3600,
        description=(
            "Seconds between polling cycles. Lower values reduce delivery latency "
            "but increase database load. Default is 30 seconds."
        ),
    )

    batch_size: int = Field(
        default=50,
        ge=1,
        le=1000,
        description=(
            "Maximum number of pending retries to fetch per poll cycle. "
            "Limits memory usage and processing time per cycle. Default is 50."
        ),
    )

    # Retry policy
    max_retry_attempts: int = Field(
        default=5,
        ge=1,
        le=20,
        description=(
            "Default maximum retry attempts before moving to DLQ. "
            "Individual delivery attempts may override this. Default is 5."
        ),
    )

    # Exponential backoff
    backoff_base_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=3600.0,
        description=(
            "Base delay in seconds for exponential backoff calculation. "
            "Actual delay = base * (multiplier ^ attempt_count). Default is 60s."
        ),
    )

    backoff_max_seconds: float = Field(
        default=3600.0,
        ge=60.0,
        le=86400.0,
        description=(
            "Maximum delay cap for exponential backoff. Prevents unbounded "
            "backoff growth. Default is 3600s (1 hour)."
        ),
    )

    backoff_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description=(
            "Multiplier for exponential backoff. "
            "delay = base * (multiplier ^ attempt_count). Default is 2.0."
        ),
    )

    # Delivery
    delivery_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Timeout for individual delivery attempts. Default is 30s.",
    )

    # Circuit breaker
    circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Consecutive failures before circuit opens.",
    )

    circuit_breaker_reset_timeout: float = Field(
        default=60.0,
        ge=1.0,
        le=3600.0,
        description="Seconds before circuit half-opens for retry.",
    )

    circuit_breaker_half_open_successes: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Successful requests required to close circuit from half-open.",
    )

    # Query timeout
    query_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Timeout for database queries. Default is 30s.",
    )


__all__ = ["ConfigRetryWorker"]
