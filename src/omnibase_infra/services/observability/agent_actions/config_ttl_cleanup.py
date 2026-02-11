# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration for observability table TTL cleanup.

Loads from environment variables with OMNIBASE_INFRA_TTL_CLEANUP_ prefix.

The TTL cleanup service periodically deletes rows older than the configured
retention period from observability tables. Each table has a designated
TTL column (created_at or updated_at) that determines row age.

Related Tickets:
    - OMN-1759: Implement 30-day TTL cleanup for observability tables
    - OMN-1743: Created the observability tables (Phase 1)
"""

from __future__ import annotations

import logging
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# Table-to-TTL-column mapping: which column determines row age for each table.
# Most tables use created_at; agent_execution_logs uses updated_at to avoid
# deleting long-running executions mid-flight.
DEFAULT_TABLE_TTL_COLUMNS: dict[str, str] = {
    "agent_actions": "created_at",
    "agent_routing_decisions": "created_at",
    "agent_transformation_events": "created_at",
    "router_performance_metrics": "created_at",
    "agent_detection_failures": "created_at",
    "agent_execution_logs": "updated_at",
    "agent_status_events": "created_at",
}


class ConfigTTLCleanup(BaseSettings):
    """Configuration for the observability table TTL cleanup service.

    Environment variables use the OMNIBASE_INFRA_TTL_CLEANUP_ prefix.
    Example: OMNIBASE_INFRA_TTL_CLEANUP_RETENTION_DAYS=30

    The cleanup service runs on a configurable interval, deleting rows older
    than the retention period in batches to avoid lock contention.

    Attributes:
        postgres_dsn: PostgreSQL connection string.
        retention_days: Number of days to retain data (default: 30).
        batch_size: Maximum rows to delete per batch per table (default: 1000).
        interval_seconds: Seconds between cleanup runs (default: 600 = 10 minutes).
        table_ttl_columns: Mapping of table names to their TTL timestamp column.
        circuit_breaker_threshold: Failures before circuit opens.
        circuit_breaker_reset_timeout: Seconds before circuit half-opens for retry.
        circuit_breaker_half_open_successes: Successes required to close from half-open.
        query_timeout_seconds: Timeout for individual DELETE queries.

    Example:
        >>> config = ConfigTTLCleanup(
        ...     postgres_dsn="postgresql://postgres:secret@localhost:5432/omnibase_infra",
        ...     retention_days=30,
        ...     batch_size=1000,
        ...     interval_seconds=600,
        ... )
    """

    model_config = SettingsConfigDict(
        env_prefix="OMNIBASE_INFRA_TTL_CLEANUP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # PostgreSQL connection
    postgres_dsn: str = Field(
        description=(
            "PostgreSQL connection string. Set via "
            "OMNIBASE_INFRA_TTL_CLEANUP_POSTGRES_DSN env var."
        ),
    )

    # Retention policy
    retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description=(
            "Number of days to retain observability data. Rows older than this "
            "are eligible for deletion. Default is 30 days."
        ),
    )

    # Batch processing
    batch_size: int = Field(
        default=1000,
        ge=100,
        le=50000,
        description=(
            "Maximum rows to delete per batch per table. Smaller batches reduce "
            "lock contention but require more iterations. Default is 1000."
        ),
    )

    # Scheduling
    interval_seconds: int = Field(
        default=600,
        ge=60,
        le=86400,
        description=(
            "Seconds between cleanup runs. Default is 600 (10 minutes). "
            "The ticket recommends 5-15 minutes."
        ),
    )

    # Table configuration
    table_ttl_columns: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_TABLE_TTL_COLUMNS),
        description=(
            "Mapping of table name to TTL timestamp column. "
            "Most tables use created_at; agent_execution_logs uses updated_at."
        ),
    )

    # Circuit breaker
    circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Failures before circuit opens.",
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
        description="Successful requests required to close circuit from half-open state.",
    )

    # Query timeout
    query_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description=(
            "Timeout in seconds for individual DELETE queries. "
            "Should be generous enough for large batch deletes."
        ),
    )

    @model_validator(mode="after")
    def validate_table_configuration(self) -> Self:
        """Ensure at least one table is configured for cleanup.

        Returns:
            Self if validation passes.

        Raises:
            ProtocolConfigurationError: If no tables are configured.
        """
        if not self.table_ttl_columns:
            from omnibase_infra.errors import ProtocolConfigurationError

            raise ProtocolConfigurationError(
                "No tables configured for TTL cleanup. "
                "Provide 'table_ttl_columns' via configuration or environment variable."
            )
        return self

    @model_validator(mode="after")
    def validate_ttl_column_names(self) -> Self:
        """Ensure TTL column names are valid.

        Only created_at and updated_at are valid TTL columns.

        Returns:
            Self if validation passes.

        Raises:
            ProtocolConfigurationError: If invalid TTL column names detected.
        """
        valid_columns = {"created_at", "updated_at"}
        invalid = {
            table: col
            for table, col in self.table_ttl_columns.items()
            if col not in valid_columns
        }
        if invalid:
            from omnibase_infra.errors import ProtocolConfigurationError

            raise ProtocolConfigurationError(
                f"Invalid TTL column names: {invalid}. Valid columns: {valid_columns}"
            )
        return self


__all__ = ["ConfigTTLCleanup", "DEFAULT_TABLE_TTL_COLUMNS"]
