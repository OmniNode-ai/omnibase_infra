# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL Idempotency Store Configuration Model.

This module provides the Pydantic configuration model for the PostgreSQL-based
idempotency store, including connection pooling, TTL, and cleanup settings.

Security Note:
    The dsn field may contain credentials. Use environment variables for
    sensitive values and ensure connection strings are not logged.

Environment Variables:
    ONEX_IDEMPOTENCY_TTL_SECONDS: TTL for idempotency records (default: 86400)
    ONEX_IDEMPOTENCY_CLEANUP_INTERVAL: Cleanup interval in seconds (default: 3600)
    ONEX_IDEMPOTENCY_BATCH_SIZE: Records per cleanup batch (default: 10000)
"""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.utils.util_env_parsing import parse_env_int

# Module-level defaults from environment variables
# These allow runtime configuration without code changes

_DEFAULT_TTL_SECONDS = parse_env_int(
    "ONEX_IDEMPOTENCY_TTL_SECONDS",
    86400,
    transport_type=EnumInfraTransportType.DATABASE,
    service_name="postgres_idempotency_store",
)
_DEFAULT_CLEANUP_INTERVAL = parse_env_int(
    "ONEX_IDEMPOTENCY_CLEANUP_INTERVAL",
    3600,
    transport_type=EnumInfraTransportType.DATABASE,
    service_name="postgres_idempotency_store",
)
_DEFAULT_BATCH_SIZE = parse_env_int(
    "ONEX_IDEMPOTENCY_BATCH_SIZE",
    10000,
    transport_type=EnumInfraTransportType.DATABASE,
    service_name="postgres_idempotency_store",
)


class ModelPostgresIdempotencyStoreConfig(BaseModel):
    """Configuration for PostgreSQL-based idempotency store.

    This model defines all configuration options for the PostgreSQL
    idempotency store, including connection settings, pooling parameters,
    and TTL-based cleanup configuration.

    Security Policy:
        - DSN may contain credentials - use environment variables
        - Never log the full DSN value
        - Use SSL in production environments

    Attributes:
        dsn: PostgreSQL connection string (e.g., "postgresql://user:pass@host:5432/db").
            Should be provided via environment variable for security.
        table_name: Name of the idempotency records table.
            Default: "idempotency_records".
        pool_min_size: Minimum number of connections in the pool.
            Default: 1.
        pool_max_size: Maximum number of connections in the pool.
            Default: 5.
        command_timeout: Timeout for database commands in seconds.
            Default: 30.0.
        ttl_seconds: Time-to-live for idempotency records in seconds.
            Records older than this will be cleaned up. Default: 86400 (24 hours).
        auto_cleanup: Whether to automatically clean up expired records.
            Default: True.
        cleanup_interval_seconds: Interval between cleanup runs in seconds.
            Default: 3600 (1 hour).
        clock_skew_tolerance_seconds: Buffer added to TTL during cleanup to prevent
            premature deletion due to clock skew between distributed nodes.
            Default: 60 (1 minute).
        cleanup_batch_size: Number of records to delete per batch during cleanup.
            Batched deletion reduces lock contention. Default: 10000.
        cleanup_max_iterations: Maximum number of batch iterations during cleanup.
            Prevents runaway loops. Default: 100.

    Example:
        >>> config = ModelPostgresIdempotencyStoreConfig(
        ...     dsn="postgresql://user:pass@localhost:5432/mydb",
        ...     table_name="idempotency_records",
        ...     pool_max_size=10,
        ...     ttl_seconds=172800,  # 48 hours
        ... )
        >>> print(config.table_name)
        idempotency_records
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    dsn: str = Field(
        description="PostgreSQL connection string (e.g., 'postgresql://user:pass@host:5432/db')",
        min_length=1,
    )

    @field_validator("dsn", mode="before")
    @classmethod
    def validate_dsn(cls, v: object) -> str:
        """Validate PostgreSQL DSN format using robust parser.

        This validator uses urllib.parse for comprehensive DSN validation,
        handling edge cases like IPv6 addresses, URL-encoded passwords,
        and query parameters.

        Edge cases validated:
            - IPv6 addresses: postgresql://user:pass@[::1]:5432/db
            - URL-encoded passwords: user:p%40ssword@host (p@ssword)
            - Query parameters: postgresql://host/db?sslmode=require
            - Missing components: postgresql://localhost/db (no user/pass/port)

        Args:
            v: DSN value (any type before Pydantic conversion)

        Returns:
            Validated DSN string

        Raises:
            ProtocolConfigurationError: If DSN format is invalid
        """
        from omnibase_infra.utils.util_dsn_validation import parse_and_validate_dsn

        # parse_and_validate_dsn handles all validation and error context
        # It will raise ProtocolConfigurationError with proper context if invalid
        parse_and_validate_dsn(v)

        # If validation passes, return the stripped string
        return v.strip() if isinstance(v, str) else str(v)

    table_name: str = Field(
        default="idempotency_records",
        description="Name of the idempotency records table",
        min_length=1,
        max_length=63,  # PostgreSQL identifier limit
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$",
    )
    pool_min_size: int = Field(
        default=1,
        description="Minimum number of connections in the pool",
        ge=1,
        le=100,
    )
    pool_max_size: int = Field(
        default=5,
        description="Maximum number of connections in the pool",
        ge=1,
        le=100,
    )
    command_timeout: float = Field(
        default=30.0,
        description="Timeout for database commands in seconds",
        ge=1.0,
        le=300.0,
    )
    ttl_seconds: int = Field(
        default=_DEFAULT_TTL_SECONDS,
        description="Time-to-live for idempotency records in seconds (env: ONEX_IDEMPOTENCY_TTL_SECONDS)",
        ge=60,  # Minimum 1 minute
        le=2592000,  # Maximum 30 days
    )
    auto_cleanup: bool = Field(
        default=True,
        description="Whether to automatically clean up expired records",
    )
    cleanup_interval_seconds: int = Field(
        default=_DEFAULT_CLEANUP_INTERVAL,
        description="Interval between cleanup runs in seconds (env: ONEX_IDEMPOTENCY_CLEANUP_INTERVAL)",
        ge=60,  # Minimum 1 minute
        le=86400,  # Maximum 24 hours
    )
    clock_skew_tolerance_seconds: int = Field(
        default=60,  # 1 minute tolerance
        description=(
            "Buffer added to TTL during cleanup to prevent premature deletion due to "
            "clock skew between distributed nodes. In distributed systems, nodes may "
            "have slightly different system times. This tolerance ensures records are "
            "not cleaned up before all nodes consider them expired. "
            "Recommended: Use NTP synchronization in production and set this to at "
            "least the maximum expected clock drift between nodes."
        ),
        ge=0,  # Can be 0 to disable tolerance
        le=3600,  # Maximum 1 hour tolerance
    )
    cleanup_batch_size: int = Field(
        default=_DEFAULT_BATCH_SIZE,
        description=(
            "Number of records to delete per batch during cleanup (env: ONEX_IDEMPOTENCY_BATCH_SIZE). "
            "Batched deletion reduces lock contention on high-volume tables by "
            "breaking large deletes into smaller transactions, allowing other "
            "operations to interleave and preventing long-running locks."
        ),
        ge=100,  # Minimum batch size
        le=100000,  # Maximum batch size
    )
    cleanup_max_iterations: int = Field(
        default=100,
        description=(
            "Maximum number of batch iterations during cleanup. "
            "Prevents runaway cleanup loops in extreme cases. "
            "Total max records deleted = cleanup_batch_size * cleanup_max_iterations."
        ),
        ge=1,
        le=1000,
    )

    @field_validator("pool_max_size", mode="after")
    @classmethod
    def validate_pool_sizes(cls, v: int, info: ValidationInfo) -> int:
        """Validate that pool_max_size >= pool_min_size.

        Args:
            v: The pool_max_size value
            info: Pydantic validation info containing other field values

        Returns:
            Validated pool_max_size

        Raises:
            ProtocolConfigurationError: If pool_max_size < pool_min_size
        """
        # Access validated data from info
        if info.data:
            pool_min_size = info.data.get("pool_min_size", 1)
            if v < pool_min_size:
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="validate_config",
                    target_name="postgres_idempotency_store",
                    correlation_id=uuid4(),
                )
                raise ProtocolConfigurationError(
                    f"pool_max_size ({v}) must be >= pool_min_size ({pool_min_size})",
                    context=context,
                    parameter="pool_max_size",
                    value=v,
                )
        return v


__all__: list[str] = ["ModelPostgresIdempotencyStoreConfig"]
