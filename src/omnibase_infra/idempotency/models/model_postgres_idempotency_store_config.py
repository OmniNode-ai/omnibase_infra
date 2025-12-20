# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL Idempotency Store Configuration Model.

This module provides the Pydantic configuration model for the PostgreSQL-based
idempotency store, including connection pooling, TTL, and cleanup settings.

Security Note:
    The dsn field may contain credentials. Use environment variables for
    sensitive values and ensure connection strings are not logged.
"""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError


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
        """Validate PostgreSQL DSN format.

        Args:
            v: DSN value (any type before Pydantic conversion)

        Returns:
            Validated DSN string

        Raises:
            ProtocolConfigurationError: If DSN format is invalid
        """
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="validate_config",
            target_name="postgres_idempotency_store",
            correlation_id=uuid4(),
        )

        if v is None:
            raise ProtocolConfigurationError(
                "dsn cannot be None",
                context=context,
                parameter="dsn",
                value=None,
            )
        if not isinstance(v, str):
            raise ProtocolConfigurationError(
                f"dsn must be a string, got {type(v).__name__}",
                context=context,
                parameter="dsn",
                value=type(v).__name__,
            )
        if not v.strip():
            raise ProtocolConfigurationError(
                "dsn cannot be empty",
                context=context,
                parameter="dsn",
                value="",
            )

        dsn = v.strip()

        # Basic PostgreSQL DSN validation
        valid_prefixes = ("postgresql://", "postgres://", "postgresql+asyncpg://")
        if not dsn.startswith(valid_prefixes):
            raise ProtocolConfigurationError(
                f"dsn must start with one of {valid_prefixes}",
                context=context,
                parameter="dsn",
                value="[REDACTED]",  # Never log DSN contents
            )

        return dsn

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
        default=86400,  # 24 hours
        description="Time-to-live for idempotency records in seconds",
        ge=60,  # Minimum 1 minute
        le=2592000,  # Maximum 30 days
    )
    auto_cleanup: bool = Field(
        default=True,
        description="Whether to automatically clean up expired records",
    )
    cleanup_interval_seconds: int = Field(
        default=3600,  # 1 hour
        description="Interval between cleanup runs in seconds",
        ge=60,  # Minimum 1 minute
        le=86400,  # Maximum 24 hours
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
