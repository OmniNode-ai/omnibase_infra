# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""DLQ Tracking Configuration Model.

This module provides the Pydantic configuration model for the PostgreSQL-based
DLQ replay tracking service, including connection pooling and table settings.

Security Note:
    The dsn field may contain credentials. Use environment variables for
    sensitive values and ensure connection strings are not logged.
"""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from omnibase_infra.dlq.constants_dlq import PATTERN_TABLE_NAME
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError


class ModelDlqTrackingConfig(BaseModel):
    """Configuration for PostgreSQL-based DLQ replay tracking service.

    This model defines all configuration options for the DLQ tracking
    service, including connection settings and pooling parameters.

    Security Policy:
        - DSN may contain credentials - use environment variables
        - Never log the full DSN value
        - Use SSL in production environments

    Attributes:
        dsn: PostgreSQL connection string (e.g., "postgresql://user:pass@host:5432/db").
            Should be provided via environment variable for security.
        storage_table: PostgreSQL table for storing DLQ replay history records.
            Default: "dlq_replay_history".
        pool_min_size: Minimum number of connections in the pool.
            Default: 1.
        pool_max_size: Maximum number of connections in the pool.
            Default: 5.
        command_timeout: Timeout for database commands in seconds.
            Default: 30.0.

    Example:
        >>> config = ModelDlqTrackingConfig(
        ...     dsn="postgresql://user:pass@localhost:5432/mydb",
        ...     storage_table="dlq_replay_history",
        ...     pool_max_size=10,
        ... )
        >>> print(config.storage_table)
        dlq_replay_history
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

    # Defense-in-depth: Table name validation is applied at both config and runtime level.
    # See constants_dlq.py for details on why both validations are intentional.
    storage_table: str = Field(
        default="dlq_replay_history",
        description="PostgreSQL table for storing DLQ replay history records",
        min_length=1,
        max_length=63,  # PostgreSQL identifier limit
        pattern=PATTERN_TABLE_NAME,
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
                    target_name="dlq_tracking_service",
                    correlation_id=uuid4(),
                )
                raise ProtocolConfigurationError(
                    f"pool_max_size ({v}) must be >= pool_min_size ({pool_min_size})",
                    context=context,
                    parameter="pool_max_size",
                    value=v,
                )
        return v


__all__: list[str] = ["ModelDlqTrackingConfig"]
