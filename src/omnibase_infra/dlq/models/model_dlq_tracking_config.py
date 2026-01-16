# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""DLQ Tracking Configuration Model.

This module provides the Pydantic configuration model for the PostgreSQL-based
DLQ replay tracking service, including connection pooling and table settings.

Security Note:
    The dsn field may contain credentials. Use environment variables for
    sensitive values and ensure connection strings are not logged.

Environment Variables:
    ONEX_DLQ_POOL_MIN_SIZE: Minimum pool connections (default: 1, range: 1-100)
    ONEX_DLQ_POOL_MAX_SIZE: Maximum pool connections (default: 5, range: 1-100)
    ONEX_DLQ_COMMAND_TIMEOUT: Command timeout in seconds (default: 30.0, range: 1.0-300.0)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from omnibase_infra.dlq.constants_dlq import PATTERN_TABLE_NAME
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.utils import validate_pool_sizes_constraint
from omnibase_infra.utils.util_env_parsing import parse_env_float, parse_env_int

# Module-level defaults from environment variables
# These allow runtime configuration without code changes
# Invalid type values raise ProtocolConfigurationError
# Out-of-range values log a warning and use default (soft validation pattern)

_DEFAULT_POOL_MIN_SIZE = parse_env_int(
    "ONEX_DLQ_POOL_MIN_SIZE",
    1,
    transport_type=EnumInfraTransportType.DATABASE,
    service_name="dlq_tracking_service",
    min_value=1,  # Minimum 1 connection
    max_value=100,  # Maximum pool size
)
_DEFAULT_POOL_MAX_SIZE = parse_env_int(
    "ONEX_DLQ_POOL_MAX_SIZE",
    5,
    transport_type=EnumInfraTransportType.DATABASE,
    service_name="dlq_tracking_service",
    min_value=1,  # Minimum 1 connection
    max_value=100,  # Maximum pool size
)
_DEFAULT_COMMAND_TIMEOUT = parse_env_float(
    "ONEX_DLQ_COMMAND_TIMEOUT",
    30.0,
    transport_type=EnumInfraTransportType.DATABASE,
    service_name="dlq_tracking_service",
    min_value=1.0,  # Minimum 1 second
    max_value=300.0,  # Maximum 5 minutes
)


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
        default=_DEFAULT_POOL_MIN_SIZE,
        description=(
            "Minimum number of connections in the pool (env: ONEX_DLQ_POOL_MIN_SIZE)"
        ),
        ge=1,
        le=100,
    )
    pool_max_size: int = Field(
        default=_DEFAULT_POOL_MAX_SIZE,
        description=(
            "Maximum number of connections in the pool (env: ONEX_DLQ_POOL_MAX_SIZE)"
        ),
        ge=1,
        le=100,
    )
    command_timeout: float = Field(
        default=_DEFAULT_COMMAND_TIMEOUT,
        description=(
            "Timeout for database commands in seconds (env: ONEX_DLQ_COMMAND_TIMEOUT)"
        ),
        ge=1.0,
        le=300.0,
    )

    @field_validator("pool_max_size", mode="after")
    @classmethod
    def validate_pool_sizes(cls, v: int, info: ValidationInfo) -> int:
        """Validate that pool_max_size >= pool_min_size.

        Delegates to shared utility for consistent validation across all config models.
        """
        if info.data:
            pool_min_size = info.data.get("pool_min_size", 1)
            return validate_pool_sizes_constraint(
                v, pool_min_size, target_name="dlq_tracking_service"
            )
        return v


__all__: list[str] = ["ModelDlqTrackingConfig"]
