# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Repository Runtime Configuration Model.

This module provides the configuration model for PostgresRepositoryRuntime.
All fields are strongly typed to eliminate Any usage and enable proper validation.

Safety Constraints:
    - max_row_limit: Prevents unbounded SELECT queries
    - timeout_ms: Prevents long-running queries from blocking resources
    - allowed_modes: Allowlist of permitted operation modes (read, write)
    - allow_write_operations: Explicit opt-in for write operations

Determinism:
    - primary_key_column: Enables ORDER BY injection for stable pagination
    - default_order_by: Default ordering clause when PK is declared

Metrics:
    - emit_metrics: Controls whether duration_ms and rows_returned are emitted

SQL Identifier Validation:
    - primary_key_column and default_order_by are validated against safe SQL
      identifier patterns to prevent SQL injection from misconfiguration.
    - Pattern: ^[a-zA-Z_][a-zA-Z0-9_]*$ for column names
    - ORDER BY allows: column [ASC|DESC] [, column [ASC|DESC]]*

Example:
    >>> config = ModelRepositoryRuntimeConfig(
    ...     max_row_limit=100,
    ...     timeout_ms=5000,
    ...     allowed_ops={"select", "insert"},
    ... )
    >>> print(config.allow_raw_operations)
    False
    >>> print("delete" in config.allowed_ops)
    False
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Safe SQL identifier pattern: starts with letter or underscore,
# followed by letters, digits, or underscores
_SQL_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# ORDER BY clause pattern: column [ASC|DESC]
# Allows whitespace around components
_ORDER_BY_COMPONENT_PATTERN = re.compile(
    r"^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*(?:ASC|DESC)?\s*$", re.IGNORECASE
)


class ModelRepositoryRuntimeConfig(BaseModel):
    """Configuration for PostgresRepositoryRuntime.

    This model controls safety constraints, allowed operations, determinism
    behavior, and metrics emission for the PostgresRepositoryRuntime.

    Attributes:
        max_row_limit: Maximum rows for multi-row selects (1-1000, default: 10).
            Prevents unbounded SELECT queries that could return massive result sets.
        timeout_ms: Query timeout in milliseconds (1000-300000, default: 30000).
            Queries exceeding this timeout are cancelled to prevent resource exhaustion.
        allowed_modes: Set of allowed operation modes (read, write).
            Both modes enabled by default. SQL safety is validated by
            omnibase_core validators at contract load time.
        allow_write_operations: Enable 'write' mode operations.
            Default: True. Set to False for read-only configurations.
        primary_key_column: Column name for ORDER BY injection.
            When set, ensures deterministic query results for pagination.
        default_order_by: Default ORDER BY clause when primary_key_column is set.
            Applied when no explicit ORDER BY is provided.
        emit_metrics: Whether to emit duration_ms and rows_returned metrics.
            Enable for observability integration. Default: True.

    Example:
        >>> from omnibase_infra.runtime.db.models import ModelRepositoryRuntimeConfig
        >>> # Restrictive config for read-only operations
        >>> readonly_config = ModelRepositoryRuntimeConfig(
        ...     allowed_modes=frozenset({"read"}),
        ...     allow_write_operations=False,
        ...     max_row_limit=50,
        ... )
        >>> # Permissive config with higher limits
        >>> admin_config = ModelRepositoryRuntimeConfig(
        ...     max_row_limit=500,
        ... )
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
    )

    # Safety constraints
    max_row_limit: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Maximum rows for multi-row selects",
    )
    timeout_ms: int = Field(
        default=30000,
        ge=1000,
        le=300000,
        description="Query timeout in milliseconds",
    )

    # Allowed operation modes (from omnibase_core contract schema)
    # Note: Specific SQL safety is validated by omnibase_core validators at contract load
    allowed_modes: frozenset[Literal["read", "write"]] = Field(
        default=frozenset({"read", "write"}),
        description="Allowed operation modes from contract (read=SELECT, write=INSERT/UPDATE/etc)",
    )

    # Feature flag for write operations (additional safety layer)
    allow_write_operations: bool = Field(
        default=True,
        description="Enable 'write' mode operations (INSERT, UPDATE, UPSERT)",
    )

    # Determinism controls
    primary_key_column: str | None = Field(
        default=None,
        description="Primary key column for ORDER BY injection to ensure deterministic results",
    )
    default_order_by: str | None = Field(
        default=None,
        description="Default ORDER BY clause when primary_key_column is declared",
    )

    # Metrics emission
    emit_metrics: bool = Field(
        default=True,
        description="Emit duration_ms and rows_returned metrics for observability",
    )

    @field_validator("primary_key_column")
    @classmethod
    def validate_primary_key_column(cls, v: str | None) -> str | None:
        """Validate primary_key_column is a safe SQL identifier.

        Prevents SQL injection from misconfigured column names.
        Pattern: ^[a-zA-Z_][a-zA-Z0-9_]*$

        Args:
            v: The column name to validate, or None.

        Returns:
            The validated column name, or None if not set.

        Raises:
            ValueError: If the column name contains unsafe characters.
        """
        if v is None:
            return v
        if not _SQL_IDENTIFIER_PATTERN.match(v):
            raise ValueError(
                f"Invalid SQL identifier for primary_key_column: '{v}'. "
                "Must start with a letter or underscore, followed by letters, "
                "digits, or underscores only (pattern: ^[a-zA-Z_][a-zA-Z0-9_]*$)."
            )
        return v

    @field_validator("default_order_by")
    @classmethod
    def validate_default_order_by(cls, v: str | None) -> str | None:
        """Validate default_order_by contains only safe SQL identifiers.

        Allows format: column [ASC|DESC] [, column [ASC|DESC]]*
        Prevents SQL injection from misconfigured ORDER BY clauses.

        Args:
            v: The ORDER BY clause to validate, or None.

        Returns:
            The validated ORDER BY clause, or None if not set.

        Raises:
            ValueError: If any column reference contains unsafe characters.
        """
        if v is None:
            return v

        # Split by comma to get individual column specifications
        components = v.split(",")

        for component in components:
            component = component.strip()
            if not component:
                raise ValueError(
                    f"Invalid ORDER BY clause: '{v}'. "
                    "Empty component found after splitting by comma."
                )
            if not _ORDER_BY_COMPONENT_PATTERN.match(component):
                raise ValueError(
                    f"Invalid ORDER BY component: '{component}' in '{v}'. "
                    "Each component must be a valid SQL identifier optionally "
                    "followed by ASC or DESC. Column names must start with a "
                    "letter or underscore, followed by letters, digits, or "
                    "underscores only."
                )
        return v


__all__: list[str] = ["ModelRepositoryRuntimeConfig"]
