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

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
        default="pattern_id",
        description="Primary key column for ORDER BY injection to ensure deterministic results",
    )
    default_order_by: str | None = Field(
        default="score DESC, pattern_id ASC",
        description="Default ORDER BY clause when primary_key_column is declared",
    )

    # Metrics emission
    emit_metrics: bool = Field(
        default=True,
        description="Emit duration_ms and rows_returned metrics for observability",
    )


__all__: list[str] = ["ModelRepositoryRuntimeConfig"]
