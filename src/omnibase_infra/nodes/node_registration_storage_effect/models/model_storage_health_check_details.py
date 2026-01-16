# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Storage Health Check Details Model.

This module provides the Pydantic model for backend-specific diagnostic
information in health check results, replacing untyped dict[str, object]
with strongly-typed fields.

Architecture:
    ModelStorageHealthCheckDetails captures backend-specific diagnostics:
    - Connection pool statistics (pool_size, active_connections)
    - Server version information
    - Replication status
    - Custom backend-specific extensions

    The model uses optional fields to accommodate different backends
    that may provide varying levels of diagnostic information.

Related:
    - ModelStorageHealthCheckResult: Parent model that contains details
    - ProtocolRegistrationStorageHandler: Protocol for storage backends
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelStorageHealthCheckDetails(BaseModel):
    """Backend-specific diagnostic information for health checks.

    This model captures diagnostic details from storage backends.
    All fields are optional as different backends provide varying
    levels of information.

    Immutability:
        This model uses frozen=True to ensure details are immutable
        once created, enabling safe sharing and logging.

    Attributes:
        pool_size: Maximum size of the connection pool (if applicable).
        active_connections: Current number of active connections.
        idle_connections: Current number of idle connections.
        waiting_connections: Number of requests waiting for connections.
        server_version: Database server version string.
        is_primary: Whether this is the primary/leader node.
        replication_lag_ms: Replication lag in milliseconds (for replicas).
        database_name: Name of the database being used.
        schema_name: Schema name within the database.
        extensions: List of enabled database extensions.

    Example (PostgreSQL):
        >>> details = ModelStorageHealthCheckDetails(
        ...     pool_size=10,
        ...     active_connections=3,
        ...     idle_connections=7,
        ...     waiting_connections=0,
        ...     server_version="PostgreSQL 16.1",
        ...     is_primary=True,
        ...     database_name="omninode_bridge",
        ...     schema_name="public",
        ... )
        >>> details.pool_size
        10

    Example (minimal details from mock backend):
        >>> details = ModelStorageHealthCheckDetails(
        ...     server_version="mock-1.0.0",
        ... )
        >>> details.pool_size is None
        True
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    pool_size: int | None = Field(
        default=None,
        description="Maximum size of the connection pool (if applicable)",
        ge=0,
    )
    active_connections: int | None = Field(
        default=None,
        description="Current number of active connections",
        ge=0,
    )
    idle_connections: int | None = Field(
        default=None,
        description="Current number of idle connections",
        ge=0,
    )
    waiting_connections: int | None = Field(
        default=None,
        description="Number of requests waiting for connections",
        ge=0,
    )
    server_version: str | None = Field(
        default=None,
        description="Database server version string",
    )
    is_primary: bool | None = Field(
        default=None,
        description="Whether this is the primary/leader node",
    )
    replication_lag_ms: float | None = Field(
        default=None,
        description="Replication lag in milliseconds (for replicas)",
        ge=0.0,
    )
    database_name: str | None = Field(
        default=None,
        description="Name of the database being used",
    )
    schema_name: str | None = Field(
        default=None,
        description="Schema name within the database",
    )
    extensions: list[str] | None = Field(
        default=None,
        description="List of enabled database extensions",
    )


__all__ = ["ModelStorageHealthCheckDetails"]
