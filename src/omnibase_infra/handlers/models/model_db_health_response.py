# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Database Health Response Model.

This module provides the Pydantic model for database handler health check
responses.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelDbHealthResponse(BaseModel):
    """Database handler health check response.

    Provides current health status and configuration details
    for the database handler.

    Attributes:
        healthy: Whether the handler can successfully connect to database
        initialized: Whether the handler has been initialized
        adapter_type: Type of handler (e.g., "db")
        pool_size: Connection pool size
        timeout_seconds: Query timeout in seconds

    Example:
        >>> health = ModelDbHealthResponse(
        ...     healthy=True,
        ...     initialized=True,
        ...     adapter_type="db",
        ...     pool_size=5,
        ...     timeout_seconds=30.0,
        ... )
        >>> print(health.healthy)
        True
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,  # Support pytest-xdist compatibility
    )

    healthy: bool = Field(
        description="Whether the handler can successfully connect to database",
    )
    initialized: bool = Field(
        description="Whether the handler has been initialized",
    )
    adapter_type: str = Field(
        description="Type of handler (e.g., 'db')",
    )
    pool_size: int = Field(
        ge=1,
        description="Connection pool size",
    )
    timeout_seconds: float = Field(
        gt=0,
        description="Query timeout in seconds",
    )


__all__: list[str] = ["ModelDbHealthResponse"]
