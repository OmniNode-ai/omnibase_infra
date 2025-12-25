# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Database Describe Response Model.

This module provides the Pydantic model for database handler metadata
and capabilities responses.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelDbDescribeResponse(BaseModel):
    """Database handler metadata and capabilities response.

    Provides handler metadata including supported operations,
    configuration, and version information.

    Attributes:
        adapter_type: Type of handler (e.g., "db")
        supported_operations: List of supported operation types
        pool_size: Connection pool size
        timeout_seconds: Query timeout in seconds
        initialized: Whether the handler has been initialized
        version: Handler version string

    Example:
        >>> describe = ModelDbDescribeResponse(
        ...     adapter_type="db",
        ...     supported_operations=["db.query", "db.execute"],
        ...     pool_size=5,
        ...     timeout_seconds=30.0,
        ...     initialized=True,
        ...     version="0.1.0-mvp",
        ... )
        >>> print(describe.supported_operations)
        ['db.query', 'db.execute']
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,  # Support pytest-xdist compatibility
    )

    adapter_type: str = Field(
        description="Type of handler (e.g., 'db')",
    )
    supported_operations: list[str] = Field(
        description="List of supported operation types",
    )
    pool_size: int = Field(
        ge=1,
        description="Connection pool size",
    )
    timeout_seconds: float = Field(
        gt=0,
        description="Query timeout in seconds",
    )
    initialized: bool = Field(
        description="Whether the handler has been initialized",
    )
    version: str = Field(
        description="Handler version string",
    )


__all__: list[str] = ["ModelDbDescribeResponse"]
