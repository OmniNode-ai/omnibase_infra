# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Database Describe Response Model.

This module provides the Pydantic model for database handler metadata
and capabilities responses.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelDbDescribeResponse(BaseModel):
    """Database handler metadata and capabilities response.

    Provides handler metadata including supported operations,
    configuration, and version information.

    Attributes:
        handler_type: Architectural role of handler (e.g., "infra_handler")
        handler_category: Behavioral classification (e.g., "effect")
        supported_operations: List of supported operation types
        pool_size: Connection pool size
        timeout_seconds: Query timeout in seconds
        initialized: Whether the handler has been initialized
        version: Handler version string
        circuit_breaker: Circuit breaker state information (optional)

    Example:
        >>> describe = ModelDbDescribeResponse(
        ...     handler_type="infra_handler",
        ...     handler_category="effect",
        ...     supported_operations=["db.query", "db.execute"],
        ...     pool_size=5,
        ...     timeout_seconds=30.0,
        ...     initialized=True,
        ...     version="0.1.0-mvp",
        ...     circuit_breaker={"state": "closed", "failures": 0},
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

    handler_type: str = Field(
        description="Architectural role of handler (e.g., 'infra_handler')",
    )
    handler_category: str = Field(
        description="Behavioral classification (e.g., 'effect')",
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
    circuit_breaker: dict[str, JsonType] | None = Field(
        default=None,
        description="Circuit breaker state information (state, failures, threshold, etc.)",
    )


__all__: list[str] = ["ModelDbDescribeResponse"]
