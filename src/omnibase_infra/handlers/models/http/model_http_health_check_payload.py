# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""HTTP Health Check Payload Model.

This module provides the Pydantic model for HTTP handler health check results.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.handlers.models.http.enum_http_operation_type import (
    EnumHttpOperationType,
)


class ModelHttpHealthCheckPayload(BaseModel):
    """Payload for HTTP handler health check result.

    Contains HTTP handler health status and configuration metrics.

    Attributes:
        operation_type: Discriminator set to "health_check"
        healthy: Whether the HTTP client is healthy
        initialized: Whether the handler has been initialized
        adapter_type: Handler type identifier (e.g., "http")
        timeout_seconds: Configured timeout in seconds
        max_request_size: Maximum request body size in bytes
        max_response_size: Maximum response body size in bytes

    Example:
        >>> payload = ModelHttpHealthCheckPayload(
        ...     healthy=True,
        ...     initialized=True,
        ...     adapter_type="http",
        ...     timeout_seconds=30.0,
        ...     max_request_size=10485760,
        ...     max_response_size=52428800,
        ... )
        >>> print(payload.healthy)
        True
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    operation_type: Literal[EnumHttpOperationType.HEALTH_CHECK] = Field(
        default=EnumHttpOperationType.HEALTH_CHECK,
        description="Operation type discriminator",
    )
    healthy: bool = Field(
        description="Whether the HTTP client is healthy",
    )
    initialized: bool = Field(
        description="Whether the handler has been initialized",
    )
    adapter_type: str = Field(
        description="Handler type identifier",
    )
    timeout_seconds: float = Field(
        description="Configured timeout in seconds",
    )
    max_request_size: int = Field(
        ge=0,
        description="Maximum request body size in bytes",
    )
    max_response_size: int = Field(
        ge=0,
        description="Maximum response body size in bytes",
    )


__all__: list[str] = ["ModelHttpHealthCheckPayload"]
