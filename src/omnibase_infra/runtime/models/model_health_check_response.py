# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Health Check Response Model.

This module provides the Pydantic model for HTTP health check endpoint responses.
The model serializes to JSON for Docker/Kubernetes health probes.

Design Pattern:
    ModelHealthCheckResponse replaces dict[str, object] responses from the
    health server with a strongly-typed model that provides:
    - Typed status field with Literal values
    - Version string for runtime identification
    - Optional details for successful health checks
    - Optional error fields for failure cases

Thread Safety:
    ModelHealthCheckResponse is immutable (frozen=True) after creation,
    making it thread-safe for concurrent read access.

Example:
    >>> from omnibase_infra.runtime.models import ModelHealthCheckResponse
    >>>
    >>> # Create a healthy response
    >>> response = ModelHealthCheckResponse(
    ...     status="healthy",
    ...     version="1.0.0",
    ...     details={"healthy": True, "handlers": {}},
    ... )
    >>> response.model_dump_json()
    '{"status":"healthy","version":"1.0.0","details":{"healthy":true,...}}'
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelHealthCheckResponse(BaseModel):
    """HTTP response model for health check endpoints.

    Encapsulates the JSON response returned by GET /health and GET /ready
    endpoints, providing type-safe serialization for health probe responses.

    For successful health checks:
        - status indicates overall health state
        - version identifies the runtime
        - details contains full health check data

    For failed health checks (exceptions during health check):
        - status is always "unhealthy"
        - error contains the exception message
        - error_type contains the exception class name
        - correlation_id enables distributed tracing

    Attributes:
        status: Overall health status of the runtime.
        version: Runtime version string for identification.
        details: Full health check data (present on success).
        error: Exception message (present on failure).
        error_type: Exception class name (present on failure).
        correlation_id: Tracing ID for debugging (present on failure).
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Overall health status of the runtime",
    )
    version: str = Field(
        ...,
        description="Runtime version string",
    )
    details: dict[str, JsonType] | None = Field(
        default=None,
        description="Full health check data from RuntimeHostProcess",
    )
    error: str | None = Field(
        default=None,
        description="Exception message if health check failed",
    )
    error_type: str | None = Field(
        default=None,
        description="Exception class name if health check failed",
    )
    correlation_id: str | None = Field(
        default=None,
        description="Correlation ID for distributed tracing on failure",
    )

    @classmethod
    def success(
        cls,
        status: Literal["healthy", "degraded", "unhealthy"],
        version: str,
        details: dict[str, JsonType],
    ) -> ModelHealthCheckResponse:
        """Create a successful health check response.

        Args:
            status: The determined health status.
            version: Runtime version string.
            details: Full health check data from the runtime.

        Returns:
            ModelHealthCheckResponse for successful health check.

        Example:
            >>> response = ModelHealthCheckResponse.success(
            ...     status="healthy",
            ...     version="1.0.0",
            ...     details={"healthy": True},
            ... )
        """
        return cls(
            status=status,
            version=version,
            details=details,
        )

    @classmethod
    def failure(
        cls,
        version: str,
        error: str,
        error_type: str,
        correlation_id: str,
    ) -> ModelHealthCheckResponse:
        """Create a failure health check response.

        Used when health_check() itself raises an exception.

        Args:
            version: Runtime version string.
            error: The exception message.
            error_type: The exception class name.
            correlation_id: Tracing ID for debugging.

        Returns:
            ModelHealthCheckResponse for failed health check.

        Example:
            >>> response = ModelHealthCheckResponse.failure(
            ...     version="1.0.0",
            ...     error="Connection refused",
            ...     error_type="ConnectionError",
            ...     correlation_id="abc-123",
            ... )
        """
        return cls(
            status="unhealthy",
            version=version,
            error=error,
            error_type=error_type,
            correlation_id=correlation_id,
        )


__all__: list[str] = ["ModelHealthCheckResponse"]
