# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Detailed Health Check Response Model.

Provides an extended health check response with per-component diagnostics,
timestamps, and dependency health breakdowns for the ``/health/detailed``
endpoint.

OMN-519: Health Check - Add degraded status detailed diagnostics.

Example:
    >>> from omnibase_infra.runtime.models import ModelDetailedHealthResponse
    >>> from omnibase_infra.runtime.models import ModelComponentHealth
    >>>
    >>> response = ModelDetailedHealthResponse(
    ...     status="degraded",
    ...     version="1.0.0",
    ...     components={
    ...         "kafka": ModelComponentHealth.healthy("kafka", latency_ms=5.0),
    ...         "consul": ModelComponentHealth.degraded("consul", error="timeout"),
    ...     },
    ...     checked_at="2025-12-08T10:05:00Z",
    ... )
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType
from omnibase_infra.runtime.models.model_component_health import ModelComponentHealth


class ModelDetailedHealthResponse(BaseModel):
    """Extended HTTP response model for ``/health/detailed`` endpoint.

    Includes per-component health breakdowns, timestamps, and full
    diagnostics in addition to the standard health check fields.

    Attributes:
        status: Overall health status of the runtime.
        version: Runtime version string for identification.
        checked_at: ISO 8601 timestamp when this health check was performed.
        components: Per-component health status breakdown.
        details: Full health check data from RuntimeHostProcess.
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
    checked_at: str | None = Field(
        default=None,
        description="ISO 8601 timestamp when health check was performed",
    )
    components: dict[str, ModelComponentHealth] | None = Field(
        default=None,
        description="Per-component health status breakdown",
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
        checked_at: str,
        components: dict[str, ModelComponentHealth],
        details: dict[str, JsonType],
    ) -> ModelDetailedHealthResponse:
        """Create a successful detailed health check response.

        Args:
            status: The determined health status.
            version: Runtime version string.
            checked_at: ISO 8601 timestamp of check.
            components: Per-component health breakdown.
            details: Full health check data from the runtime.

        Returns:
            ModelDetailedHealthResponse for successful health check.
        """
        return cls(
            status=status,
            version=version,
            checked_at=checked_at,
            components=components,
            details=details,
        )

    @classmethod
    def failure(
        cls,
        version: str,
        error: str,
        error_type: str,
        correlation_id: str,
    ) -> ModelDetailedHealthResponse:
        """Create a failure detailed health check response.

        Args:
            version: Runtime version string.
            error: The exception message.
            error_type: The exception class name.
            correlation_id: Tracing ID for debugging.

        Returns:
            ModelDetailedHealthResponse for failed health check.
        """
        return cls(
            status="unhealthy",
            version=version,
            error=error,
            error_type=error_type,
            correlation_id=correlation_id,
        )


__all__: list[str] = ["ModelDetailedHealthResponse"]
