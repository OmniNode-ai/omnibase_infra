# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Component Health Model for detailed diagnostics.

Provides a Pydantic model representing the health status of an individual
infrastructure component (e.g., Kafka, PostgreSQL, Consul). Used by the
``/health/detailed`` endpoint to return per-component health breakdowns.

OMN-519: Health Check - Add degraded status detailed diagnostics.

Example:
    >>> from omnibase_infra.runtime.models import ModelComponentHealth
    >>>
    >>> # Create a healthy component
    >>> healthy = ModelComponentHealth(
    ...     name="kafka",
    ...     status="healthy",
    ...     latency_ms=5.2,
    ... )
    >>>
    >>> # Create a degraded component with error
    >>> degraded = ModelComponentHealth(
    ...     name="consul",
    ...     status="degraded",
    ...     error="connection timeout",
    ...     last_healthy="2025-12-08T10:00:00Z",
    ... )
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelComponentHealth(BaseModel):
    """Health status of a single infrastructure component.

    Represents the detailed health status of one component in the runtime,
    including timing information and error details for degraded/unhealthy
    states.

    Attributes:
        name: Component identifier (e.g., "kafka", "postgres", "consul").
        status: Health status of this component.
        latency_ms: Last health check latency in milliseconds, if measured.
        error: Error message if component is degraded or unhealthy.
        last_healthy: ISO 8601 timestamp of the last successful health check.
        details: Additional component-specific health data.
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    name: str = Field(
        ...,
        description="Component identifier (e.g., 'kafka', 'postgres', 'consul')",
        min_length=1,
    )
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Health status of this component",
    )
    latency_ms: float | None = Field(
        default=None,
        description="Last health check latency in milliseconds",
    )
    error: str | None = Field(
        default=None,
        description="Error message if component is degraded or unhealthy",
    )
    last_healthy: str | None = Field(
        default=None,
        description="ISO 8601 timestamp of last successful health check",
    )
    details: dict[str, JsonType] | None = Field(
        default=None,
        description="Additional component-specific health data",
    )

    @classmethod
    def healthy(
        cls,
        name: str,
        latency_ms: float | None = None,
        last_healthy: str | None = None,
        details: dict[str, JsonType] | None = None,
    ) -> ModelComponentHealth:
        """Create a healthy component status.

        Args:
            name: Component identifier.
            latency_ms: Health check latency in milliseconds.
            last_healthy: ISO 8601 timestamp of last successful check.
            details: Additional health data.

        Returns:
            ModelComponentHealth with healthy status.
        """
        return cls(
            name=name,
            status="healthy",
            latency_ms=latency_ms,
            last_healthy=last_healthy,
            details=details,
        )

    @classmethod
    def degraded(
        cls,
        name: str,
        error: str,
        last_healthy: str | None = None,
        latency_ms: float | None = None,
        details: dict[str, JsonType] | None = None,
    ) -> ModelComponentHealth:
        """Create a degraded component status.

        Args:
            name: Component identifier.
            error: Error message describing the degradation.
            last_healthy: ISO 8601 timestamp of last successful check.
            latency_ms: Health check latency in milliseconds.
            details: Additional health data.

        Returns:
            ModelComponentHealth with degraded status.
        """
        return cls(
            name=name,
            status="degraded",
            error=error,
            last_healthy=last_healthy,
            latency_ms=latency_ms,
            details=details,
        )

    @classmethod
    def unhealthy(
        cls,
        name: str,
        error: str,
        last_healthy: str | None = None,
        details: dict[str, JsonType] | None = None,
    ) -> ModelComponentHealth:
        """Create an unhealthy component status.

        Args:
            name: Component identifier.
            error: Error message describing the failure.
            last_healthy: ISO 8601 timestamp of last successful check.
            details: Additional health data.

        Returns:
            ModelComponentHealth with unhealthy status.
        """
        return cls(
            name=name,
            status="unhealthy",
            error=error,
            last_healthy=last_healthy,
            details=details,
        )


__all__: list[str] = ["ModelComponentHealth"]
