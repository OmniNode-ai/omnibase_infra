# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Health Check Configuration Model for Service Discovery.

This module provides ModelHealthCheckConfig, representing health check
configuration for service registration operations.

Architecture:
    ModelHealthCheckConfig captures health check parameters:
    - endpoint: HTTP endpoint for health checks
    - interval: Time between checks
    - timeout: Maximum time to wait for response

    This model is backend-agnostic and can be translated to
    Consul, Kubernetes, Etcd-specific health check formats.

Related:
    - ModelServiceRegistration: Uses this model for health_check field
    - ProtocolServiceDiscoveryHandler: Handler protocol for backends
    - OMN-1131: Capability-oriented node architecture
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Duration string pattern: supports formats like "10s", "1m", "500ms", "1h", "2d"
# Format: <positive_number><unit> where unit is one of: ms, s, m, h, d
DURATION_PATTERN = re.compile(r"^([1-9]\d*|0)(\.\d+)?(ms|s|m|h|d)$")


class ModelHealthCheckConfig(BaseModel):
    """Health check configuration for service registration.

    Defines how the service discovery backend should verify
    that a service instance is healthy and responding.

    Immutability:
        This model uses frozen=True to ensure configs are immutable
        once created, enabling safe concurrent access.

    Attributes:
        endpoint: HTTP endpoint path for health checks (e.g., "/health").
        interval: Time between health checks (e.g., "10s", "1m").
        timeout: Maximum time to wait for response (e.g., "5s").
        method: HTTP method for health check (default: "GET").
        expected_status: Expected HTTP status code (default: 200).

    Example:
        >>> config = ModelHealthCheckConfig(
        ...     endpoint="/health",
        ...     interval="10s",
        ...     timeout="5s",
        ... )
        >>> config.endpoint
        '/health'
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    endpoint: str = Field(
        default="/health",
        description="HTTP endpoint path for health checks",
    )
    interval: str = Field(
        default="10s",
        description="Time between health checks (e.g., '10s', '1m')",
    )
    timeout: str = Field(
        default="5s",
        description="Maximum time to wait for response (e.g., '5s')",
    )
    method: str = Field(
        default="GET",
        description="HTTP method for health check",
    )
    expected_status: int = Field(
        default=200,
        description="Expected HTTP status code for healthy response",
        ge=100,
        le=599,
    )

    @field_validator("interval", "timeout")
    @classmethod
    def validate_duration_format(cls, v: str) -> str:
        """Validate that duration strings follow the expected format.

        Supported formats:
            - "10s" (seconds)
            - "1m" (minutes)
            - "500ms" (milliseconds)
            - "1h" (hours)
            - "2d" (days)
            - Decimal values like "1.5s" are supported

        Args:
            v: The duration string to validate.

        Returns:
            The validated duration string.

        Raises:
            ValueError: If the duration string format is invalid.

        Example:
            >>> ModelHealthCheckConfig(interval="10s", timeout="5s")  # Valid
            >>> ModelHealthCheckConfig(interval="invalid")  # Raises ValueError
        """
        if not DURATION_PATTERN.match(v):
            raise ValueError(
                f"Invalid duration format: '{v}'. "
                f"Expected format: <number><unit> where unit is one of: ms, s, m, h, d. "
                f"Examples: '10s', '1m', '500ms', '1.5h'"
            )
        return v


__all__ = ["ModelHealthCheckConfig"]
