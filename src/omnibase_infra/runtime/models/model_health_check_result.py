# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Health Check Result Model.

This module provides the Pydantic model for component health check operation results.

Design Pattern:
    ModelHealthCheckResult replaces tuple[str, Any] returns from
    check_health() with a strongly-typed model that provides:
    - Component type identification
    - Typed health status with structured details
    - Factory methods for common healthy/unhealthy patterns

Thread Safety:
    ModelHealthCheckResult is immutable (frozen=True) after creation,
    making it thread-safe for concurrent read access.

Example:
    >>> from omnibase_infra.runtime.models import ModelHealthCheckResult
    >>>
    >>> # Create a healthy result
    >>> healthy = ModelHealthCheckResult.healthy_result("kafka")
    >>> healthy.healthy
    True
    >>>
    >>> # Create an unhealthy result (timeout)
    >>> unhealthy = ModelHealthCheckResult.timeout_result("db", 5.0)
    >>> unhealthy.healthy
    False
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelHealthCheckResult(BaseModel):
    """Result of a handler health check operation.

    Encapsulates the result of checking a single handler's health,
    providing the handler type, health status, and detailed health data.

    Attributes:
        handler_type: The handler type identifier (e.g., "http", "db").
        healthy: Whether the handler is healthy and operational.
        details: Detailed health check data returned by the handler.
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    handler_type: str = Field(
        ...,
        description="Handler type identifier",
        min_length=1,
    )
    healthy: bool = Field(
        ...,
        description="Whether the handler is healthy",
    )
    details: dict[str, JsonType] = Field(
        default_factory=dict,
        description="Detailed health check data from the handler",
    )

    @classmethod
    def healthy_result(
        cls,
        handler_type: str,
        note: str = "",
    ) -> ModelHealthCheckResult:
        """Create a healthy result for a handler.

        Args:
            handler_type: The handler type identifier.
            note: Optional note about the healthy status. Empty string means no note.

        Returns:
            ModelHealthCheckResult indicating healthy status.

        Example:
            >>> result = ModelHealthCheckResult.healthy_result("kafka")
            >>> result.healthy
            True
        """
        details: dict[str, JsonType] = {"healthy": True}
        if note:
            details["note"] = note
        return cls(handler_type=handler_type, healthy=True, details=details)

    @classmethod
    def no_health_check_result(cls, handler_type: str) -> ModelHealthCheckResult:
        """Create a result for a handler without health_check method.

        By convention, handlers without health_check are assumed healthy.

        Args:
            handler_type: The handler type identifier.

        Returns:
            ModelHealthCheckResult indicating healthy (no health_check method).

        Example:
            >>> result = ModelHealthCheckResult.no_health_check_result("custom")
            >>> result.healthy
            True
        """
        return cls(
            handler_type=handler_type,
            healthy=True,
            details={"healthy": True, "note": "no health_check method"},
        )

    @classmethod
    def timeout_result(
        cls,
        handler_type: str,
        timeout_seconds: float,
    ) -> ModelHealthCheckResult:
        """Create an unhealthy result for a health check timeout.

        Args:
            handler_type: The handler type identifier.
            timeout_seconds: The timeout duration that was exceeded.

        Returns:
            ModelHealthCheckResult indicating timeout failure.

        Example:
            >>> result = ModelHealthCheckResult.timeout_result("db", 5.0)
            >>> result.healthy
            False
        """
        return cls(
            handler_type=handler_type,
            healthy=False,
            details={
                "healthy": False,
                "error": f"health check timeout after {timeout_seconds}s",
            },
        )

    @classmethod
    def error_result(
        cls,
        handler_type: str,
        error: str,
    ) -> ModelHealthCheckResult:
        """Create an unhealthy result for a health check exception.

        Args:
            handler_type: The handler type identifier.
            error: The error message from the exception.

        Returns:
            ModelHealthCheckResult indicating error failure.

        Example:
            >>> result = ModelHealthCheckResult.error_result(
            ...     "vault",
            ...     "Authentication token expired",
            ... )
            >>> result.healthy
            False
        """
        return cls(
            handler_type=handler_type,
            healthy=False,
            details={"healthy": False, "error": error},
        )

    @classmethod
    def from_handler_response(
        cls,
        handler_type: str,
        health_response: object,
    ) -> ModelHealthCheckResult:
        """Create a result from a raw handler health check response.

        Parses the handler's health check response and extracts health status.

        Args:
            handler_type: The handler type identifier.
            health_response: The raw response from handler.health_check().

        Returns:
            ModelHealthCheckResult from the response.

        Example:
            >>> response = {"healthy": True, "lag": 100}
            >>> result = ModelHealthCheckResult.from_handler_response(
            ...     "kafka",
            ...     response,
            ... )
            >>> result.healthy
            True
        """
        if isinstance(health_response, dict):
            healthy = bool(health_response.get("healthy", False))
            return cls(
                handler_type=handler_type,
                healthy=healthy,
                details=health_response,
            )
        # Non-dict response - treat as details, assume healthy
        # Convert to string representation for JsonType compatibility
        return cls(
            handler_type=handler_type,
            healthy=True,
            details={"raw_response": str(health_response)},
        )

    def __str__(self) -> str:
        """Return a human-readable string representation for debugging.

        Returns:
            String format: "ModelHealthCheckResult(handler_type='...', healthy=...)"
        """
        status = "healthy" if self.healthy else "unhealthy"
        return f"ModelHealthCheckResult(handler_type='{self.handler_type}', {status})"


__all__: list[str] = ["ModelHealthCheckResult"]
