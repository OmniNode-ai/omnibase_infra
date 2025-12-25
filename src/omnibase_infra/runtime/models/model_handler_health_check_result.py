# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Health Check Result Model.

This module provides the Pydantic model for handler health check operation results.

Design Pattern:
    ModelHandlerHealthCheckResult replaces tuple[str, JsonValue] returns from
    check_handler_health() with a strongly-typed model that provides:
    - Handler type identification
    - Typed health status with structured details
    - Factory methods for common healthy/unhealthy patterns

Thread Safety:
    ModelHandlerHealthCheckResult is immutable (frozen=True) after creation,
    making it thread-safe for concurrent read access.

Example:
    >>> from omnibase_infra.runtime.models import ModelHandlerHealthCheckResult
    >>>
    >>> # Create a healthy result
    >>> healthy = ModelHandlerHealthCheckResult.healthy_result("kafka")
    >>> healthy.healthy
    True
    >>>
    >>> # Create an unhealthy result (timeout)
    >>> unhealthy = ModelHandlerHealthCheckResult.timeout_result("db", 5.0)
    >>> unhealthy.healthy
    False
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from omnibase_core.types import JsonValue


class ModelHandlerHealthCheckResult(BaseModel):
    """Result of a handler health check operation.

    Encapsulates the result of checking a single handler's health,
    providing the handler type, health status, and detailed health data.

    Attributes:
        handler_type: The handler type identifier (e.g., "http", "db").
        healthy: Whether the handler is healthy and operational.
        details: Detailed health check data returned by the handler.
    """

    model_config = ConfigDict(
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
    details: dict[str, object] = Field(
        default_factory=dict,
        description="Detailed health check data from the handler",
    )

    @classmethod
    def healthy_result(
        cls,
        handler_type: str,
        note: str = "",
    ) -> "ModelHandlerHealthCheckResult":
        """Create a healthy result for a handler.

        Args:
            handler_type: The handler type identifier.
            note: Optional note about the healthy status. Empty string means no note.

        Returns:
            ModelHandlerHealthCheckResult indicating healthy status.

        Example:
            >>> result = ModelHandlerHealthCheckResult.healthy_result("kafka")
            >>> result.healthy
            True
        """
        details: dict[str, object] = {"healthy": True}
        if note:
            details["note"] = note
        return cls(handler_type=handler_type, healthy=True, details=details)

    @classmethod
    def no_health_check_result(cls, handler_type: str) -> "ModelHandlerHealthCheckResult":
        """Create a result for a handler without health_check method.

        By convention, handlers without health_check are assumed healthy.

        Args:
            handler_type: The handler type identifier.

        Returns:
            ModelHandlerHealthCheckResult indicating healthy (no health_check method).

        Example:
            >>> result = ModelHandlerHealthCheckResult.no_health_check_result("custom")
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
    ) -> "ModelHandlerHealthCheckResult":
        """Create an unhealthy result for a health check timeout.

        Args:
            handler_type: The handler type identifier.
            timeout_seconds: The timeout duration that was exceeded.

        Returns:
            ModelHandlerHealthCheckResult indicating timeout failure.

        Example:
            >>> result = ModelHandlerHealthCheckResult.timeout_result("db", 5.0)
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
    ) -> "ModelHandlerHealthCheckResult":
        """Create an unhealthy result for a health check exception.

        Args:
            handler_type: The handler type identifier.
            error: The error message from the exception.

        Returns:
            ModelHandlerHealthCheckResult indicating error failure.

        Example:
            >>> result = ModelHandlerHealthCheckResult.error_result(
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
        health_response: "JsonValue",
    ) -> "ModelHandlerHealthCheckResult":
        """Create a result from a raw handler health check response.

        Parses the handler's health check response and extracts health status.

        Args:
            handler_type: The handler type identifier.
            health_response: The raw response from handler.health_check().

        Returns:
            ModelHandlerHealthCheckResult from the response.

        Example:
            >>> response = {"healthy": True, "lag": 100}
            >>> result = ModelHandlerHealthCheckResult.from_handler_response(
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
        return cls(
            handler_type=handler_type,
            healthy=True,
            details={"raw_response": health_response},
        )


__all__: list[str] = ["ModelHandlerHealthCheckResult"]
