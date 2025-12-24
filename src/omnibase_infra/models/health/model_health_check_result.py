# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Health Check Result Model for infrastructure components."""

from pydantic import BaseModel, ConfigDict, Field


class ModelHealthCheckResult(BaseModel):
    """Result of a health check operation on infrastructure components.

    Provides standardized health check responses with typed fields
    replacing loose dict[str, object] returns.

    Attributes:
        healthy: Whether the component is healthy and operational.
        reason: Human-readable explanation of the health status.
        error_type: Exception type name if health check failed.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    healthy: bool = Field(
        ...,
        description="Whether the component is healthy and operational.",
    )
    reason: str = Field(
        ...,
        description="Human-readable explanation of the health status.",
    )
    error_type: str | None = Field(
        default=None,
        description="Exception type name if health check failed.",
    )


__all__ = ["ModelHealthCheckResult"]
