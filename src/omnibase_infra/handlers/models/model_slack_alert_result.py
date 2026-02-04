# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Slack Alert Result Model.

Provides the response model for Slack webhook operations.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelSlackAlertResult(BaseModel):
    """Response from Slack webhook operations.

    This model captures the result of a Slack alert send operation,
    including success/failure status, timing, and retry information.

    Attributes:
        success: Whether the alert was delivered successfully
        duration_ms: Time taken for the operation in milliseconds
        correlation_id: UUID from the original request for tracing
        error: Sanitized error message if success is False
        error_code: Error code for programmatic handling
        retry_count: Number of retry attempts made

    Example:
        >>> result = ModelSlackAlertResult(
        ...     success=True,
        ...     duration_ms=123.45,
        ...     correlation_id=uuid4(),
        ... )
        >>> print(result.success)
        True
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    success: bool = Field(
        ...,
        description="Whether the alert was delivered successfully",
    )
    duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken for the operation in milliseconds",
    )
    correlation_id: UUID = Field(
        ...,
        description="UUID from the original request for tracing",
    )
    error: str | None = Field(
        default=None,
        description="Sanitized error message if success is False",
    )
    error_code: str | None = Field(
        default=None,
        description="Error code for programmatic handling",
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of retry attempts made",
    )


__all__ = ["ModelSlackAlertResult"]
