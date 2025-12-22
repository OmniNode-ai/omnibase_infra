# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Backend Result Model for Registry Effect Operations.

This module provides ModelBackendResult, representing the result of an individual
backend operation (Consul or PostgreSQL) within the dual-registration workflow.

Architecture:
    ModelBackendResult captures the outcome of a single backend operation:
    - success: Whether the operation completed successfully
    - error: Error message if the operation failed (sanitized)
    - duration_ms: Time taken for the operation
    - retries: Number of retry attempts made

    This model is used within ModelRegistryResponse to report per-backend status,
    enabling partial failure detection and targeted retry strategies.

Security:
    Error messages MUST be sanitized before inclusion. Never include:
    - Credentials, connection strings, or secrets
    - Internal IP addresses or hostnames
    - PII (names, emails, etc.)

    See CLAUDE.md "Error Sanitization Guidelines" for complete rules.

Related:
    - ModelRegistryResponse: Uses this model for consul_result and postgres_result
    - RegistryEffect: Effect node that produces these results
    - OMN-954: Partial failure scenario testing
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelBackendResult(BaseModel):
    """Result of an individual backend operation.

    Captures the outcome of a single backend operation (Consul or PostgreSQL)
    within the dual-registration workflow. Used to enable partial failure
    detection and targeted retry strategies.

    Immutability:
        This model uses frozen=True to ensure backend results are immutable
        once created, supporting safe concurrent access and comparison.

    Attributes:
        success: Whether the backend operation completed successfully.
        error: Sanitized error message if success is False.
        error_code: Optional error code for programmatic handling.
        duration_ms: Time taken for the operation in milliseconds.
        retries: Number of retry attempts made before final result.
        backend_id: Optional identifier for the backend instance.

    Example:
        >>> result = ModelBackendResult(
        ...     success=True,
        ...     duration_ms=45.2,
        ...     retries=0,
        ... )
        >>> result.success
        True

    Example (failure case):
        >>> result = ModelBackendResult(
        ...     success=False,
        ...     error="Connection refused to database host",
        ...     error_code="DATABASE_CONNECTION_ERROR",
        ...     duration_ms=5000.0,
        ...     retries=3,
        ... )
        >>> result.success
        False
        >>> result.error
        'Connection refused to database host'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: bool = Field(
        ...,
        description="Whether the backend operation completed successfully",
    )
    error: str | None = Field(
        default=None,
        description="Sanitized error message if success is False",
    )
    error_code: str | None = Field(
        default=None,
        description="Error code for programmatic handling (e.g., DATABASE_CONNECTION_ERROR)",
    )
    duration_ms: float = Field(
        default=0.0,
        description="Time taken for the operation in milliseconds",
        ge=0.0,
    )
    retries: int = Field(
        default=0,
        description="Number of retry attempts made before final result",
        ge=0,
    )
    backend_id: str | None = Field(
        default=None,
        description="Optional identifier for the backend instance",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for tracing the operation",
    )


__all__ = ["ModelBackendResult"]
