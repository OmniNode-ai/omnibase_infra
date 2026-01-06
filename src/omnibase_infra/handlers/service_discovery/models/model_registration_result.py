# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Result Model.

This module provides the ModelRegistrationResult class representing the outcome
of a service registration operation.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelRegistrationResult(BaseModel):
    """Result of a service registration operation.

    Captures the outcome of registering a service with the service discovery
    backend. Immutable once created.

    Attributes:
        success: Whether the registration completed successfully.
        service_id: The registered service ID.
        error: Sanitized error message if registration failed.
        duration_ms: Time taken for the operation in milliseconds.
        backend_type: The backend that handled the registration.
        correlation_id: Correlation ID for tracing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: bool = Field(
        ...,
        description="Whether the registration completed successfully",
    )
    service_id: UUID = Field(
        ...,
        description="The registered service ID",
    )
    error: str | None = Field(
        default=None,
        description="Sanitized error message if registration failed",
    )
    duration_ms: float = Field(
        default=0.0,
        description="Time taken for the operation in milliseconds",
        ge=0.0,
    )
    backend_type: str = Field(
        default="unknown",
        description="The backend type that handled the registration",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for tracing",
    )


__all__ = ["ModelRegistrationResult"]
