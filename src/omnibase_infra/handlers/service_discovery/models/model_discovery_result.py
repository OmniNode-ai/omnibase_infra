# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Discovery Result Model.

This module provides the ModelDiscoveryResult class representing the outcome
of a service discovery query.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.handlers.service_discovery.models.model_service_info import (
    ModelServiceInfo,
)


class ModelDiscoveryResult(BaseModel):
    """Result of a service discovery query.

    Contains the list of services matching a discovery query along with
    operation metadata. Immutable once created.

    Attributes:
        success: Whether the discovery query completed successfully.
        services: List of discovered services matching the query.
        error: Sanitized error message if discovery failed.
        duration_ms: Time taken for the operation in milliseconds.
        backend_type: The backend that handled the discovery.
        correlation_id: Correlation ID for tracing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    success: bool = Field(
        ...,
        description="Whether the discovery query completed successfully",
    )
    services: tuple[ModelServiceInfo, ...] = Field(
        default=(),
        description="List of discovered services matching the query",
    )
    error: str | None = Field(
        default=None,
        description="Sanitized error message if discovery failed",
    )
    duration_ms: float = Field(
        default=0.0,
        description="Time taken for the operation in milliseconds",
        ge=0.0,
    )
    backend_type: str = Field(
        default="unknown",
        description="The backend type that handled the discovery",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for tracing",
    )


__all__ = ["ModelDiscoveryResult"]
