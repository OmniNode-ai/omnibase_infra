# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Storage Result Model.

This module provides the ModelStorageResult class representing the outcome
of a storage query operation.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.handlers.registration_storage.models.model_registration_record import (
    ModelRegistrationRecord,
)


class ModelStorageResult(BaseModel):
    """Result of a storage query operation.

    Contains the list of registration records matching a query along with
    operation metadata. Immutable once created.

    Attributes:
        success: Whether the query completed successfully.
        records: List of registration records matching the query.
        total_count: Total number of matching records (for pagination).
        error: Sanitized error message if query failed.
        duration_ms: Time taken for the operation in milliseconds.
        backend_type: The backend that handled the query.
        correlation_id: Correlation ID for tracing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: bool = Field(
        ...,
        description="Whether the query completed successfully",
    )
    records: tuple[ModelRegistrationRecord, ...] = Field(
        default=(),
        description="List of registration records matching the query",
    )
    total_count: int = Field(
        default=0,
        description="Total number of matching records (for pagination)",
        ge=0,
    )
    error: str | None = Field(
        default=None,
        description="Sanitized error message if query failed",
    )
    duration_ms: float = Field(
        default=0.0,
        description="Time taken for the operation in milliseconds",
        ge=0.0,
    )
    backend_type: str = Field(
        default="unknown",
        description="The backend type that handled the query",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for tracing",
    )


__all__ = ["ModelStorageResult"]
