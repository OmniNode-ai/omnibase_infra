# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Upsert Result Model.

This module provides the ModelUpsertResult class representing the outcome
of an upsert operation on registration storage.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelUpsertResult(BaseModel):
    """Result of an upsert operation on registration storage.

    Captures the outcome of inserting or updating a registration record.
    Immutable once created.

    Attributes:
        success: Whether the upsert completed successfully.
        node_id: The node ID that was upserted.
        was_insert: True if this was an insert, False if update.
        error: Sanitized error message if upsert failed.
        duration_ms: Time taken for the operation in milliseconds.
        backend_type: The backend that handled the upsert.
        correlation_id: Correlation ID for tracing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: bool = Field(
        ...,
        description="Whether the upsert completed successfully",
    )
    node_id: UUID = Field(
        ...,
        description="The node ID that was upserted",
    )
    was_insert: bool = Field(
        default=False,
        description="True if this was an insert, False if update",
    )
    error: str | None = Field(
        default=None,
        description="Sanitized error message if upsert failed",
    )
    duration_ms: float = Field(
        default=0.0,
        description="Time taken for the operation in milliseconds",
        ge=0.0,
    )
    backend_type: str = Field(
        default="unknown",
        description="The backend type that handled the upsert",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for tracing",
    )


__all__ = ["ModelUpsertResult"]
