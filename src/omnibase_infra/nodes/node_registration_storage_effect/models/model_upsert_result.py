# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Upsert Result Model for Registration Storage Operations.

This module provides ModelUpsertResult, representing the result of an
insert or update operation against registration storage.

Architecture:
    ModelUpsertResult captures:
    - success: Whether the operation completed successfully
    - node_id: ID of the affected record
    - operation: Whether it was an "insert" or "update"
    - error: Error message if operation failed

    This model supports idempotent upsert operations where the same
    request may result in either insert or update depending on
    existing data.

Related:
    - ModelRegistrationRecord: Record that was inserted/updated
    - ProtocolRegistrationStorageHandler: Protocol that produces results
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelUpsertResult(BaseModel):
    """Result model for registration storage upsert operations.

    Indicates the outcome of an insert or update operation.
    Success indicates the record was successfully stored; operation
    indicates whether it was a new insert or an update to existing.

    Immutability:
        This model uses frozen=True to ensure results are immutable
        once created, enabling safe sharing and logging.

    Attributes:
        success: Whether the operation completed successfully.
        node_id: UUID of the affected registration record.
        operation: Type of operation performed ("insert" or "update").
        error: Error message if success is False (sanitized).
        duration_ms: Time taken for the operation in milliseconds.
        backend_type: The backend that handled the upsert.
        correlation_id: Correlation ID for request tracing.

    Example (success):
        >>> result = ModelUpsertResult(
        ...     success=True,
        ...     node_id=some_uuid,
        ...     operation="insert",
        ...     duration_ms=45.2,
        ...     backend_type="postgresql",
        ...     correlation_id=correlation_id,
        ... )
        >>> result.was_inserted()
        True

    Example (failure):
        >>> result = ModelUpsertResult(
        ...     success=False,
        ...     node_id=some_uuid,
        ...     operation="update",
        ...     error="Connection timeout to database",
        ...     duration_ms=5000.0,
        ...     backend_type="postgresql",
        ...     correlation_id=correlation_id,
        ... )
        >>> result.success
        False
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    success: bool = Field(
        ...,
        description="Whether the operation completed successfully",
    )
    node_id: UUID = Field(
        ...,
        description="UUID of the affected registration record",
    )
    operation: Literal["insert", "update"] = Field(
        ...,
        description="Type of operation performed",
    )
    error: str | None = Field(
        default=None,
        description="Error message if success is False (sanitized - no secrets)",
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
        description="Correlation ID for request tracing",
    )

    def was_inserted(self) -> bool:
        """Check if this result represents a new record insertion.

        Returns:
            True if operation was "insert" and success is True.
        """
        return self.success and self.operation == "insert"

    def was_updated(self) -> bool:
        """Check if this result represents an existing record update.

        Returns:
            True if operation was "update" and success is True.
        """
        return self.success and self.operation == "update"


__all__ = ["ModelUpsertResult"]
