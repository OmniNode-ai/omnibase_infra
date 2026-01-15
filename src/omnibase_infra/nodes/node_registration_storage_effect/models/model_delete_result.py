# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Delete Result Model for Registration Storage Operations.

This module provides ModelDeleteResult, representing the result of a
delete operation against registration storage.

Architecture:
    ModelDeleteResult captures:
    - success: Whether the operation completed successfully
    - node_id: ID of the record that was targeted for deletion
    - deleted: Whether a record was actually deleted (vs not found)
    - error: Error message if operation failed
    - duration_ms: Time taken for the operation
    - backend_type: The backend that handled the delete
    - correlation_id: Correlation ID for distributed tracing

    This model replaces primitive bool returns with a strongly-typed
    result that provides richer diagnostics and traceability.

Related:
    - ModelRegistrationRecord: Record that was deleted
    - ProtocolRegistrationStorageHandler: Protocol that produces results
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelDeleteResult(BaseModel):
    """Result model for registration storage delete operations.

    Indicates the outcome of a delete operation. Success indicates the
    operation completed without errors; deleted indicates whether a
    record was actually found and removed.

    Immutability:
        This model uses frozen=True to ensure results are immutable
        once created, enabling safe sharing and logging.

    Attributes:
        success: Whether the operation completed successfully (no errors).
        node_id: UUID of the registration record targeted for deletion.
        deleted: Whether a record was actually deleted (False if not found).
        error: Error message if success is False (sanitized).
        duration_ms: Time taken for the operation in milliseconds.
        backend_type: The backend that handled the delete.
        correlation_id: Correlation ID for request tracing.

    Example (record deleted):
        >>> result = ModelDeleteResult(
        ...     success=True,
        ...     node_id=some_uuid,
        ...     deleted=True,
        ...     duration_ms=12.5,
        ...     backend_type="postgresql",
        ...     correlation_id=correlation_id,
        ... )
        >>> result.was_deleted()
        True

    Example (record not found):
        >>> result = ModelDeleteResult(
        ...     success=True,
        ...     node_id=some_uuid,
        ...     deleted=False,
        ...     duration_ms=8.3,
        ...     backend_type="postgresql",
        ...     correlation_id=correlation_id,
        ... )
        >>> result.was_deleted()
        False

    Example (operation failed):
        >>> result = ModelDeleteResult(
        ...     success=False,
        ...     node_id=some_uuid,
        ...     deleted=False,
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
        description="Whether the operation completed successfully (no errors)",
    )
    node_id: UUID = Field(
        ...,
        description="UUID of the registration record targeted for deletion",
    )
    deleted: bool = Field(
        ...,
        description="Whether a record was actually deleted (False if not found)",
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
        description="The backend type that handled the delete",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for request tracing",
    )

    def was_deleted(self) -> bool:
        """Check if this result represents a successful deletion.

        Returns:
            True if operation succeeded and a record was deleted.
        """
        return self.success and self.deleted


__all__ = ["ModelDeleteResult"]
