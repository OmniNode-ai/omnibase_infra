# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Storage Result Model for Registration Storage Operations.

This module provides ModelStorageResult, representing the result of a
query operation against registration storage.

Architecture:
    ModelStorageResult contains:
    - records: List of matching registration records
    - total_count: Total number of matching records (for pagination)
    - query_metadata: Optional metadata about query execution

    This model supports pagination - total_count may exceed len(records)
    when limit/offset pagination is used.

Related:
    - ModelStorageQuery: Query model that produces these results
    - ModelRegistrationRecord: Record type contained in results
    - ProtocolRegistrationStorageHandler: Protocol that produces results
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .model_registration_record import ModelRegistrationRecord


class ModelStorageResult(BaseModel):
    """Result model for registration storage query operations.

    Contains the list of matching records and pagination metadata.
    Supports efficient pagination by providing total_count separately
    from the actual records returned.

    Immutability:
        This model uses frozen=True to ensure results are immutable
        once created, enabling safe sharing and caching.

    Attributes:
        success: Whether the query completed successfully.
        records: List of matching registration records.
        total_count: Total number of records matching the query.
        error: Error message if query failed (sanitized).
        duration_ms: Time taken for the operation in milliseconds.
        backend_type: The backend that handled the query.
        correlation_id: Correlation ID for request tracing.

    Example:
        >>> # Result with pagination info
        >>> result = ModelStorageResult(
        ...     success=True,
        ...     records=(record1, record2),
        ...     total_count=150,  # More records exist
        ...     duration_ms=45.2,
        ...     backend_type="postgresql",
        ...     correlation_id=correlation_id,
        ... )
        >>> result.has_more_records()
        True
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    success: bool = Field(
        ...,
        description="Whether the query completed successfully",
    )
    records: tuple[ModelRegistrationRecord, ...] = Field(
        default_factory=tuple,
        description="List of matching registration records",
    )
    total_count: int = Field(
        default=0,
        description="Total number of records matching the query (for pagination)",
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
        description="Correlation ID for request tracing",
    )

    @model_validator(mode="after")
    def validate_count_consistency(self) -> ModelStorageResult:
        """Validate that total_count is consistent with records.

        total_count should be >= len(records) since pagination may
        return a subset of matching records.

        Returns:
            The validated result model.

        Raises:
            ValueError: If total_count < len(records).
        """
        if self.total_count < len(self.records):
            raise ValueError(
                f"total_count ({self.total_count}) cannot be less than "
                f"number of returned records ({len(self.records)})"
            )
        return self

    def has_more_records(self) -> bool:
        """Check if there are more records beyond the returned set.

        Returns:
            True if total_count > len(records), indicating more records exist.
        """
        return self.total_count > len(self.records)

    def is_empty(self) -> bool:
        """Check if the result contains no records.

        Returns:
            True if no records were returned.
        """
        return len(self.records) == 0


__all__ = ["ModelStorageResult"]
