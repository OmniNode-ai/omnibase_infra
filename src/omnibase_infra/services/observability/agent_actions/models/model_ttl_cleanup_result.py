# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""TTL Cleanup Result Model.

This module defines the result model for a single TTL cleanup run.
Each run iterates over all configured tables and deletes rows older
than the retention period. The result tracks per-table deletion counts
and timing metrics for observability.

Related Tickets:
    - OMN-1759: Implement 30-day TTL cleanup for observability tables

Example:
    >>> from datetime import datetime, UTC
    >>> from uuid import uuid4
    >>> result = ModelTTLCleanupResult(
    ...     correlation_id=uuid4(),
    ...     started_at=datetime.now(UTC),
    ...     completed_at=datetime.now(UTC),
    ...     tables_cleaned=tuple(
    ...         {"agent_actions": 150, "agent_routing_decisions": 0}.items()
    ...     ),
    ...     total_rows_deleted=150,
    ...     duration_ms=1234,
    ... )
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelTTLCleanupResult(BaseModel):
    """Result of a single TTL cleanup run.

    Tracks per-table deletion counts and timing metrics for observability
    and health monitoring.

    Attributes:
        correlation_id: Unique identifier for this cleanup run.
        started_at: Timestamp when the cleanup run started.
        completed_at: Timestamp when the cleanup run completed.
        tables_cleaned: Immutable pairs of (table_name, rows_deleted).
        total_rows_deleted: Sum of all rows deleted across all tables.
        duration_ms: Total duration of the cleanup run in milliseconds.
        errors: Immutable pairs of (table_name, error_message) for failed tables.

    Example:
        >>> result = ModelTTLCleanupResult(
        ...     correlation_id=uuid4(),
        ...     started_at=datetime.now(UTC),
        ...     completed_at=datetime.now(UTC),
        ...     tables_cleaned=tuple({"agent_actions": 500}.items()),
        ...     total_rows_deleted=500,
        ...     duration_ms=2345,
        ... )
        >>> if result:
        ...     print(f"Deleted {result.total_rows_deleted} rows")
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    correlation_id: UUID = Field(
        ...,
        description="Unique identifier for this cleanup run.",
    )
    started_at: datetime = Field(
        ...,
        description="Timestamp when the cleanup run started.",
    )
    completed_at: datetime = Field(
        ...,
        description="Timestamp when the cleanup run completed.",
    )
    tables_cleaned: tuple[tuple[str, int], ...] = Field(
        default_factory=tuple,
        description="Immutable pairs of (table_name, rows_deleted) in this run.",
    )
    total_rows_deleted: int = Field(
        default=0,
        ge=0,
        description="Sum of all rows deleted across all tables.",
    )
    duration_ms: int = Field(
        default=0,
        ge=0,
        description="Total duration of the cleanup run in milliseconds.",
    )
    errors: tuple[tuple[str, str], ...] = Field(
        default_factory=tuple,
        description="Immutable pairs of (table_name, error_message) for failed tables.",
    )

    def __bool__(self) -> bool:
        """Allow using result in boolean context.

        Warning:
            **Non-standard __bool__ behavior**: Returns ``True`` when
            at least one row was deleted. Differs from typical Pydantic
            behavior where any model instance is truthy.
        """
        return self.total_rows_deleted > 0


__all__ = ["ModelTTLCleanupResult"]
