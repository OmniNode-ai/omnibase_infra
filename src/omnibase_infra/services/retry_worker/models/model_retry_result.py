# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Retry result model for the RetryWorker service.

Tracks per-run metrics including successful retries, failures, and DLQ moves
for observability and health monitoring.

Related Tickets:
    - OMN-1454: Implement RetryWorker for subscription notification delivery
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelRetryResult(BaseModel):
    """Result of a single retry worker run.

    Captures metrics from one polling cycle: how many retries succeeded,
    failed, or were moved to the dead letter queue.

    Attributes:
        correlation_id: Unique identifier for this retry run.
        started_at: Timestamp when the retry run started.
        completed_at: Timestamp when the retry run completed.
        retries_attempted: Total number of retry attempts in this run.
        retries_succeeded: Number of retries that delivered successfully.
        retries_failed: Number of retries that failed again.
        moved_to_dlq: Number of attempts moved to DLQ (max retries exceeded).
        duration_ms: Total duration of the retry run in milliseconds.
        errors: Immutable pairs of (attempt_id, error_message) for failures.

    Example:
        >>> from datetime import datetime, UTC
        >>> from uuid import uuid4
        >>> result = ModelRetryResult(
        ...     correlation_id=uuid4(),
        ...     started_at=datetime.now(UTC),
        ...     completed_at=datetime.now(UTC),
        ...     retries_attempted=5,
        ...     retries_succeeded=3,
        ...     retries_failed=1,
        ...     moved_to_dlq=1,
        ...     duration_ms=234,
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    correlation_id: UUID = Field(
        ..., description="Unique identifier for this retry run."
    )
    started_at: datetime = Field(
        ..., description="Timestamp when the retry run started."
    )
    completed_at: datetime = Field(
        ..., description="Timestamp when the retry run completed."
    )
    retries_attempted: int = Field(
        default=0, ge=0, description="Total number of retry attempts in this run."
    )
    retries_succeeded: int = Field(
        default=0, ge=0, description="Number of retries that delivered successfully."
    )
    retries_failed: int = Field(
        default=0, ge=0, description="Number of retries that failed again."
    )
    moved_to_dlq: int = Field(
        default=0, ge=0, description="Number of attempts moved to DLQ."
    )
    duration_ms: int = Field(
        default=0, ge=0, description="Total duration of the retry run in milliseconds."
    )
    errors: tuple[tuple[str, str], ...] = Field(
        default_factory=tuple,
        description="Immutable pairs of (attempt_id, error_message) for failures.",
    )

    def __bool__(self) -> bool:
        """Allow using result in boolean context.

        Warning:
            **Non-standard __bool__ behavior**: Returns ``True`` when
            at least one retry was attempted. Differs from typical Pydantic
            behavior where any model instance is truthy.
        """
        return self.retries_attempted > 0


__all__ = ["ModelRetryResult"]
