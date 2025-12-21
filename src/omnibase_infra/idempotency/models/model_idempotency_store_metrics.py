# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Metrics model for idempotency store observability.

This module provides a Pydantic model for tracking operational metrics
of the idempotency store, enabling monitoring of:
- Duplicate detection rate (duplicates / total checks)
- Store error rate (failed checks / total checks)
- Cleanup effectiveness (records deleted per run)

Example:
    >>> from omnibase_infra.idempotency.models import ModelIdempotencyStoreMetrics
    >>>
    >>> metrics = ModelIdempotencyStoreMetrics(
    ...     total_checks=1000,
    ...     duplicate_count=50,
    ...     error_count=5,
    ... )
    >>> print(f"Duplicate rate: {metrics.duplicate_rate:.2%}")
    Duplicate rate: 5.00%
    >>> print(f"Error rate: {metrics.error_rate:.2%}")
    Error rate: 0.50%
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ModelIdempotencyStoreMetrics(BaseModel):
    """Metrics for idempotency store observability.

    Tracks operational statistics for monitoring store health and performance.
    All counters are monotonically increasing over the lifetime of the store
    instance, except for last_cleanup_deleted which is reset each cleanup.

    Attributes:
        total_checks: Total number of check_and_record calls.
        duplicate_count: Number of duplicates detected (check_and_record returned False).
        error_count: Number of failed check operations (exceptions raised).
        total_cleanup_deleted: Total records cleaned up across all cleanup runs.
        last_cleanup_deleted: Records deleted in the most recent cleanup run.
        last_cleanup_at: Timestamp of the most recent cleanup run.
    """

    total_checks: int = Field(
        default=0,
        ge=0,
        description="Total check_and_record calls",
    )
    duplicate_count: int = Field(
        default=0,
        ge=0,
        description="Number of duplicates detected",
    )
    error_count: int = Field(
        default=0,
        ge=0,
        description="Number of failed checks",
    )
    total_cleanup_deleted: int = Field(
        default=0,
        ge=0,
        description="Total records cleaned up across all runs",
    )
    last_cleanup_deleted: int = Field(
        default=0,
        ge=0,
        description="Records deleted in last cleanup",
    )
    last_cleanup_at: datetime | None = Field(
        default=None,
        description="Timestamp of last cleanup run",
    )

    @property
    def duplicate_rate(self) -> float:
        """Calculate duplicate detection rate.

        Returns:
            Ratio of duplicates to total checks (0.0 to 1.0).
            Returns 0.0 if no checks have been made.
        """
        if self.total_checks == 0:
            return 0.0
        return self.duplicate_count / self.total_checks

    @property
    def error_rate(self) -> float:
        """Calculate error rate.

        Returns:
            Ratio of errors to total checks (0.0 to 1.0).
            Returns 0.0 if no checks have been made.
        """
        if self.total_checks == 0:
            return 0.0
        return self.error_count / self.total_checks

    @property
    def success_count(self) -> int:
        """Calculate number of successful new message recordings.

        Returns:
            Number of check_and_record calls that returned True (new messages).
        """
        return self.total_checks - self.duplicate_count - self.error_count

    @property
    def success_rate(self) -> float:
        """Calculate success rate (new messages recorded / total checks).

        Returns:
            Ratio of successful new recordings to total checks (0.0 to 1.0).
            Returns 0.0 if no checks have been made.
        """
        if self.total_checks == 0:
            return 0.0
        return self.success_count / self.total_checks


__all__: list[str] = ["ModelIdempotencyStoreMetrics"]
