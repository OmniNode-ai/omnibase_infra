# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Result model for the baselines batch computation pipeline.

This module defines ``ModelBatchComputeBaselinesResult``, the primary output
container returned by ``ServiceBatchComputeBaselines.compute_and_persist()``.
It captures row counts written to each of the three baselines tables
(``baselines_comparisons``, ``baselines_trend``, ``baselines_breakdown``) and
any error messages emitted by phases that failed non-fatally, allowing
downstream callers to inspect partial-success runs without catching exceptions.

The model is immutable (``frozen=True``) and ORM-compatible
(``from_attributes=True``) so it can be constructed directly from database
row mappings in tests and integration fixtures.

Related Tickets:
    - OMN-2305: Create baselines tables and populate treatment/control comparisons
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelBatchComputeBaselinesResult(BaseModel):
    """Result of a single baselines batch computation run.

    Tracks per-table row counts and phase errors. Individual phase
    failures are captured in ``errors`` rather than raised, allowing
    later phases to still execute.

    Attributes:
        comparisons_rows: Rows written to baselines_comparisons.
        trend_rows: Rows written to baselines_trend.
        breakdown_rows: Rows written to baselines_breakdown.
        errors: Tuple of error messages from failed phases.
        started_at: When the computation started.
        completed_at: When the computation completed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    comparisons_rows: int = Field(
        default=0, ge=0, description="Rows written to baselines_comparisons."
    )
    trend_rows: int = Field(
        default=0, ge=0, description="Rows written to baselines_trend."
    )
    breakdown_rows: int = Field(
        default=0, ge=0, description="Rows written to baselines_breakdown."
    )
    errors: tuple[str, ...] = Field(
        default_factory=tuple, description="Error messages from failed phases."
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the computation started.",
    )
    completed_at: datetime = Field(
        description="When the computation completed.",
    )

    @property
    def total_rows(self) -> int:
        """Total rows written across all three tables."""
        return self.comparisons_rows + self.trend_rows + self.breakdown_rows

    @property
    def has_errors(self) -> bool:
        """True if any phase encountered an error."""
        return len(self.errors) > 0


__all__: list[str] = ["ModelBatchComputeBaselinesResult"]
