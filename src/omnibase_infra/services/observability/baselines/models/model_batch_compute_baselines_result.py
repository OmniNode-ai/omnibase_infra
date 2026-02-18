# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Result model for the baselines batch computation pipeline.

Tracks per-table row counts and any phase errors from a single
ServiceBatchComputeBaselines.compute_and_persist() run.

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

    model_config = ConfigDict(frozen=True, extra="forbid")

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
    completed_at: datetime | None = Field(
        default=None,
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
