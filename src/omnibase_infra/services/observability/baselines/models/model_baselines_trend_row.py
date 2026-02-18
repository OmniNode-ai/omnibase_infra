# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Row model for the baselines_trend table.

Represents a single (cohort, date) time-series data point for trend analysis.
Used by the /api/baselines/trend endpoint.

Related Tickets:
    - OMN-2305: Create baselines tables and populate treatment/control comparisons
"""

from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelBaselinesTrendRow(BaseModel):
    """One row from the baselines_trend table.

    Represents a single cohort's metrics for one day. Pairs of
    (treatment, control) rows for the same date form a trend data point
    for the dashboard.

    Attributes:
        id: Primary key UUID.
        trend_date: The date this row covers.
        cohort: A/B cohort: 'treatment' or 'control'.
        session_count: Number of sessions in this cohort/day.
        success_rate: Success rate for this cohort/day (0.0-1.0).
        avg_latency_ms: Average latency for this cohort/day.
        avg_cost_tokens: Average token cost for this cohort/day.
        roi_pct: ROI relative to control baseline for this day.
        computed_at: When this row was last computed.
        created_at: When this row was first created.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    id: UUID = Field(..., description="Primary key UUID.")
    trend_date: date = Field(..., description="The date this row covers.")
    cohort: str = Field(..., description="A/B cohort: 'treatment' or 'control'.")

    session_count: int = Field(
        default=0, ge=0, description="Number of sessions in this cohort/day."
    )
    success_rate: float | None = Field(
        default=None,
        description="Success rate for this cohort/day (0.0-1.0).",
    )
    avg_latency_ms: float | None = Field(
        default=None,
        description="Average latency for this cohort/day.",
    )
    avg_cost_tokens: float | None = Field(
        default=None,
        description="Average token cost for this cohort/day.",
    )
    roi_pct: float | None = Field(
        default=None,
        description="ROI relative to control baseline for this day.",
    )

    computed_at: datetime = Field(..., description="When this row was last computed.")
    created_at: datetime = Field(..., description="When this row was first created.")


__all__: list[str] = ["ModelBaselinesTrendRow"]
