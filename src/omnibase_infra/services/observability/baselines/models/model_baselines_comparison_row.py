# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Row model for the baselines_comparisons table.

Represents a single daily A/B comparison between treatment and control groups.
Used as both the DB row representation and the API response model for
``/api/baselines/comparisons`` and ``/api/baselines/summary``.

``from_attributes=True`` is set on the model config so that instances can be
constructed directly from asyncpg ``Record`` objects (or any ORM row) returned
by the repository layer — no intermediate ``dict`` conversion is required.
``extra="forbid"`` ensures that any unexpected column returned by a query
raises a ``ValidationError`` immediately rather than silently being dropped.

Related Tickets:
    - OMN-2305: Create baselines tables and populate treatment/control comparisons
"""

from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelBaselinesComparisonRow(BaseModel):
    """One row from the baselines_comparisons table.

    Represents a daily A/B comparison between treatment (sessions with
    high-confidence pattern injection) and control (sessions without or
    with low-confidence injection) groups.

    Attributes:
        id: Primary key UUID.
        comparison_date: The date this comparison covers.
        period_label: Human-readable period label (e.g. "2026-02-18").
        treatment_sessions: Count of treatment group sessions.
        treatment_success_rate: Success rate in treatment group (0.0-1.0).
        treatment_avg_latency_ms: Average latency for treatment sessions.
        treatment_avg_cost_tokens: Average token usage in treatment group.
        treatment_total_tokens: Total tokens consumed by treatment sessions.
        control_sessions: Count of control group sessions.
        control_success_rate: Success rate in control group (0.0-1.0).
        control_avg_latency_ms: Average latency for control sessions.
        control_avg_cost_tokens: Average token usage in control group.
        control_total_tokens: Total tokens consumed by control sessions.
        roi_pct: ROI percentage. Positive = treatment outperforms control.
        latency_improvement_pct: Latency improvement %. Positive = treatment faster.
        cost_improvement_pct: Cost improvement %. Positive = treatment cheaper.
        sample_size: Total sessions (treatment + control).
        computed_at: When this row was last computed.
        created_at: When this row was first created.
        updated_at: When this row was last updated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    id: UUID = Field(..., description="Primary key UUID.")
    comparison_date: date = Field(..., description="The date this comparison covers.")
    period_label: str | None = Field(
        default=None, description="Human-readable period label."
    )

    # Treatment group
    treatment_sessions: int = Field(
        default=0, ge=0, description="Count of treatment group sessions."
    )
    treatment_success_rate: float | None = Field(
        default=None,
        description="Success rate in treatment group (0.0-1.0).",
    )
    treatment_avg_latency_ms: float | None = Field(
        default=None,
        description="Average latency for treatment sessions.",
    )
    treatment_avg_cost_tokens: float | None = Field(
        default=None,
        description="Average token usage in treatment group.",
    )
    treatment_total_tokens: int = Field(
        default=0, ge=0, description="Total tokens consumed by treatment sessions."
    )

    # Control group
    control_sessions: int = Field(
        default=0, ge=0, description="Count of control group sessions."
    )
    control_success_rate: float | None = Field(
        default=None,
        description="Success rate in control group (0.0-1.0).",
    )
    control_avg_latency_ms: float | None = Field(
        default=None,
        description="Average latency for control sessions.",
    )
    control_avg_cost_tokens: float | None = Field(
        default=None,
        description="Average token usage in control group.",
    )
    control_total_tokens: int = Field(
        default=0, ge=0, description="Total tokens consumed by control sessions."
    )

    # Derived ROI metrics
    roi_pct: float | None = Field(
        default=None,
        description="ROI percentage. Positive = treatment outperforms control.",
    )
    latency_improvement_pct: float | None = Field(
        default=None,
        description="Latency improvement %. Positive = treatment faster.",
    )
    cost_improvement_pct: float | None = Field(
        default=None,
        description="Cost improvement %. Positive = treatment cheaper.",
    )
    sample_size: int = Field(
        default=0, ge=0, description="Total sessions (treatment + control)."
    )

    # Timestamps
    computed_at: datetime = Field(..., description="When this row was last computed.")
    created_at: datetime = Field(..., description="When this row was first created.")
    updated_at: datetime = Field(..., description="When this row was last updated.")


__all__: list[str] = ["ModelBaselinesComparisonRow"]
