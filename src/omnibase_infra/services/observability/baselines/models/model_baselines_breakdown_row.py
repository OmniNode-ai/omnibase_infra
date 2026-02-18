# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Row model for the baselines_breakdown table.

Represents per-pattern performance breakdown for the /api/baselines/breakdown
endpoint. One row per selected_agent (treated as a pattern proxy).

Confidence Field Semantics:
    The ``confidence`` field is a **proxy** for sample sufficiency, not a
    statistical confidence interval. It is set to ``treatment_success_rate``
    when ``sample_count >= 20``, and ``NULL`` below that threshold. A non-NULL
    value signals that the sample size is large enough to treat the success
    rate as meaningful; it does not represent a probability or significance level.

Construction from asyncpg Records:
    ``from_attributes=True`` in the model config enables constructing
    instances directly from asyncpg ``Record`` objects (which expose column
    values as attributes), without an intermediate ``dict(**row)`` conversion.

Related Tickets:
    - OMN-2305: Create baselines tables and populate treatment/control comparisons
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelBaselinesBreakdownRow(BaseModel):
    """One row from the baselines_breakdown table.

    Represents a single pattern's treatment vs control performance.
    The pattern_id is deterministically derived from md5(selected_agent)::uuid
    for stable cross-run identity.

    Attributes:
        id: Primary key UUID.
        pattern_id: Deterministic UUID from md5(selected_agent)::uuid.
        pattern_label: Human-readable label (selected_agent name).
        treatment_success_rate: Success rate in treatment cohort.
        control_success_rate: Success rate in control cohort.
        roi_pct: Pattern-specific ROI percentage.
        sample_count: Total sessions for this pattern.
        treatment_count: Treatment group sessions for this pattern.
        control_count: Control group sessions for this pattern.
        confidence: Confidence proxy: treatment_success_rate when sample_count >= 20,
            NULL below the minimum sample threshold. This is not a statistical
            confidence interval; it signals whether the sample size is sufficient
            to treat the success rate as meaningful.
        computed_at: When this row was last computed.
        created_at: When this row was first created.
        updated_at: When this row was last updated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    id: UUID = Field(..., description="Primary key UUID.")
    pattern_id: UUID = Field(
        ..., description="Deterministic UUID from md5(selected_agent)::uuid."
    )
    pattern_label: str | None = Field(
        default=None, description="Human-readable label (selected_agent name)."
    )

    treatment_success_rate: float | None = Field(
        default=None,
        description="Success rate in treatment cohort.",
    )
    control_success_rate: float | None = Field(
        default=None,
        description="Success rate in control cohort.",
    )
    roi_pct: float | None = Field(
        default=None,
        description="Pattern-specific ROI percentage.",
    )
    sample_count: int = Field(
        default=0, ge=0, description="Total sessions for this pattern."
    )
    treatment_count: int = Field(
        default=0, ge=0, description="Treatment group sessions for this pattern."
    )
    control_count: int = Field(
        default=0, ge=0, description="Control group sessions for this pattern."
    )
    confidence: float | None = Field(
        default=None,
        description=(
            "Confidence proxy: treatment_success_rate when sample_count >= 20, "
            "NULL below the minimum sample threshold. This is not a statistical "
            "confidence interval; it signals whether the sample size is sufficient "
            "to treat the success rate as meaningful."
        ),
    )

    computed_at: datetime = Field(..., description="When this row was last computed.")
    created_at: datetime = Field(..., description="When this row was first created.")
    updated_at: datetime = Field(..., description="When this row was last updated.")


__all__: list[str] = ["ModelBaselinesBreakdownRow"]
