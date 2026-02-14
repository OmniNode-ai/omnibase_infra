# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Result of a single A/B run variant (baseline or candidate).

Captures both cost metrics and outcome metrics for one variant
of the A/B comparison.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumRunVariant
from omnibase_infra.models.baseline.model_cost_metrics import ModelCostMetrics
from omnibase_infra.models.baseline.model_outcome_metrics import ModelOutcomeMetrics


class ModelBaselineRunResult(BaseModel):
    """Result of a single run variant in an A/B comparison.

    Pairs the run variant label (BASELINE or CANDIDATE) with its
    captured cost and outcome metrics.

    Attributes:
        run_id: Unique identifier for this run.
        variant: Whether this is the BASELINE or CANDIDATE run.
        correlation_id: Correlation ID for distributed tracing.
        cost_metrics: Token, time, and retry metrics for this run.
        outcome_metrics: Pass/fail, flake, and review metrics.
        started_at: Timestamp when the run started.
        completed_at: Timestamp when the run completed.
        error: Error message (empty if success).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    run_id: UUID = Field(
        ...,
        description="Unique identifier for this run.",
    )
    variant: EnumRunVariant = Field(
        ...,
        description="Whether this is the BASELINE or CANDIDATE run.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing.",
    )
    cost_metrics: ModelCostMetrics = Field(
        ...,
        description="Token, time, and retry metrics for this run.",
    )
    outcome_metrics: ModelOutcomeMetrics = Field(
        ...,
        description="Pass/fail, flake, and review metrics.",
    )
    started_at: datetime = Field(
        ...,
        description="Timestamp when the run started.",
    )
    completed_at: datetime = Field(
        ...,
        description="Timestamp when the run completed.",
    )
    error: str = Field(
        default="",
        description="Error message (empty if success).",
    )

    def __bool__(self) -> bool:
        """Allow using result in boolean context.

        Warning:
            **Non-standard __bool__ behavior**: Returns ``True`` only when
            no error is present. Differs from typical Pydantic behavior.
        """
        return not self.error


__all__: list[str] = ["ModelBaselineRunResult"]
