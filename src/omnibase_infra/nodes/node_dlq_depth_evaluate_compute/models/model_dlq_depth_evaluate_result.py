# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Run-level result of the DLQ depth/arrival evaluation (OMN-16769)."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.models.enum_dlq_depth_verdict import (
    EnumDlqDepthVerdict,
)
from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.models.model_dlq_topic_verdict import (
    ModelDlqTopicVerdict,
)


class ModelDlqDepthEvaluateResult(BaseModel):
    """Run-level outcome: the full histogram plus the alert decision."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(...)
    evaluated_at: datetime = Field(...)
    window_seconds: int = Field(..., ge=60)

    verdicts: tuple[ModelDlqTopicVerdict, ...] = Field(
        default=(),
        description=(
            "Every observed topic, ordered by arrivals_in_window descending then "
            "topic name — the histogram the operator reads, worst first."
        ),
    )

    topics_observed: int = Field(default=0, ge=0)
    topics_alerting: int = Field(default=0, ge=0)
    total_arrivals_in_window: int = Field(default=0, ge=0)
    total_retained_depth: int = Field(default=0, ge=0)

    alert_triggered: bool = Field(
        default=False,
        description=(
            "True when at least one topic breached its bound. The scheduled "
            "workflow maps this to a non-zero exit so the run goes RED — a red "
            "run is the alert surface, not a side effect of one."
        ),
    )

    @property
    def alerting_verdicts(self) -> tuple[ModelDlqTopicVerdict, ...]:
        """Only the breaching rows, in the same worst-first order."""
        return tuple(
            verdict
            for verdict in self.verdicts
            if verdict.verdict is not EnumDlqDepthVerdict.OK
        )


__all__ = ["ModelDlqDepthEvaluateResult"]
