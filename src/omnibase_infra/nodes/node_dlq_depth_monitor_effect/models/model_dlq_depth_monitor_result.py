# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Result model for the read-only DLQ depth probe (OMN-16769)."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.models.model_dlq_depth_evaluate_result import (
    ModelDlqDepthEvaluateResult,
)


class ModelDlqDepthMonitorResult(BaseModel):
    """Probe outcome: what was swept, plus the full evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(...)
    evaluated_at: datetime = Field(...)
    window_seconds: int = Field(..., ge=60)
    topics_matched: int = Field(
        default=0,
        ge=0,
        description="Topics matching the DLQ prefix on this broker.",
    )
    evaluation: ModelDlqDepthEvaluateResult = Field(
        ..., description="The per-topic histogram and alert decision."
    )

    @property
    def alert_triggered(self) -> bool:
        """Convenience passthrough — the run-gating decision."""
        return self.evaluation.alert_triggered


__all__ = ["ModelDlqDepthMonitorResult"]
