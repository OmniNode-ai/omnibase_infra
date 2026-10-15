# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result of one In Progress demotion pass (OMN-18489)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_in_progress_demotion_outcome import (
    ModelInProgressDemotionOutcome,
)


class ModelInProgressDemotionResult(BaseModel):
    """Counts and per-ticket outcomes of one demotion pass."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    window_hours: int = Field(..., gt=0)
    dry_run: bool = Field(...)
    candidates_scanned: int = Field(default=0, ge=0)
    tickets_demoted: int = Field(default=0, ge=0)
    tickets_would_demote: int = Field(default=0, ge=0)
    tickets_skipped: int = Field(default=0, ge=0)
    tickets_errored: int = Field(default=0, ge=0)
    outcomes: tuple[ModelInProgressDemotionOutcome, ...] = Field(default=())
    success: bool = Field(
        default=True,
        description="False when the In Progress enumeration itself failed; no ticket was judged.",
    )
    error_message: str = Field(default="")


__all__ = ["ModelInProgressDemotionResult"]
