# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One ticket's outcome from the In Progress demotion leg (OMN-18489)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_in_progress_demotion_decision import (
    EnumInProgressDemotionDecision,
)


class ModelInProgressDemotionOutcome(BaseModel):
    """The decision and its reason for one In Progress ticket."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    ticket_identifier: str = Field(..., description="Linear identifier, e.g. OMN-1234.")
    decision: EnumInProgressDemotionDecision = Field(...)
    reason: str = Field(
        ...,
        description="Why this decision: the condition that withheld it, or the resolutions that cleared it.",
    )


__all__ = ["ModelInProgressDemotionOutcome"]
