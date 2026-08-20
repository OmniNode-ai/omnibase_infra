# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-ticket outcome record for the evidence autoclose sweep."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)


class ModelEvidenceAutocloseOutcome(BaseModel):
    """One companion-PR / bound-ticket pair's terminal sweep decision."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    ticket_id: str = Field(
        default="", description="Bound Linear ticket id, e.g. OMN-16106."
    )
    companion_pr_number: int = Field(
        default=0, description="Merged OCC companion PR number."
    )
    companion_pr_url: str = Field(
        default="", description="Merged OCC companion PR URL."
    )
    decision: EnumEvidenceAutocloseDecision = Field(
        ..., description="Terminal classification for this pair."
    )
    reason: str = Field(default="", description="Human-readable explanation.")
    dod_verify_total_checks: int = Field(default=0, ge=0)
    dod_verify_verified_count: int = Field(default=0, ge=0)
    dod_verify_failed_count: int = Field(default=0, ge=0)
    linear_comment_posted: bool = Field(
        default=False, description="Whether an audit/gap comment was posted."
    )
    applied: bool = Field(
        default=False,
        description="True only when a real Linear mutation was made (apply=True run).",
    )


__all__ = ["ModelEvidenceAutocloseOutcome"]
