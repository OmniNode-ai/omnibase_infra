# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The outcome of one board check execution.

Ticket: OMN-19930
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.lab_proof.enum_lab_proof_check import EnumLabProofCheck
from omnibase_infra.lab_proof.model_lab_proof_check_result import (
    ModelLabProofCheckResult,
)
from omnibase_infra.nodes.node_board_probe_effect.models.enum_board_check_id import (
    EnumBoardCheckId,
)
from omnibase_infra.nodes.node_board_probe_effect.models.enum_board_check_surface_class import (
    EnumBoardCheckSurfaceClass,
)
from omnibase_infra.nodes.node_board_probe_effect.models.enum_board_probe_outcome import (
    EnumBoardProbeOutcome,
)


class ModelBoardProbeResult(BaseModel):
    """One check, one subject, one outcome, and every reason behind it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    check_id: EnumBoardCheckId
    surface_class: EnumBoardCheckSurfaceClass
    subject: str = Field(min_length=1)
    outcome: EnumBoardProbeOutcome
    reasons: tuple[str, ...] = Field(min_length=1)
    evidence_items: tuple[str, ...] = ()
    observed_at: datetime

    @property
    def status(self) -> str:
        """``success`` only for PASS; ``failure`` for FAIL and INDETERMINATE.

        ``onex node`` classifies a handler result by its ``status``, so a run of
        this node exits non-zero unless the check passed.
        """
        return "success" if self.outcome is EnumBoardProbeOutcome.PASS else "failure"

    def as_lab_proof_check_result(self) -> ModelLabProofCheckResult:
        """The lab proof's view: only PASS passes, INDETERMINATE fails."""
        return ModelLabProofCheckResult(
            check=EnumLabProofCheck(self.check_id.value),
            passed=self.outcome is EnumBoardProbeOutcome.PASS,
            detail=f"{self.outcome.value}: " + "; ".join(self.reasons),
            evidence_items=self.evidence_items,
        )


__all__ = ["ModelBoardProbeResult"]
