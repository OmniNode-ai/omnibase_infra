# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Whether one pull request is exempt from a lab proof, and why.

Ticket: OMN-19565
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.lab_proof.enum_lab_proof_exempt_class import (
    EnumLabProofExemptClass,
)
from omnibase_infra.lab_proof.enum_lab_proof_kind import EnumLabProofKind


class ModelLabProofExemptionDecision(BaseModel):
    """The classifier's answer. ``exempt_class`` is set exactly when exempt.

    ``proof_kind`` is ``exempt`` when exempt, ``script_replay`` for a
    workflow-only (``.github/``-only) diff, and None when the profile's variant
    match decides.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str
    exempt: bool
    exempt_class: EnumLabProofExemptClass | None = None
    proof_kind: EnumLabProofKind | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> ModelLabProofExemptionDecision:
        if self.exempt != (self.proof_kind is EnumLabProofKind.EXEMPT):
            raise ValueError(
                f"repo {self.repo}: exempt is true exactly when proof_kind is exempt"
            )
        if self.exempt != (self.exempt_class is not None):
            raise ValueError(
                f"repo {self.repo}: exempt_class is set exactly when exempt"
            )
        return self


__all__ = ["ModelLabProofExemptionDecision"]
