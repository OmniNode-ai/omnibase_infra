# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Whether one pull request is exempt from a lab proof, and why.

Ticket: OMN-19565
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.lab_proof.enum_lab_proof_exempt_class import (
    EnumLabProofExemptClass,
)


class ModelLabProofExemptionDecision(BaseModel):
    """The classifier's answer. ``exempt_class`` is set exactly when exempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str
    exempt: bool
    exempt_class: EnumLabProofExemptClass | None = None
    reason: str = Field(min_length=1)


__all__ = ["ModelLabProofExemptionDecision"]
