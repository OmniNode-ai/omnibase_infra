# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Everything the verdict handler reads.

Ticket: OMN-19572
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.lab_proof.enum_lab_proof_check import EnumLabProofCheck
from omnibase_infra.lab_proof.model_lab_proof_plan import ModelLabProofPlan
from omnibase_infra.lab_proof.model_lab_proof_result import ModelLabProofResult
from omnibase_infra.lab_proof.model_lab_proof_run_report import (
    ModelLabProofRunReport,
)


class ModelLabProofVerdictRequest(BaseModel):
    """The plan, what its run observed, the row's mandatory checks, and an optional base control.

    ``base_result`` is the verdict of the same steps at the merge base. When a
    head run fails and the base run fails the same checks, the failure is
    dev-inherited, not the PR's (interim recipes common frame 8).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan: ModelLabProofPlan
    report: ModelLabProofRunReport
    mandatory_checks: tuple[EnumLabProofCheck, ...] = Field(min_length=1)
    base_result: ModelLabProofResult | None = None


__all__ = ["ModelLabProofVerdictRequest"]
