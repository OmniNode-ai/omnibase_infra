# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Shared validation pipeline event -- adjudication requested.

Published by the validation orchestrator once the executor's checks are in;
consumed by the validation adjudicator.

Ticket: OMN-21056
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.validation.model_check_result import ModelCheckResult


class ModelValidationAdjudicationRequested(BaseModel):
    """Request to adjudicate the check results of one validation plan.

    Attributes:
        plan_id: Validation plan that was executed.
        candidate_id: Pattern candidate under validation.
        score_threshold: Minimum score for a PASS verdict.
        check_results: Results of every check in the plan.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    plan_id: UUID = Field(..., description="Validation plan that was executed.")
    candidate_id: UUID = Field(..., description="Pattern candidate under validation.")
    score_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Minimum score for PASS verdict."
    )
    check_results: tuple[ModelCheckResult, ...] = Field(
        default_factory=tuple, description="Results of every check in the plan."
    )


__all__: list[str] = ["ModelValidationAdjudicationRequested"]
