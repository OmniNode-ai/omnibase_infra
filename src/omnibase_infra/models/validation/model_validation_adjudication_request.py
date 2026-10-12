# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Shared validation pipeline model — adjudication request.

Published by the orchestrator on ``validation-adjudication-requested`` and
consumed by the adjudicator. It carries everything the verdict needs, so the
adjudicator holds no state between requests.

Ticket: OMN-21056
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.validation.model_check_result import ModelCheckResult


class ModelValidationAdjudicationRequest(BaseModel):
    """Request to the adjudicator to turn check results into a verdict.

    Attributes:
        plan_id: Reference to the validation plan that was executed.
        candidate_id: Reference to the pattern candidate.
        check_results: Every check result the executor produced.
        score_threshold: Minimum score for a PASS verdict (0.0-1.0).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    plan_id: UUID = Field(..., description="Reference to the validation plan.")
    candidate_id: UUID = Field(..., description="Reference to the pattern candidate.")
    check_results: tuple[ModelCheckResult, ...] = Field(
        default_factory=tuple, description="Individual check results to adjudicate."
    )
    score_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Minimum score for PASS verdict."
    )


__all__: list[str] = ["ModelValidationAdjudicationRequest"]
