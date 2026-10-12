# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Shared validation pipeline model -- request to adjudicate executor results.

Published by the validation orchestrator on
``onex.evt.platform.validation-adjudication-requested.v1`` once the executor has
reported ``checks-completed``; consumed by the validation adjudicator.

Ticket: OMN-21056
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.validation.model_check_result import (
    ModelCheckResult,
)


class ModelValidationAdjudicationRequested(BaseModel):
    """Check results handed to the adjudicator for a PASS/FAIL/QUARANTINE verdict.

    Attributes:
        plan_id: Validation plan the checks were executed for.
        candidate_id: Pattern candidate under validation.
        check_results: Individual check results reported by the executor.
        score_threshold: Minimum score for a PASS verdict (0.0-1.0).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    plan_id: UUID = Field(..., description="Reference to the validation plan.")
    candidate_id: UUID = Field(..., description="Reference to the pattern candidate.")
    check_results: tuple[ModelCheckResult, ...] = Field(
        default_factory=tuple, description="Individual check results."
    )
    score_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Minimum score for PASS verdict."
    )


__all__: list[str] = ["ModelValidationAdjudicationRequested"]
