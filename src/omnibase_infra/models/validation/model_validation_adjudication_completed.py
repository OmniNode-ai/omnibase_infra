# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Shared validation pipeline event -- adjudication completed.

Published by the validation adjudicator with the verdict it reached; consumed
by the validation orchestrator, which publishes the result. The fields mirror
the adjudicator's own verdict model so the orchestrator does not import a
sibling node's models.

Ticket: OMN-21056
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumValidationVerdict


class ModelValidationAdjudicationCompleted(BaseModel):
    """Verdict reached by the adjudicator for one validation plan.

    Attributes:
        plan_id: Validation plan that was adjudicated.
        candidate_id: Pattern candidate that was validated.
        verdict: Final decision (PASS / FAIL / QUARANTINE).
        score: Weighted pass ratio in [0.0, 1.0].
        total_checks: Total number of checks executed.
        passed_checks: Number of checks that passed.
        failed_checks: Number of checks that failed.
        skipped_checks: Number of checks that were skipped.
        blocking_failures: Check codes that caused a FAIL verdict.
        quarantine_reasons: Human-readable reasons for QUARANTINE.
        adjudicated_at: Timestamp when the verdict was produced.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    plan_id: UUID = Field(..., description="Validation plan that was adjudicated.")
    candidate_id: UUID = Field(..., description="Pattern candidate that was validated.")
    verdict: EnumValidationVerdict = Field(
        ..., description="Final decision (PASS / FAIL / QUARANTINE)."
    )
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Weighted pass ratio in [0.0, 1.0]."
    )
    total_checks: int = Field(..., ge=0, description="Total number of checks executed.")
    passed_checks: int = Field(..., ge=0, description="Number of checks that passed.")
    failed_checks: int = Field(..., ge=0, description="Number of checks that failed.")
    skipped_checks: int = Field(
        ..., ge=0, description="Number of checks that were skipped."
    )
    blocking_failures: tuple[str, ...] = Field(
        default_factory=tuple, description="Check codes that caused a FAIL verdict."
    )
    quarantine_reasons: tuple[str, ...] = Field(
        default_factory=tuple, description="Human-readable reasons for QUARANTINE."
    )
    adjudicated_at: datetime = Field(
        ..., description="Timestamp when the verdict was produced."
    )


__all__: list[str] = ["ModelValidationAdjudicationCompleted"]
