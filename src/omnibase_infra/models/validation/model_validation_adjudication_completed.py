# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Orchestrator-side view of the adjudicator's verdict event.

The adjudicator publishes its verdict on
``onex.evt.platform.validation-adjudication-completed.v1``. The orchestrator owns
this consumer-side model so it never imports the adjudicator node's models; unknown
producer fields are ignored rather than rejected.

Ticket: OMN-21056
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumValidationVerdict


class ModelValidationAdjudicationCompleted(BaseModel):
    """Verdict reported by the validation adjudicator for one candidate."""

    model_config = ConfigDict(frozen=True, extra="ignore", from_attributes=True)

    candidate_id: UUID = Field(..., description="Candidate that was validated.")
    plan_id: UUID = Field(..., description="Validation plan used.")
    verdict: EnumValidationVerdict = Field(..., description="PASS / FAIL / QUARANTINE.")
    score: float = Field(..., ge=0.0, le=1.0, description="Weighted pass ratio.")
    total_checks: int = Field(..., ge=0, description="Total checks executed.")
    passed_checks: int = Field(..., ge=0, description="Checks that passed.")
    failed_checks: int = Field(..., ge=0, description="Checks that failed.")
    skipped_checks: int = Field(..., ge=0, description="Checks that were skipped.")
    blocking_failures: tuple[str, ...] = Field(
        default_factory=tuple, description="Check codes that caused a FAIL verdict."
    )
    quarantine_reasons: tuple[str, ...] = Field(
        default_factory=tuple, description="Reasons for a QUARANTINE verdict."
    )
    adjudicated_at: datetime = Field(..., description="When the verdict was produced.")


__all__: list[str] = ["ModelValidationAdjudicationCompleted"]
