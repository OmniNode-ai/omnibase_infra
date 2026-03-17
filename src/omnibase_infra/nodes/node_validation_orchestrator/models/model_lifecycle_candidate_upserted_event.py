# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Event model for validation lifecycle candidate tier transition.

Field contract pinned to omnidash ``ValidationCandidateUpsertedSchema`` in
``shared/validation-types.ts``.

Reference: OMN-5184 (Dashboard Data Pipeline Gaps), Batch B
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelLifecycleCandidateUpsertedEvent(BaseModel):
    """Emitted when a validation candidate's lifecycle tier changes.

    Topic: ``onex.evt.validation.lifecycle-candidate-upserted.v1``
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    event_type: Literal["ValidationCandidateUpserted"] = "ValidationCandidateUpserted"
    candidate_id: str = Field(min_length=1)
    rule_name: str = Field(
        min_length=1
    )  # pattern-ok: pinned to omnidash ValidationCandidateUpsertedSchema
    rule_id: str = Field(min_length=1)
    tier: Literal["observed", "suggested", "shadow_apply", "promoted", "default"] = (
        Field(description="Lifecycle tier (1-5 mapped to names)")
    )
    status: Literal["pending", "pass", "fail", "quarantine"] = Field(
        description="Current candidate status"
    )
    source_repo: str = Field(min_length=1)
    entered_tier_at: datetime = Field(
        description="When the candidate entered the current tier"
    )
    last_validated_at: datetime = Field(
        description="When the candidate was last validated"
    )
    pass_streak: int = Field(ge=0)
    fail_streak: int = Field(ge=0)
    total_runs: int = Field(ge=0)
    timestamp: datetime = Field(description="ISO-8601 UTC timestamp")

    @field_validator("timestamp", "entered_tier_at", "last_validated_at")
    @classmethod
    def validate_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return v


__all__ = ["ModelLifecycleCandidateUpsertedEvent"]
