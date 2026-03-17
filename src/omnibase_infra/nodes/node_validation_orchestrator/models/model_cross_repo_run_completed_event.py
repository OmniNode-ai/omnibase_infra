# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Event model for cross-repo validation run completion.

Field contract pinned to omnidash ``ValidationRunCompletedSchema`` in
``shared/validation-types.ts``.

Reference: OMN-5184 (Dashboard Data Pipeline Gaps), Batch B
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelCrossRepoRunCompletedEvent(BaseModel):
    """Emitted after all repos have been checked in a validation run.

    Topic: ``onex.evt.validation.cross-repo-run-completed.v1``
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    event_type: Literal["ValidationRunCompleted"] = "ValidationRunCompleted"
    run_id: str = Field(min_length=1)
    status: Literal["passed", "failed", "error"] = Field(description="Final run status")
    total_violations: int = Field(ge=0)
    violations_by_severity: dict[str, int] | None = Field(
        default=None, description="Violation counts keyed by severity level"
    )
    duration_ms: int = Field(ge=0, description="Total run duration in milliseconds")
    timestamp: datetime = Field(description="ISO-8601 UTC timestamp")

    @field_validator("timestamp")
    @classmethod
    def validate_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return v


__all__ = ["ModelCrossRepoRunCompletedEvent"]
