# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Event model for cross-repo validation run start.

Field contract pinned to omnidash ``ValidationRunStartedSchema`` in
``shared/validation-types.ts``.

Reference: OMN-5184 (Dashboard Data Pipeline Gaps), Batch B
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelCrossRepoRunStartedEvent(BaseModel):
    """Emitted at the beginning of a cross-repo validation run.

    Topic: ``onex.evt.validation.cross-repo-run-started.v1``
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    event_type: Literal["ValidationRunStarted"] = "ValidationRunStarted"
    run_id: str = Field(min_length=1)  # pattern-ok: opaque run identifier, not a UUID
    repos: list[str] = Field(description="List of repository names being validated")
    validators: list[str] = Field(description="List of validator names to execute")
    triggered_by: str = Field(
        default="", description="User or system that initiated the run"
    )
    timestamp: datetime = Field(description="ISO-8601 UTC timestamp")

    @field_validator("timestamp")
    @classmethod
    def validate_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return v


__all__ = ["ModelCrossRepoRunStartedEvent"]
