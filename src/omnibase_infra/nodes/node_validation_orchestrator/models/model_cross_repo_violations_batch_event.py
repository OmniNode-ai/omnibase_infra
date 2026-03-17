# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Event model for cross-repo validation violations batch.

Field contract pinned to omnidash ``ValidationViolationsBatchSchema`` in
``shared/validation-types.ts``.

Reference: OMN-5184 (Dashboard Data Pipeline Gaps), Batch B
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.nodes.node_validation_orchestrator.models.model_violation import (
    ModelViolation,
)


class ModelCrossRepoViolationsBatchEvent(BaseModel):
    """Emitted per-batch as violations are found during validation.

    Topic: ``onex.evt.validation.cross-repo-violations-batch.v1``
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    event_type: Literal["ValidationViolationsBatch"] = "ValidationViolationsBatch"
    run_id: str = Field(min_length=1)
    violations: list[ModelViolation] = Field(
        description="Violations found in this batch"
    )
    batch_index: int = Field(
        ge=0, description="Sequential batch index for dedup on replay"
    )
    timestamp: datetime = Field(description="ISO-8601 UTC timestamp")

    @field_validator("timestamp")
    @classmethod
    def validate_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return v


__all__ = ["ModelCrossRepoViolationsBatchEvent"]
