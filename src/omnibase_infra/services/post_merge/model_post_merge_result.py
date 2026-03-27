# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pydantic model for aggregated post-merge check chain results.

Related Tickets:
    - OMN-6727: post-merge consumer chain
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.services.post_merge.enum_check_stage import EnumCheckStage
from omnibase_infra.services.post_merge.model_post_merge_finding import (
    ModelPostMergeFinding,
)


class ModelPostMergeResult(BaseModel):
    """Aggregate result from the post-merge check chain for a single PR.

    Published to ``onex.evt.github.post-merge-result.v1`` after all stages
    complete.

    Related Tickets:
        - OMN-6727: post-merge consumer chain
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    repo: str = Field(description="Repository full name")
    pr_number: int = Field(ge=1, description="Pull request number")
    merge_sha: str = Field(description="Merge commit SHA")
    findings: list[ModelPostMergeFinding] = Field(
        default_factory=list,
        description="All findings from the check chain",
    )
    stages_completed: list[EnumCheckStage] = Field(
        default_factory=list,
        description="Stages that completed successfully",
    )
    stages_failed: list[EnumCheckStage] = Field(
        default_factory=list,
        description="Stages that failed (error, not findings)",
    )
    tickets_created: list[str] = Field(
        default_factory=list,
        description="Linear ticket IDs created for findings",
    )
    started_at: datetime = Field(description="When the check chain started")
    completed_at: datetime = Field(description="When the check chain completed")


__all__ = ["ModelPostMergeResult"]
