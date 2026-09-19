# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Classification result for a single PR in the merge-sweep workflow."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_merge_sweep_classify_compute.models.enum_classify_skip_reason import (
    EnumClassifySkipReason,
)
from omnibase_infra.nodes.node_merge_sweep_pr_list_effect.models.model_pr_info import (
    ModelPRInfo,
)


class ModelPRClassification(BaseModel):
    """Classification of a single PR into Track A or Track B."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    pr: ModelPRInfo = Field(..., description="The PR being classified.")
    track: Literal["A", "B", "SKIP"] = Field(
        ...,
        description=(
            "A = merge-ready (enable auto-merge), "
            "B = needs polish (CI failures, conflicts, changes requested), "
            "SKIP = draft or not actionable."
        ),
    )
    reason: str = Field(
        default="", description="Human-readable reason for classification."
    )
    skip_reason: EnumClassifySkipReason = Field(
        default=EnumClassifySkipReason.NOT_SKIPPED,
        description=(
            "Typed reason the PR was withheld, so a consumer can branch on it "
            "without matching prose (OMN-18823)."
        ),
    )
    excluded_account: str = Field(
        default="",
        description=(
            "The collaborator account that caused a COLLABORATOR_EXCLUDED "
            "skip. Empty for every other outcome."
        ),
    )
