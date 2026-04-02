# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Classification result for a single PR in the merge-sweep workflow."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_merge_sweep_pr_list_effect.models.model_pr_info import (
    ModelPRInfo,
)


class ModelPRClassification(BaseModel):
    """Classification of a single PR into Track A or Track B."""

    model_config = ConfigDict(frozen=True, extra="forbid")

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
