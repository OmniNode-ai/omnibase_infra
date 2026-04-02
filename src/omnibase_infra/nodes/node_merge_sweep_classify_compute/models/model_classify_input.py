# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Input model for PR classification compute."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_merge_sweep_pr_list_effect.models.model_pr_info import (
    ModelPRInfo,
)


class ModelClassifyInput(BaseModel):
    """Input to the classification compute node."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    prs: tuple[ModelPRInfo, ...] = Field(..., description="PRs to classify.")
    require_approval: bool = Field(
        default=True, description="Whether to require review approval for Track A."
    )
