# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result model for completed mergeability-gate workflow."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_mergeability_evaluate_compute.models.enum_mergeability_status import (
    EnumMergeabilityStatus,
)


class ModelMergeabilityResult(BaseModel):
    """Final result emitted when mergeability-gate workflow completes."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(
        ..., description="Correlation ID from the originating command."
    )
    pr_number: int = Field(..., description="GitHub PR number.")
    repo: str = Field(..., description="GitHub repo slug.")
    status: EnumMergeabilityStatus = Field(..., description="Mergeability verdict.")
    blocked_reasons: tuple[str, ...] = Field(default_factory=tuple)
    split_reasons: tuple[str, ...] = Field(default_factory=tuple)
    label_applied: str = Field(
        default="", description="Label that was applied to the PR."
    )
