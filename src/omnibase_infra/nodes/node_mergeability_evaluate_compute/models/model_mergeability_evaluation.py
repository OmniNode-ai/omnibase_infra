# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Evaluation result model for mergeability gate compute node."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_mergeability_evaluate_compute.models.enum_mergeability_status import (
    EnumMergeabilityStatus,
)


class ModelMergeabilityEvaluation(BaseModel):
    """Result of evaluating a PR's mergeability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    pr_number: int = Field(..., description="GitHub PR number.")
    repo: str = Field(..., description="GitHub repo slug.")
    status: EnumMergeabilityStatus = Field(..., description="Mergeability verdict.")
    blocked_reasons: tuple[str, ...] = Field(
        default_factory=tuple, description="Reasons the PR is blocked."
    )
    split_reasons: tuple[str, ...] = Field(
        default_factory=tuple, description="Reasons the PR should be split."
    )
