# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result model for PR classification compute."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_merge_sweep_classify_compute.models.model_pr_classification import (
    ModelPRClassification,
)


class ModelClassifyResult(BaseModel):
    """Result of classifying PRs into Track A / Track B / SKIP."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    track_a: tuple[ModelPRClassification, ...] = Field(
        default_factory=tuple, description="Merge-ready PRs."
    )
    track_b: tuple[ModelPRClassification, ...] = Field(
        default_factory=tuple, description="PRs needing polish."
    )
    skipped: tuple[ModelPRClassification, ...] = Field(
        default_factory=tuple, description="Drafts and non-actionable PRs."
    )
    total_classified: int = Field(default=0, description="Total PRs classified.")
