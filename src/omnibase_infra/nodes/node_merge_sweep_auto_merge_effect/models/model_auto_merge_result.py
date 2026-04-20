# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result model for auto-merge effect."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_merge_sweep_auto_merge_effect.models.model_auto_merge_outcome import (
    ModelAutoMergeOutcome,
)


class ModelAutoMergeResult(BaseModel):
    """Result of enabling auto-merge across multiple PRs."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    outcomes: tuple[ModelAutoMergeOutcome, ...] = Field(
        default_factory=tuple, description="Per-PR outcomes."
    )
    total_enabled: int = Field(default=0, description="PRs with auto-merge enabled.")
    total_enqueued: int = Field(
        default=0,
        description=(
            "PRs successfully enqueued into the merge queue. For merge-queue repos "
            "this is the meaningful success metric — enabling auto-merge alone "
            "does not cause a PR to merge."
        ),
    )
    total_failed: int = Field(default=0, description="PRs that failed.")
    success: bool = Field(default=True, description="Overall success.")
