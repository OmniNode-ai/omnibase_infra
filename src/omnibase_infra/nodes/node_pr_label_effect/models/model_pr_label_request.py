# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request model for PR label effect."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_mergeability_evaluate_compute.models.model_mergeability_evaluation import (
    EnumMergeabilityStatus,
)


class ModelPrLabelRequest(BaseModel):
    """Request to apply mergeability labels to a GitHub PR."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    pr_number: int = Field(..., description="GitHub PR number.")
    repo: str = Field(..., description="GitHub repo slug.")
    status: EnumMergeabilityStatus = Field(
        ..., description="Mergeability verdict to apply as label."
    )
