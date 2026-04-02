# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Aggregate reducer state — all capability scores."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_routing_score_reducer.models.model_capability_score import (
    ModelCapabilityScore,
)


class ModelReducerState(BaseModel):
    """Aggregate reducer state — all capability scores."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Latest correlation ID.")
    scores: tuple[ModelCapabilityScore, ...] = Field(
        default_factory=tuple,
        description="Capability scores per (model, task_type).",
    )
    total_outcomes_processed: int = Field(
        default=0, description="Total routing outcomes processed."
    )
