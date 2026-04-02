# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Output model for RSD score compute node."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_rsd_score_compute.models.model_rsd_ticket_score import (
    ModelRsdTicketScore,
)


class ModelRsdScoreResult(BaseModel):
    """Result of batch RSD priority scoring."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    ticket_scores: tuple[ModelRsdTicketScore, ...] = Field(
        ..., description="Priority scores for each ticket."
    )
    ranked_ticket_ids: tuple[str, ...] = Field(
        ..., description="Ticket IDs ordered by descending priority."
    )
