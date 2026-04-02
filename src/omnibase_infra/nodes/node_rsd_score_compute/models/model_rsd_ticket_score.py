# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Ticket score model for RSD scoring."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_rsd_score_compute.models.model_rsd_factor_score import (
    ModelRsdFactorScore,
)


class ModelRsdTicketScore(BaseModel):
    """Complete RSD priority score for a single ticket."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    ticket_id: str = Field(min_length=1)  # pattern-ok: Linear ticket IDs are strings
    final_score: float = Field(..., description="Final priority score (0.0-1.0).")
    factors: tuple[ModelRsdFactorScore, ...] = Field(..., description="Factor scores.")
    algorithm_version: str = Field(
        default="rsd-5-factor-v1", description="Algorithm version."
    )
