# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Effectiveness entry model for savings estimation.

Related Tickets:
    - OMN-6964: Token savings emitter
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_savings_estimation_compute.models.enum_model_tier import (
    EnumModelTier,
)


class ModelEffectivenessEntry(BaseModel):
    """A single injection effectiveness measurement.

    Represents one session or batch of effectiveness data from which
    savings can be derived.

    Attributes:
        utilization_score: How effectively injected context was used (0.0-1.0).
        patterns_count: Number of patterns injected in this session.
        tokens_saved: Estimated tokens not re-generated due to injection.
        model_tier: Which model tier was used for this session.
        is_output_tokens: Whether tokens_saved refers to output tokens.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    utilization_score: float = Field(
        ..., ge=0.0, le=1.0, description="Injection utilization score (0.0-1.0)"
    )
    patterns_count: int = Field(..., ge=0, description="Number of patterns injected")
    tokens_saved: int = Field(
        ..., ge=0, description="Tokens not re-generated due to injection"
    )
    model_tier: EnumModelTier = Field(
        default=EnumModelTier.OPUS, description="Model tier for pricing"
    )
    is_output_tokens: bool = Field(
        default=False, description="True if tokens_saved are output tokens"
    )


__all__: list[str] = ["ModelEffectivenessEntry"]
