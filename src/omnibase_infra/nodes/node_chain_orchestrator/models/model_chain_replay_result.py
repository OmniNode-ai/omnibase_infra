# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Output of the chain replay compute node."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .model_chain_step import ModelChainStep


class ModelChainReplayResult(BaseModel):
    """Output of the chain replay compute node."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID")
    adapted_steps: tuple[ModelChainStep, ...] = Field(
        ..., description="Chain steps adapted to new context"
    )
    adaptation_summary: str = Field(
        ..., description="Human-readable summary of adaptations made"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence that the adaptation is valid",
    )


__all__ = ["ModelChainReplayResult"]
