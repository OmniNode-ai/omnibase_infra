# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Path decision made by the chain orchestrator after retrieval."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelChainRetrievalDecision(BaseModel):
    """Replay-or-explore decision derived from a chain retrieval result."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    path: Literal["replay", "explore"] = Field(
        ..., description="Next workflow path: replay a cached chain or explore"
    )
    chain_id: UUID | None = Field(
        default=None, description="Matched chain ID; None when the path is explore"
    )
    is_hit: bool = Field(..., description="Whether retrieval found a usable chain")


__all__ = ["ModelChainRetrievalDecision"]
