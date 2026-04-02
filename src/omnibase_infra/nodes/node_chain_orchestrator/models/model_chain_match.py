# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Single match from Qdrant similarity search."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .model_chain_entry import ModelChainEntry


class ModelChainMatch(BaseModel):
    """Single match from Qdrant similarity search."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chain_entry: ModelChainEntry = Field(..., description="Matched chain trajectory")
    similarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Cosine similarity score"
    )
    distance: float = Field(..., ge=0.0, description="Cosine distance (1 - similarity)")


__all__ = ["ModelChainMatch"]
