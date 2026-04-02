# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Output of the chain retrieval effect."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .model_chain_match import ModelChainMatch


class ModelChainRetrievalResult(BaseModel):
    """Output of the chain retrieval effect."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID")
    matches: tuple[ModelChainMatch, ...] = Field(
        default_factory=tuple, description="Matched chains ordered by similarity"
    )
    best_match_similarity: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Highest similarity score"
    )
    query_embedding: list[float] = Field(
        default_factory=list, description="Embedding vector used for query"
    )
    is_hit: bool = Field(
        default=False, description="True if best match >= similarity threshold"
    )


__all__ = ["ModelChainRetrievalResult"]
