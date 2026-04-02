# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Model for a single vector search result."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelVectorSearchHit(BaseModel):
    """A single search result from vector similarity search."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ONEX_EXCLUDE: pattern_validator - Qdrant IDs can be UUID or arbitrary string
    doc_id: str = Field(..., description="Document ID.")
    score: float = Field(..., description="Cosine similarity score.")
    text: str = Field(default="", description="Document text.")
    metadata: dict[str, object] = Field(  # ONEX_EXCLUDE: dict_str_any
        default_factory=dict,
    )
