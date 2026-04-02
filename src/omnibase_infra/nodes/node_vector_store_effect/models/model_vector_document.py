# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Model for a document to upsert into the vector store."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelVectorDocument(BaseModel):
    """A document to upsert into the vector store."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(..., description="Document text content.")
    metadata: dict[str, object] = Field(  # ONEX_EXCLUDE: dict_str_any
        default_factory=dict,
    )
    # ONEX_EXCLUDE: pattern_validator - Qdrant IDs can be UUID or arbitrary string
    doc_id: str = Field(
        default="", description="Optional document ID. Auto-generated if empty."
    )
