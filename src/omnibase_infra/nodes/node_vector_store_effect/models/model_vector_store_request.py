# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Input model for vector store operations."""

from __future__ import annotations

from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_vector_store_effect.models.model_vector_document import (
    ModelVectorDocument,
)


class ModelVectorStoreRequest(BaseModel):
    """Request for a vector store operation (upsert or search)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(
        default_factory=uuid4, description="Workflow correlation ID."
    )
    operation: Literal["upsert", "search"] = Field(..., description="Operation type.")
    collection_name: str = Field(  # ONEX_EXCLUDE: entity_name_pattern
        default="vectors", description="Qdrant collection name."
    )
    # Upsert fields
    documents: tuple[ModelVectorDocument, ...] = Field(
        default_factory=tuple, description="Documents to upsert (for upsert operation)."
    )
    # Search fields
    query: str = Field(
        default="", description="Search query text (for search operation)."
    )
    limit: int = Field(default=10, ge=1, le=100, description="Max results for search.")
    metadata_filter: dict[str, object] = Field(  # ONEX_EXCLUDE: dict_str_any
        default_factory=dict,
    )
