# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Output model for vector store operations."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_vector_store_effect.models.model_vector_search_hit import (
    ModelVectorSearchHit,
)


class ModelVectorStoreResult(BaseModel):
    """Result of a vector store operation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    success: bool = Field(..., description="Whether the operation succeeded.")
    operation: str = Field(default="", description="Operation that was performed.")
    # Upsert results
    upserted_ids: tuple[str, ...] = Field(
        default_factory=tuple, description="IDs of upserted documents."
    )
    # Search results
    hits: tuple[ModelVectorSearchHit, ...] = Field(
        default_factory=tuple, description="Search results."
    )
    error_message: str = Field(
        default="", description="Error detail if success is False."
    )
