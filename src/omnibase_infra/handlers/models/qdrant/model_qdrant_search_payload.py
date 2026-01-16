# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Search payload model for Qdrant handler."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.handlers.models.qdrant.model_qdrant_search_result import (
    ModelQdrantSearchResult,
)


class ModelQdrantSearchPayload(BaseModel):
    """Payload for qdrant.search operation result.

    Attributes:
        operation_type: Discriminator for payload type
        collection_name: Name of the collection searched (ONEX_EXCLUDE: entity_name_pattern)
        results: List of search results
        limit: Maximum number of results requested
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    operation_type: Literal["qdrant.search"] = Field(
        default="qdrant.search",
        description="Operation type discriminator",
    )
    collection_name: str = Field(
        description="Name of the collection searched"
    )  # ONEX_EXCLUDE: entity_name - Qdrant uses collection_name as the primary identifier, not a reference
    results: list[ModelQdrantSearchResult] = Field(
        default_factory=list,
        description="List of search results",
    )
    limit: int = Field(default=10, description="Maximum number of results requested")


__all__: list[str] = ["ModelQdrantSearchPayload"]
