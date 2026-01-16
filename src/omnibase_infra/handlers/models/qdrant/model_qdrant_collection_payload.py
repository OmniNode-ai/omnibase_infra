# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Collection payload model for Qdrant handler."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelQdrantCollectionPayload(BaseModel):
    """Payload for qdrant.create_collection operation result.

    Attributes:
        operation_type: Discriminator for payload type
        collection_name: Name of the created collection
        vector_size: Dimension of vectors in the collection
        distance: Distance metric used (cosine, dot, euclidean)
        success: Whether the creation was successful
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    operation_type: Literal["qdrant.create_collection"] = Field(
        default="qdrant.create_collection",
        description="Operation type discriminator",
    )
    collection_name: str = Field(
        description="Name of the created collection"
    )  # ONEX_EXCLUDE: entity_name - Qdrant uses collection_name as the primary identifier, not a reference
    vector_size: int = Field(description="Dimension of vectors in the collection")
    distance: str = Field(
        default="cosine",
        description="Distance metric used (cosine, dot, euclidean)",
    )
    success: bool = Field(
        default=True, description="Whether the creation was successful"
    )


__all__: list[str] = ["ModelQdrantCollectionPayload"]
