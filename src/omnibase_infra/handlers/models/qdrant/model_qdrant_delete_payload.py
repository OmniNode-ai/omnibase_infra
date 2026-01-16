# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Delete payload model for Qdrant handler."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelQdrantDeletePayload(BaseModel):
    """Payload for qdrant.delete operation result.

    Attributes:
        operation_type: Discriminator for payload type
        collection_name: Name of the collection
        point_ids: IDs of the deleted points
        success: Whether the delete was successful
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    operation_type: Literal["qdrant.delete"] = Field(
        default="qdrant.delete",
        description="Operation type discriminator",
    )
    collection_name: str = Field(
        description="Name of the collection"
    )  # ONEX_EXCLUDE: entity_name - Qdrant uses collection_name as the primary identifier, not a reference
    point_ids: list[str | UUID] = Field(description="IDs of the deleted points")
    success: bool = Field(default=True, description="Whether the delete was successful")


__all__: list[str] = ["ModelQdrantDeletePayload"]
