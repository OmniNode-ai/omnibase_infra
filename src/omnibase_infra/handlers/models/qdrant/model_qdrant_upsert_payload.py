# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Upsert payload model for Qdrant handler."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelQdrantUpsertPayload(BaseModel):
    """Payload for qdrant.upsert operation result.

    Attributes:
        operation_type: Discriminator for payload type
        collection_name: Name of the collection
        point_id: ID of the upserted point
        success: Whether the upsert was successful
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    operation_type: Literal["qdrant.upsert"] = Field(
        default="qdrant.upsert",
        description="Operation type discriminator",
    )
    collection_name: str = Field(
        description="Name of the collection"
    )  # ONEX_EXCLUDE: entity_name - Qdrant uses collection_name as the primary identifier, not a reference
    point_id: str | UUID = Field(description="ID of the upserted point")
    success: bool = Field(default=True, description="Whether the upsert was successful")


__all__: list[str] = ["ModelQdrantUpsertPayload"]
