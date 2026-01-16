# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Search result model for Qdrant handler."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.types import JsonDict


class ModelQdrantSearchResult(BaseModel):
    """Single search result from Qdrant.

    Attributes:
        id: Point ID
        score: Similarity score
        payload: Point payload data
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    id: str | UUID = Field(description="Point ID")
    score: float = Field(description="Similarity score")
    payload: JsonDict | None = Field(default=None, description="Point payload data")


__all__: list[str] = ["ModelQdrantSearchResult"]
