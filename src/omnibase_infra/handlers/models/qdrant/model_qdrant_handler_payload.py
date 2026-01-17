# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler payload model for Qdrant operations."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

from omnibase_infra.handlers.models.qdrant.model_qdrant_collection_payload import (
    ModelQdrantCollectionPayload,
)
from omnibase_infra.handlers.models.qdrant.model_qdrant_delete_payload import (
    ModelQdrantDeletePayload,
)
from omnibase_infra.handlers.models.qdrant.model_qdrant_search_payload import (
    ModelQdrantSearchPayload,
)
from omnibase_infra.handlers.models.qdrant.model_qdrant_upsert_payload import (
    ModelQdrantUpsertPayload,
)


def _get_qdrant_operation_type(value: object) -> str:
    """Extract operation_type from Qdrant payload for discriminated union."""
    if isinstance(value, dict):
        return str(value.get("operation_type", ""))
    return getattr(value, "operation_type", "")


QdrantPayload = Annotated[
    Annotated[ModelQdrantCollectionPayload, Tag("qdrant.create_collection")]
    | Annotated[ModelQdrantUpsertPayload, Tag("qdrant.upsert")]
    | Annotated[ModelQdrantSearchPayload, Tag("qdrant.search")]
    | Annotated[ModelQdrantDeletePayload, Tag("qdrant.delete")],
    Discriminator(_get_qdrant_operation_type),
]
"""Discriminated union of all Qdrant payload types."""


class ModelQdrantHandlerPayload(BaseModel):
    """Wrapper for Qdrant handler payloads using discriminated union.

    Attributes:
        data: The typed payload from the discriminated union
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    data: QdrantPayload = Field(description="The typed payload")


__all__: list[str] = ["ModelQdrantHandlerPayload", "QdrantPayload"]
