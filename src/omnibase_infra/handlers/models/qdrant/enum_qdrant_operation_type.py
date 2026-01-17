# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Qdrant operation type enumeration for discriminated unions."""

from __future__ import annotations

from enum import Enum


class EnumQdrantOperationType(str, Enum):
    """Qdrant operation types for handler routing.

    Attributes:
        CREATE_COLLECTION: Create a new vector collection
        UPSERT: Insert or update vectors
        SEARCH: Search for similar vectors
        DELETE: Delete vectors by ID
    """

    CREATE_COLLECTION = "qdrant.create_collection"
    UPSERT = "qdrant.upsert"
    SEARCH = "qdrant.search"
    DELETE = "qdrant.delete"


__all__: list[str] = ["EnumQdrantOperationType"]
