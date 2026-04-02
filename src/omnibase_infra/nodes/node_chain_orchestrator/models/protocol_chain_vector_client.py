# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Protocol for vector store client used by chain retrieval and storage."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProtocolChainVectorClient(Protocol):
    """Protocol for vector store client used by chain retrieval and storage."""

    def collection_exists(self, collection_name: str) -> bool: ...

    def create_collection(
        self,
        collection_name: str,
        vectors_config: object = ...,
    ) -> None: ...

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> list[object]: ...

    def upsert(
        self,
        collection_name: str,
        points: list[object],
    ) -> None: ...


__all__ = ["ProtocolChainVectorClient"]
