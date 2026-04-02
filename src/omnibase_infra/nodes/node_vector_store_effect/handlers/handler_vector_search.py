# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that performs similarity search against Qdrant.

This is an EFFECT handler -- it performs I/O (Qdrant search + embedding API).

Ported from archive: ai-dev/containers/vector_store.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

import httpx

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_vector_store_effect.models.model_vector_search_hit import (
    ModelVectorSearchHit,
)
from omnibase_infra.nodes.node_vector_store_effect.models.model_vector_store_request import (
    ModelVectorStoreRequest,
)
from omnibase_infra.nodes.node_vector_store_effect.models.model_vector_store_result import (
    ModelVectorStoreResult,
)

logger = logging.getLogger(__name__)


class HandlerVectorSearch:
    """Performs similarity search in a Qdrant collection.

    Generates an embedding for the query text, then searches Qdrant by cosine
    similarity and returns ranked hits.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._qdrant_client = qdrant_client
        self._owns_http = http_client is None
        self._http_client = http_client or httpx.AsyncClient(timeout=60.0)

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(self, request: ModelVectorStoreRequest) -> ModelVectorStoreResult:
        """Search for similar documents in the vector store.

        Args:
            request: Vector store request with query text.

        Returns:
            ModelVectorStoreResult with ranked search hits.
        """
        if not request.query:
            return ModelVectorStoreResult(
                correlation_id=request.correlation_id,
                success=False,
                operation="search",
                error_message="No query text provided for search.",
            )

        logger.info(
            "Vector search | collection=%s query=%s... limit=%d correlation_id=%s",
            request.collection_name,
            request.query[:50],
            request.limit,
            request.correlation_id,
        )

        # Generate embedding for query
        try:
            embeddings = await self._generate_embeddings([request.query])
            query_vector = embeddings[0]
        except Exception as exc:
            logger.exception(
                "Query embedding failed | correlation_id=%s", request.correlation_id
            )
            return ModelVectorStoreResult(
                correlation_id=request.correlation_id,
                success=False,
                operation="search",
                error_message=f"Query embedding failed: {exc}",
            )

        # Build optional Qdrant filter
        query_filter = None
        if request.metadata_filter:
            try:
                from qdrant_client.http.models import FieldCondition, Filter, MatchValue

                conditions = [
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in request.metadata_filter.items()
                ]
                query_filter = Filter(must=conditions)
            except ImportError:
                pass  # qdrant_client filter types are optional; search proceeds without metadata filter

        # Search Qdrant
        try:
            client = self._get_qdrant_client()
            response = await asyncio.to_thread(
                client.query_points,
                collection_name=request.collection_name,
                query=query_vector,
                limit=request.limit,
                query_filter=query_filter,
            )
            results = response.points
        except Exception as exc:
            logger.exception(
                "Qdrant search failed | correlation_id=%s", request.correlation_id
            )
            return ModelVectorStoreResult(
                correlation_id=request.correlation_id,
                success=False,
                operation="search",
                error_message=f"Qdrant search failed: {exc}",
            )

        hits = tuple(
            ModelVectorSearchHit(
                doc_id=str(r.id),
                score=r.score,
                text=r.payload.get("text", "") if r.payload else "",
                metadata={k: v for k, v in (r.payload or {}).items() if k != "text"},
            )
            for r in results
        )

        logger.info(
            "Vector search complete | hits=%d correlation_id=%s",
            len(hits),
            request.correlation_id,
        )
        return ModelVectorStoreResult(
            correlation_id=request.correlation_id,
            success=True,
            operation="search",
            hits=hits,
        )

    def _get_qdrant_client(self) -> QdrantClient:
        """Get or lazily create a Qdrant client."""
        if self._qdrant_client is not None:
            return self._qdrant_client

        from qdrant_client import QdrantClient

        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        self._qdrant_client = QdrantClient(url=qdrant_url)
        return self._qdrant_client

    async def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via the configured embedding endpoint."""
        embedding_url = os.environ.get("LLM_EMBEDDING_URL", "")
        if not embedding_url:
            raise RuntimeError("LLM_EMBEDDING_URL is not configured")

        response = await self._http_client.post(
            f"{embedding_url.rstrip('/')}/v1/embeddings",
            json={"input": texts, "model": "embedding"},
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    async def close(self) -> None:
        """Release HTTP resources if owned by this handler."""
        if self._owns_http and self._http_client is not None:
            await self._http_client.aclose()
