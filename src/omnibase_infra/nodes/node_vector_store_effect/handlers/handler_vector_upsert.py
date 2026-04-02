# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that upserts documents with embeddings into Qdrant.

This is an EFFECT handler -- it performs I/O (Qdrant upsert + embedding API).

Ported from archive: ai-dev/containers/vector_store.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid as uuid_mod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

import httpx

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_vector_store_effect.models.model_vector_store_request import (
    ModelVectorStoreRequest,
)
from omnibase_infra.nodes.node_vector_store_effect.models.model_vector_store_result import (
    ModelVectorStoreResult,
)

logger = logging.getLogger(__name__)


class HandlerVectorUpsert:
    """Upserts documents into a Qdrant collection.

    Generates embeddings via the configured embedding endpoint, then stores
    vectors + payloads in Qdrant.
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
        """Upsert documents into the vector store.

        Args:
            request: Vector store request with documents to upsert.

        Returns:
            ModelVectorStoreResult with upserted document IDs.
        """
        if not request.documents:
            return ModelVectorStoreResult(
                correlation_id=request.correlation_id,
                success=False,
                operation="upsert",
                error_message="No documents provided for upsert.",
            )

        logger.info(
            "Vector upsert | collection=%s docs=%d correlation_id=%s",
            request.collection_name,
            len(request.documents),
            request.correlation_id,
        )

        texts = [doc.text for doc in request.documents]

        # Generate embeddings via the configured embedding endpoint
        try:
            embeddings = await self._generate_embeddings(texts)
        except Exception as exc:
            logger.exception(
                "Embedding generation failed | correlation_id=%s",
                request.correlation_id,
            )
            return ModelVectorStoreResult(
                correlation_id=request.correlation_id,
                success=False,
                operation="upsert",
                error_message=f"Embedding generation failed: {exc}",
            )

        # Build Qdrant points
        try:
            from qdrant_client.models import PointStruct
        except ImportError:
            return ModelVectorStoreResult(
                correlation_id=request.correlation_id,
                success=False,
                operation="upsert",
                error_message="qdrant-client is required for vector store operations.",
            )

        doc_ids: list[str] = []
        points: list[PointStruct] = []
        for doc, embedding in zip(request.documents, embeddings, strict=True):
            doc_id = doc.doc_id or str(uuid_mod.uuid4())
            doc_ids.append(doc_id)
            payload = {"text": doc.text, **doc.metadata}
            points.append(PointStruct(id=doc_id, vector=embedding, payload=payload))

        try:
            client = self._get_qdrant_client()
            await asyncio.to_thread(
                client.upsert,
                collection_name=request.collection_name,
                points=points,
            )
        except Exception as exc:
            logger.exception(
                "Qdrant upsert failed | correlation_id=%s", request.correlation_id
            )
            return ModelVectorStoreResult(
                correlation_id=request.correlation_id,
                success=False,
                operation="upsert",
                error_message=f"Qdrant upsert failed: {exc}",
            )

        logger.info(
            "Vector upsert complete | collection=%s ids=%d correlation_id=%s",
            request.collection_name,
            len(doc_ids),
            request.correlation_id,
        )
        return ModelVectorStoreResult(
            correlation_id=request.correlation_id,
            success=True,
            operation="upsert",
            upserted_ids=tuple(doc_ids),
        )

    def _get_qdrant_client(self) -> QdrantClient:
        """Get or lazily create a Qdrant client."""
        if self._qdrant_client is not None:
            return self._qdrant_client

        from qdrant_client import QdrantClient

        qdrant_url = os.environ.get(  # ONEX_EXCLUDE: archive port
            "QDRANT_URL", "http://localhost:6333"
        )
        self._qdrant_client = QdrantClient(url=qdrant_url)
        return self._qdrant_client

    async def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via the configured embedding endpoint."""
        embedding_url = os.environ.get(  # ONEX_EXCLUDE: archive port
            "LLM_EMBEDDING_URL", ""
        )
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
