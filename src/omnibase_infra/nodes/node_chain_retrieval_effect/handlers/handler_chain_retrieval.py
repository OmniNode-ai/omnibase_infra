# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that embeds a prompt and queries Qdrant for similar verified chains.

This is an EFFECT handler -- it performs I/O (embedding API + Qdrant query).
"""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_chain_orchestrator.models import (
    ModelChainEntry,
    ModelChainMatch,
    ModelChainRetrievalResult,
)
from omnibase_infra.nodes.node_chain_orchestrator.models.protocol_chain_embedding_client import (
    ProtocolChainEmbeddingClient,
)
from omnibase_infra.nodes.node_chain_orchestrator.models.protocol_chain_vector_client import (
    ProtocolChainVectorClient,
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "onex_chain_trajectories"


class HandlerChainRetrieval:
    """Embeds a prompt and searches Qdrant for similar verified chains."""

    def __init__(
        self,
        embedding_client: ProtocolChainEmbeddingClient | None = None,
        qdrant_client: ProtocolChainVectorClient | None = None,
        vector_size: int = 4096,
    ) -> None:
        self._embedding_client = embedding_client
        self._qdrant_client = qdrant_client
        self._vector_size = vector_size

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        prompt_text: str,
        correlation_id: UUID,
        similarity_threshold: float = 0.85,
        limit: int = 5,
    ) -> ModelChainRetrievalResult:
        """Embed prompt and search Qdrant for similar chains.

        Args:
            prompt_text: The prompt to embed and search for.
            correlation_id: Workflow correlation ID.
            similarity_threshold: Minimum cosine similarity for a hit.
            limit: Maximum number of results to return.

        Returns:
            ModelChainRetrievalResult with matches and hit/miss flag.
        """
        logger.info(
            "Retrieving chains for prompt (correlation_id=%s, threshold=%.2f)",
            correlation_id,
            similarity_threshold,
        )

        # Step 1: Embed the prompt
        try:
            query_embedding = await self._get_embedding(prompt_text)
        except Exception:
            logger.exception(
                "Failed to embed prompt (correlation_id=%s)", correlation_id
            )
            return ModelChainRetrievalResult(
                correlation_id=correlation_id,
                matches=(),
                best_match_similarity=0.0,
                query_embedding=[],
                is_hit=False,
            )

        # Step 2: Ensure collection exists
        await self._ensure_collection()
        if self._qdrant_client is None:
            logger.warning(
                "HandlerChainRetrieval: qdrant_client not configured (correlation_id=%s)",
                correlation_id,
            )
            return ModelChainRetrievalResult(
                correlation_id=correlation_id,
                matches=(),
                best_match_similarity=0.0,
                query_embedding=query_embedding,
                is_hit=False,
            )

        # Step 3: Query Qdrant
        try:
            hits = await asyncio.to_thread(
                self._qdrant_client.search,
                COLLECTION_NAME,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=similarity_threshold * 0.5,  # wider net, filter later
            )
        except Exception:
            logger.exception("Qdrant search failed (correlation_id=%s)", correlation_id)
            return ModelChainRetrievalResult(
                correlation_id=correlation_id,
                matches=(),
                best_match_similarity=0.0,
                query_embedding=query_embedding,
                is_hit=False,
            )

        if not hits:
            logger.info("No chain matches found (correlation_id=%s)", correlation_id)
            return ModelChainRetrievalResult(
                correlation_id=correlation_id,
                matches=(),
                best_match_similarity=0.0,
                query_embedding=query_embedding,
                is_hit=False,
            )

        # Step 4: Build matches
        # Qdrant ScoredPoint objects have .payload, .score, .id attributes
        matches: list[ModelChainMatch] = []
        for hit in hits:
            try:
                payload = getattr(hit, "payload", None)
                score: float = getattr(hit, "score", 0.0)
                point_id = getattr(hit, "id", "unknown")
                if payload is None:
                    logger.warning(
                        "Skipping hit without payload (point_id=%s)", point_id
                    )
                    continue
                chain_entry = ModelChainEntry.model_validate(payload)
                matches.append(
                    ModelChainMatch(
                        chain_entry=chain_entry,
                        similarity_score=score,
                        distance=1.0 - score,
                    )
                )
            except (KeyError, ValueError, TypeError):
                logger.warning(
                    "Skipping invalid chain payload (point_id=%s)",
                    getattr(hit, "id", "unknown"),
                )
                continue

        best_similarity = max(m.similarity_score for m in matches) if matches else 0.0
        is_hit = best_similarity >= similarity_threshold

        logger.info(
            "Chain retrieval: %d matches, best=%.3f, hit=%s (correlation_id=%s)",
            len(matches),
            best_similarity,
            is_hit,
            correlation_id,
        )

        return ModelChainRetrievalResult(
            correlation_id=correlation_id,
            matches=tuple(
                sorted(matches, key=lambda m: m.similarity_score, reverse=True)
            ),
            best_match_similarity=best_similarity,
            query_embedding=query_embedding,
            is_hit=is_hit,
        )

    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text via the embedding client."""
        if self._embedding_client is None:
            raise RuntimeError(
                "HandlerChainRetrieval: embedding_client is not configured. "
                "Pass an embedding_client or configure via DI container."
            )
        return await self._embedding_client.get_embedding(text)

    async def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it doesn't exist."""
        if self._qdrant_client is None:
            logger.warning(
                "HandlerChainRetrieval: qdrant_client is None, skipping collection check"
            )
            return
        try:
            exists = await asyncio.to_thread(
                self._qdrant_client.collection_exists, COLLECTION_NAME
            )
            if not exists:
                from qdrant_client.models import Distance, VectorParams

                await asyncio.to_thread(
                    self._qdrant_client.create_collection,
                    COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self._vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    "Created Qdrant collection '%s' (vector_size=%d)",
                    COLLECTION_NAME,
                    self._vector_size,
                )
        except (ImportError, OSError, RuntimeError):
            logger.warning("Could not ensure collection '%s' exists", COLLECTION_NAME)
