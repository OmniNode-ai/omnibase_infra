# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Decision embedding pipeline for Qdrant vector store.

Embeds architectural and design decisions made during multi-session
coordination into Qdrant for semantic recall. Each decision is a single
text string embedded as one vector -- no chunking required.

Doctrine D7: Decision recall is enrichment only -- does not bloat
default resume context. Consumers pull from Qdrant on demand.

Part of the Multi-Session Coordination Layer (OMN-6850, Task 12).

Components:
    - ModelDecisionRecord: Typed record for a single decision.
    - build_embedding_text: Formats a decision record into embedding input.
    - DecisionEmbedder: Async client that embeds and upserts to Qdrant.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime
from uuid import UUID, uuid5

import httpx
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

logger = logging.getLogger(__name__)

# Qdrant collection configuration
COLLECTION_NAME = "session_decisions"
VECTOR_DIMENSION = 1024
DISTANCE_METRIC = qdrant_models.Distance.COSINE

# Namespace for deterministic point IDs (uuid5)
_DECISION_NAMESPACE = UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # NAMESPACE_DNS

# Default embedding endpoint
_DEFAULT_EMBEDDING_URL = "http://192.168.86.200:8100"


class ModelDecisionRecord(BaseModel):
    """A single decision record for embedding into Qdrant.

    Each record represents one architectural or design decision made
    during task execution. The combination of task_id + decision_text
    hash produces a deterministic point ID for idempotent upserts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(  # onex:str-not-uuid — Linear ticket ID, not a UUID
        ...,
        min_length=1,
        max_length=64,
        description="Linear ticket ID (e.g., 'OMN-1234').",
    )
    session_id: str = Field(  # onex:str-not-uuid — CLI session ID, not a UUID
        ...,
        min_length=1,
        description="CLI session ID that produced this decision.",
    )
    decision_text: str = Field(
        ...,
        min_length=1,
        description="The decision content.",
    )
    context: str = Field(
        default="",
        description="Additional context for the decision.",
    )
    timestamp: datetime = Field(
        ...,
        description="When the decision was made.",
    )


def build_embedding_text(record: ModelDecisionRecord) -> str:
    """Format a decision record into a single string for embedding.

    The format is designed to produce semantically meaningful vectors
    that capture both the decision content and its task context.

    Args:
        record: The decision record to format.

    Returns:
        A formatted string suitable for embedding.
    """
    text = f"Task {record.task_id}: {record.decision_text}"
    if record.context:
        text += f". Context: {record.context}"
    return text


def _decision_point_id(task_id: str, decision_text: str) -> str:
    """Compute a deterministic Qdrant point ID for a decision.

    Uses uuid5(NAMESPACE_DNS, "{task_id}:{sha256_of_decision_text}")
    so re-upserting the same decision is idempotent.

    Args:
        task_id: The Linear ticket ID.
        decision_text: The raw decision text.

    Returns:
        A UUID string for the Qdrant point ID.
    """
    decision_hash = hashlib.sha256(decision_text.encode()).hexdigest()[:16]
    return str(uuid5(_DECISION_NAMESPACE, f"{task_id}:{decision_hash}"))


class DecisionEmbedder:
    """Async client that embeds decision records and upserts to Qdrant.

    Uses httpx for async embedding requests to the Qwen3-Embedding-8B
    endpoint and qdrant-client for vector storage.

    Usage::

        embedder = DecisionEmbedder()
        await embedder.ensure_collection()
        await embedder.embed_and_upsert(record)
    """

    def __init__(
        self,
        *,
        qdrant_url: str | None = None,
        qdrant_port: int | None = None,
        embedding_url: str | None = None,
    ) -> None:
        self._qdrant_url = qdrant_url or os.getenv(
            "QDRANT_URL", "http://localhost:6333"
        )
        self._qdrant_port = qdrant_port or int(os.getenv("QDRANT_PORT", "6333"))
        self._embedding_url = embedding_url or os.getenv(
            "LLM_EMBEDDING_URL", _DEFAULT_EMBEDDING_URL
        )
        self._qdrant = QdrantClient(url=self._qdrant_url)

    async def ensure_collection(self) -> None:
        """Create the session_decisions collection if it does not exist."""
        collections = self._qdrant.get_collections().collections
        existing = {c.name for c in collections}
        if COLLECTION_NAME not in existing:
            self._qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qdrant_models.VectorParams(
                    size=VECTOR_DIMENSION,
                    distance=DISTANCE_METRIC,
                ),
            )
            logger.info("Created Qdrant collection '%s'", COLLECTION_NAME)

    async def _get_embedding(self, text: str) -> list[float]:
        """Request an embedding vector from the embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            httpx.HTTPStatusError: If the embedding request fails.
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._embedding_url}/v1/embeddings",
                json={"input": text, "model": "default"},
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]

    async def embed_and_upsert(self, record: ModelDecisionRecord) -> str:
        """Embed a decision record and upsert it into Qdrant.

        Args:
            record: The decision record to embed and store.

        Returns:
            The point ID (UUID string) of the upserted point.
        """
        text = build_embedding_text(record)
        vector = await self._get_embedding(text)

        point_id = _decision_point_id(record.task_id, record.decision_text)

        self._qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "task_id": record.task_id,
                        "session_id": record.session_id,
                        "decision_text": record.decision_text,
                        "context": record.context,
                        "timestamp": record.timestamp.isoformat(),
                        "embedding_text": text,
                    },
                ),
            ],
        )
        logger.info(
            "Upserted decision for task %s (point_id=%s)", record.task_id, point_id
        )
        return point_id

    async def search_similar(
        self,
        query_text: str,
        *,
        limit: int = 5,
        task_id_filter: str | None = None,
    ) -> list[qdrant_models.ScoredPoint]:
        """Search for decisions similar to the query text.

        Args:
            query_text: The text to find similar decisions for.
            limit: Maximum number of results.
            task_id_filter: Optional filter to restrict results to a specific task.

        Returns:
            A list of scored points from Qdrant.
        """
        vector = await self._get_embedding(query_text)

        query_filter = None
        if task_id_filter:
            query_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="task_id",
                        match=qdrant_models.MatchValue(value=task_id_filter),
                    ),
                ],
            )

        return self._qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            query_filter=query_filter,
            limit=limit,
        ).points
