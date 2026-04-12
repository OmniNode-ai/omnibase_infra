# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Decision embedding models and utilities for Qdrant semantic retrieval.

Embeds decision records and stores them in Qdrant for semantic retrieval.

Collection: "session_decisions"
Vector: 1024-dim from Qwen3-Embedding-8B (LLM_EMBEDDING_URL)
Payload: task_id, session_id, decision_text, context, timestamp

Embedding text format:
  "Task {task_id}: {decision_text}. Context: {context}"

Uses EmbeddingClient pattern (async httpx to port 8100).
Uses qdrant_client for storage.

Point ID: uuid5(NAMESPACE_DNS, f"{task_id}:{decision_hash}")
  - Dedup key: same decision text under same task_id -> same point ID (idempotent upsert)

Part of the Multi-Session Coordination Layer (OMN-6850, Task 12).
"""

from __future__ import annotations

import hashlib
import logging
import os
from uuid import UUID, uuid5

import httpx
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

logger = logging.getLogger(__name__)

# Qdrant collection name for decision vectors.
COLLECTION_NAME = "session_decisions"

# Vector dimension for Qwen3-Embedding-8B.
VECTOR_DIM = 1024

# Namespace UUID for deterministic point ID generation.
_NAMESPACE = UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # NAMESPACE_DNS


class ModelDecisionRecord(BaseModel):
    """A single architectural or design decision made during a task.

    Immutable record capturing a decision for embedding and semantic search.
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
        description="CLI session ID that recorded the decision.",
    )
    decision_text: str = Field(
        ...,
        min_length=1,
        description="The decision text.",
    )
    context: str = Field(
        default="",
        description="Additional context around the decision.",
    )
    timestamp: str = Field(
        default="",
        description="ISO 8601 timestamp of when the decision was made.",
    )


def build_embedding_text(record: ModelDecisionRecord) -> str:
    """Build embedding text from a decision record.

    Format: "Task {task_id}: {decision_text}. Context: {context}"

    Args:
        record: The decision record to embed.

    Returns:
        Formatted text suitable for embedding generation.
    """
    parts = [f"Task {record.task_id}: {record.decision_text}"]
    if record.context:
        parts.append(f"Context: {record.context}")
    return ". ".join(parts)


def decision_point_id(task_id: str, decision_text: str) -> str:
    """Generate a deterministic UUID for deduplication.

    Same task_id + decision_text always produces the same point ID,
    enabling idempotent upserts.

    Args:
        task_id: The ticket ID.
        decision_text: The decision text.

    Returns:
        A UUID string for use as the Qdrant point ID.
    """
    decision_hash = hashlib.sha256(decision_text.encode()).hexdigest()[:16]
    return str(uuid5(_NAMESPACE, f"{task_id}:{decision_hash}"))


class EmbeddingClient:
    """Async HTTP client for generating embeddings via OpenAI-compatible API.

    Calls POST {base_url}/v1/embeddings with the standard request format.
    Default base_url is read from LLM_EMBEDDING_URL env var.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = base_url or os.environ.get("LLM_EMBEDDING_URL", "")

    async def embed(self, text: str) -> list[float]:
        """Generate a 1024-dim embedding vector for the given text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            httpx.HTTPStatusError: If the embedding API returns an error.
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._base_url}/v1/embeddings",
                json={"input": text, "model": "embedding"},
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]


class DecisionEmbedderWriter:
    """Writes decision embeddings to Qdrant.

    Thin wrapper around QdrantClient for upserting decision vectors
    into the session_decisions collection.
    """

    def __init__(self, qdrant: QdrantClient) -> None:
        self._qdrant = qdrant

    def ensure_collection(self) -> None:
        """Create the session_decisions collection if it does not exist."""
        collections = self._qdrant.get_collections().collections
        existing = {c.name for c in collections}
        if COLLECTION_NAME not in existing:
            self._qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qdrant_models.VectorParams(
                    size=VECTOR_DIM,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection %s", COLLECTION_NAME)

    def upsert(
        self,
        record: ModelDecisionRecord,
        vector: list[float],
    ) -> None:
        """Upsert a decision embedding into Qdrant.

        Uses deterministic point ID for idempotent upserts.

        Args:
            record: The decision record metadata.
            vector: The embedding vector (1024-dim).
        """
        point_id = decision_point_id(record.task_id, record.decision_text)
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
                        "timestamp": record.timestamp,
                    },
                ),
            ],
        )


__all__ = [
    "COLLECTION_NAME",
    "DecisionEmbedderWriter",
    "EmbeddingClient",
    "ModelDecisionRecord",
    "VECTOR_DIM",
    "build_embedding_text",
    "decision_point_id",
]
