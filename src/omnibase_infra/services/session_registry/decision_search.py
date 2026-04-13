# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Semantic search over session decisions stored in Qdrant.

Methods:
  - search(query, limit, task_id) -> list[ModelDecisionSearchResult]
    Embeds the query, searches Qdrant "session_decisions" collection.
    Optional task_id filter to scope search to a specific task.

  - search_related(task_id, limit) -> list[ModelDecisionSearchResult]
    Finds decisions from OTHER tasks that are semantically similar to
    decisions made in the given task.

  - format_results(results) -> str
    Human-readable formatted output for context injection.

Collection: "session_decisions"
Embedding: Qwen3-Embedding-8B on port 8100 via OpenAI-compatible API.

Doctrine D7: Qdrant-based decision recall is enrichment only. It should
not bloat default resume context unless it demonstrably improves real
resume quality in measured use.

Part of the Multi-Session Coordination Layer (OMN-6850, Task 14).
"""

from __future__ import annotations

import logging
import os

import httpx
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "session_decisions"
_DEFAULT_EMBEDDING_URL = os.getenv("LLM_EMBEDDING_URL", "")
_DEFAULT_EMBEDDING_MODEL = "Qwen3-Embedding-8B"


class ModelDecisionSearchResult(BaseModel):
    """A single decision search result from Qdrant.

    Attributes:
        task_id: The task that made this decision.
        decision_text: The decision content.
        context: Context in which the decision was made.
        score: Similarity score (0.0 to 1.0).
        timestamp: When the decision was recorded.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(  # onex:str-not-uuid — Linear ticket ID, not a UUID
        ..., description="Task that made this decision"
    )
    decision_text: str = Field(..., description="Decision content")
    context: str = Field(default="", description="Context of the decision")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    timestamp: str = Field(default="", description="When the decision was recorded")


class DecisionSearchClient:
    """Client for semantic search over session decisions in Qdrant.

    Args:
        qdrant: QdrantClient instance (None for format-only usage).
        embedder: Callable that takes text and returns embedding vector.
            If None, uses the default HTTP embedding client.
    """

    def __init__(
        self,
        qdrant: QdrantClient | None = None,
        embedder: object | None = None,
    ) -> None:
        self._qdrant = qdrant
        self._embedder = embedder

    async def _embed(self, text: str) -> list[float]:
        """Embed text using the configured embedding endpoint.

        Falls back to the default Qwen3-Embedding-8B endpoint if no
        custom embedder is provided.
        """
        if self._embedder is not None and callable(self._embedder):
            result = self._embedder(text)
            if isinstance(result, list):
                return result
            # Assume awaitable
            return await result  # type: ignore[return-value]

        base_url = os.getenv("LLM_EMBEDDING_URL", _DEFAULT_EMBEDDING_URL)
        model = os.getenv("LLM_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/v1/embeddings",
                json={"input": text, "model": model},
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]  # type: ignore[no-any-return]

    async def search(
        self,
        query: str,
        limit: int = 5,
        task_id: str | None = None,
    ) -> list[ModelDecisionSearchResult]:
        """Search for decisions semantically similar to the query.

        Args:
            query: Natural language search query.
            limit: Maximum number of results to return.
            task_id: Optional filter to scope results to a specific task.

        Returns:
            List of decision search results ordered by similarity.
        """
        if self._qdrant is None:
            logger.warning("Qdrant client not configured; returning empty results")
            return []

        vector = await self._embed(query)

        query_filter = None
        if task_id is not None:
            query_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="task_id",
                        match=qdrant_models.MatchValue(value=task_id),
                    ),
                ],
            )

        query_result = self._qdrant.query_points(
            collection_name=_COLLECTION_NAME,
            query=vector,
            query_filter=query_filter,
            limit=limit,
        )

        return [
            ModelDecisionSearchResult(
                task_id=str(pt.payload.get("task_id", "")) if pt.payload else "",
                decision_text=str(pt.payload.get("decision_text", ""))
                if pt.payload
                else "",
                context=str(pt.payload.get("context", "")) if pt.payload else "",
                score=round(pt.score, 4),
                timestamp=str(pt.payload.get("timestamp", "")) if pt.payload else "",
            )
            for pt in query_result.points
        ]

    async def search_related(
        self,
        task_id: str,
        limit: int = 5,
    ) -> list[ModelDecisionSearchResult]:
        """Find decisions from OTHER tasks similar to a given task's decisions.

        Retrieves decisions for the given task_id, then searches for
        semantically similar decisions from other tasks.

        Args:
            task_id: The task whose decisions to find related matches for.
            limit: Maximum number of results to return.

        Returns:
            List of related decisions from other tasks.
        """
        if self._qdrant is None:
            logger.warning("Qdrant client not configured; returning empty results")
            return []

        # First, get decisions for the given task
        task_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="task_id",
                    match=qdrant_models.MatchValue(value=task_id),
                ),
            ],
        )

        task_decisions = self._qdrant.scroll(
            collection_name=_COLLECTION_NAME,
            scroll_filter=task_filter,
            limit=10,
            with_vectors=True,
        )

        points, _ = task_decisions
        if not points:
            logger.info("No decisions found for task %s", task_id)
            return []

        # Use the first decision's vector as the search query
        # and exclude the source task from results
        search_vector = points[0].vector
        if not isinstance(search_vector, list):
            logger.warning("Unexpected vector type for task %s", task_id)
            return []

        exclude_filter = qdrant_models.Filter(
            must_not=[
                qdrant_models.FieldCondition(
                    key="task_id",
                    match=qdrant_models.MatchValue(value=task_id),
                ),
            ],
        )

        query_result = self._qdrant.query_points(
            collection_name=_COLLECTION_NAME,
            query=search_vector,
            query_filter=exclude_filter,
            limit=limit,
        )

        return [
            ModelDecisionSearchResult(
                task_id=str(pt.payload.get("task_id", "")) if pt.payload else "",
                decision_text=str(pt.payload.get("decision_text", ""))
                if pt.payload
                else "",
                context=str(pt.payload.get("context", "")) if pt.payload else "",
                score=round(pt.score, 4),
                timestamp=str(pt.payload.get("timestamp", "")) if pt.payload else "",
            )
            for pt in query_result.points
        ]

    def format_results(self, results: list[ModelDecisionSearchResult]) -> str:
        """Format search results as human-readable text for context injection.

        Args:
            results: List of decision search results to format.

        Returns:
            Formatted string suitable for injecting into agent context.
        """
        if not results:
            return "No related decisions found."

        lines: list[str] = ["--- RELATED DECISIONS ---"]
        for r in results:
            lines.append(f"[{r.task_id}] (score: {r.score}) {r.decision_text}")
            if r.context:
                lines.append(f"  Context: {r.context}")
            if r.timestamp:
                lines.append(f"  Timestamp: {r.timestamp}")
            lines.append("")
        lines.append("---")
        return "\n".join(lines)


__all__ = ["DecisionSearchClient", "ModelDecisionSearchResult"]
