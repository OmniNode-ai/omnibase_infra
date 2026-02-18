# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Embedding similarity enrichment adapter for ProtocolContextEnrichment.

Embeds the prompt using the GTE-Qwen2 embedding model, queries Qdrant for
the most similar past sessions or patterns, and returns the top-K results
as a structured Markdown summary.

Architecture:
    - Implements ProtocolContextEnrichment from omnibase_spi
    - Delegates embedding generation to HandlerEmbeddingOpenaiCompatible
    - Delegates similarity search to HandlerQdrant (lazy-initialized)
    - Returns ContractEnrichmentResult with enrichment_type="similarity"

Embedding Strategy:
    The ``prompt`` parameter is embedded to form the query vector.  The
    ``context`` parameter is intentionally ignored by this adapter because
    the semantic search is driven entirely by the prompt.  Callers that
    need to factor in raw context should use a summarization enricher first.

Token Estimation:
    Token count is estimated at 4 characters per token (rough heuristic).
    Actual counts depend on the model tokenizer but this is sufficient
    for budget accounting purposes.

Related Tickets:
    - OMN-2261: Embedding similarity enrichment handler
    - OMN-2252: ProtocolContextEnrichment SPI contract
    - OMN-2112: HandlerEmbeddingOpenaiCompatible
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.models.vector import (
    ModelVectorConnectionConfig,
    ModelVectorSearchResult,
)
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
)
from omnibase_infra.handlers.handler_qdrant import HandlerQdrant
from omnibase_infra.nodes.node_llm_embedding_effect.handlers.handler_embedding_openai_compatible import (
    HandlerEmbeddingOpenaiCompatible,
)
from omnibase_infra.nodes.node_llm_embedding_effect.models.model_llm_embedding_request import (
    ModelLlmEmbeddingRequest,
)
from omnibase_spi.contracts.enrichment.contract_enrichment_result import (
    ContractEnrichmentResult,
)

logger = logging.getLogger(__name__)

# Prompt version -- bump this when the markdown template changes.
_PROMPT_VERSION: str = "v1.0"

# Default embedding model identifier.
_DEFAULT_EMBEDDING_MODEL: str = "gte-qwen2-1.5b"

# Default Qdrant collection name for session/pattern storage.
_DEFAULT_COLLECTION: str = "onex_sessions"

# Default number of similar results to retrieve.
_DEFAULT_TOP_K: int = 3

# Rough token estimation: 4 characters per token.
_CHARS_PER_TOKEN: int = 4

# Markdown header for the no-results edge case.
_NO_RESULTS_MARKDOWN: str = "## No Similar Sessions Found"


class AdapterSimilarityEnrichment:
    """Context enrichment adapter that searches for similar past sessions.

    Implements ``ProtocolContextEnrichment``.  The ``prompt`` is embedded
    via GTE-Qwen2, and the resulting vector is used to query Qdrant for
    the top-K most similar stored entries.  Results are returned as a
    Markdown summary.

    The ``context`` parameter accepted by ``enrich()`` is not used in the
    similarity computation -- the search is prompt-driven by design.

    Attributes:
        handler_type: ``INFRA_HANDLER`` -- infrastructure-level handler.
        handler_category: ``EFFECT`` -- performs external I/O (HTTP + Qdrant).

    Example:
        >>> adapter = AdapterSimilarityEnrichment()
        >>> result = await adapter.enrich(
        ...     prompt="How does the registration orchestrator work?",
        ...     context="",
        ... )
        >>> print(result.summary_markdown)
    """

    def __init__(
        self,
        embedding_base_url: str | None = None,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
        qdrant_url: str | None = None,
        qdrant_collection: str = _DEFAULT_COLLECTION,
        top_k: int = _DEFAULT_TOP_K,
        score_threshold: float | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            embedding_base_url: Base URL of the GTE-Qwen2 embedding endpoint.
                Defaults to the ``LLM_EMBEDDING_URL`` environment variable.
                Required -- raises ``ValueError`` when unset.
            embedding_model: Model identifier sent in embedding requests.
            qdrant_url: Base URL of the Qdrant instance.  Defaults to the
                ``QDRANT_URL`` environment variable.
                Required -- raises ``ValueError`` when unset.
            qdrant_collection: Name of the Qdrant collection to query.
            top_k: Maximum number of similar results to return.
            score_threshold: Optional minimum similarity score filter applied
                by Qdrant before results are returned.  ``None`` disables the
                threshold (all results up to ``top_k`` are returned).
        """
        _embedding_base_url = embedding_base_url or os.environ.get("LLM_EMBEDDING_URL")
        if not _embedding_base_url:
            raise ValueError(
                "embedding_base_url is required. Set LLM_EMBEDDING_URL environment variable."
            )
        self._embedding_base_url: str = _embedding_base_url
        self._embedding_model: str = embedding_model
        _qdrant_url = qdrant_url or os.environ.get("QDRANT_URL")
        if not _qdrant_url:
            raise ValueError(
                "qdrant_url is required. Set QDRANT_URL environment variable."
            )
        self._qdrant_url: str = _qdrant_url
        self._qdrant_collection: str = qdrant_collection
        self._top_k: int = top_k
        self._score_threshold: float | None = score_threshold

        self._embedding_handler = HandlerEmbeddingOpenaiCompatible(
            "gte-qwen2-enrichment"
        )
        # Qdrant handler is lazily initialized on first use.
        self._qdrant_handler: HandlerQdrant | None = None
        # Lock to prevent double-initialization under concurrent asyncio.gather() calls.
        self._qdrant_init_lock: asyncio.Lock = asyncio.Lock()

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: INFRA_HANDLER."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: EFFECT (embedding HTTP + Qdrant query)."""
        return EnumHandlerTypeCategory.EFFECT

    async def enrich(
        self,
        prompt: str,
        context: str,
    ) -> ContractEnrichmentResult:
        """Enrich a prompt by finding similar past sessions via Qdrant.

        Embeds ``prompt`` using GTE-Qwen2, queries Qdrant for the top-K
        most similar entries, and formats the results as Markdown.

        The ``context`` argument is accepted for protocol compliance but is
        not used in the similarity computation -- the search is driven
        entirely by the prompt vector.

        Args:
            prompt: The user prompt or query to embed and search against.
            context: Unused.  Accepted for ``ProtocolContextEnrichment``
                protocol compliance.

        Returns:
            ``ContractEnrichmentResult`` with:

            - ``enrichment_type="similarity"``
            - ``summary_markdown``: Markdown-formatted list of similar sessions,
              or ``"## No Similar Sessions Found"`` when Qdrant returns nothing.
            - ``token_count``: Estimated token count of the summary.
            - ``relevance_score``: Average Qdrant similarity score across
              returned results; 0.0 when no results are found.
            - ``model_used``: Embedding model identifier.
            - ``prompt_version``: Template version (``"v1.0"``).
            - ``latency_ms``: End-to-end wall time in milliseconds.

        Raises:
            InfraConnectionError: Propagated from ``HandlerQdrant`` on
                connection failures or Qdrant timeouts.
            InfraProtocolError: Propagated from
                ``HandlerEmbeddingOpenaiCompatible`` on malformed embedding
                responses.
        """
        start = time.perf_counter()

        # Step 1: Embed the prompt.
        embedding_request = ModelLlmEmbeddingRequest(
            base_url=self._embedding_base_url,
            model=self._embedding_model,
            texts=(prompt,),
        )
        embedding_response = await self._embedding_handler.execute(embedding_request)
        query_vector = list(embedding_response.embeddings[0].vector)

        # Step 2: Ensure Qdrant is initialized (lazy).
        await self._ensure_qdrant_initialized()
        if self._qdrant_handler is None:
            raise RuntimeError("Qdrant handler failed to initialize")

        # Step 3: Query Qdrant for similar vectors.
        search_results = await self._qdrant_handler.query_similar(
            query_vector=query_vector,
            top_k=self._top_k,
            index_name=self._qdrant_collection,
            score_threshold=self._score_threshold,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        # Step 4: Handle empty results edge case.
        if not search_results.results:
            logger.debug(
                "No similar sessions found in collection '%s'. latency_ms=%.1f",
                self._qdrant_collection,
                latency_ms,
            )
            return ContractEnrichmentResult(
                summary_markdown=_NO_RESULTS_MARKDOWN,
                token_count=len(_NO_RESULTS_MARKDOWN) // _CHARS_PER_TOKEN,
                relevance_score=0.0,
                enrichment_type="similarity",
                latency_ms=latency_ms,
                model_used=self._embedding_model,
                prompt_version=_PROMPT_VERSION,
            )

        # Step 5: Format results as Markdown.
        summary = _format_results_markdown(search_results.results)
        scores = [r.score for r in search_results.results]
        avg_score = sum(scores) / len(scores)
        # Clamp to [0.0, 1.0] for protocol compliance (relevance_score field constraint).
        relevance_score = max(0.0, min(1.0, avg_score))
        token_count = max(0, len(summary) // _CHARS_PER_TOKEN)

        logger.debug(
            "Similarity enrichment complete. "
            "collection=%s results=%d avg_score=%.3f token_count=%d latency_ms=%.1f",
            self._qdrant_collection,
            len(search_results.results),
            avg_score,
            token_count,
            latency_ms,
        )

        return ContractEnrichmentResult(
            summary_markdown=summary,
            token_count=token_count,
            relevance_score=relevance_score,
            enrichment_type="similarity",
            latency_ms=latency_ms,
            model_used=self._embedding_model,
            prompt_version=_PROMPT_VERSION,
        )

    async def close(self) -> None:
        """Close the adapter and release resources from both handlers.

        Closes ``HandlerQdrant`` (if it was ever initialized) and
        ``HandlerEmbeddingOpenaiCompatible`` in that order.  The Qdrant
        handler is guarded by a ``None`` check because it is lazily
        initialized -- callers that never triggered ``enrich()`` will
        not have an active Qdrant connection to close.

        The embedding handler is always closed unconditionally because it
        is created eagerly in ``__init__`` and may hold an open HTTP
        client session.

        Returns:
            None

        Raises:
            Exception: Any exception raised by the underlying handler
                ``close()`` implementations is propagated to the caller.
                Both handlers are closed sequentially; an error in Qdrant
                teardown will prevent the embedding handler from being
                closed.
        """
        if self._qdrant_handler is not None:
            await self._qdrant_handler.close()
        await self._embedding_handler.close()

    async def _ensure_qdrant_initialized(self) -> None:
        """Lazily initialize HandlerQdrant on the first call (thread-safe via asyncio.Lock).

        Creates a minimal ``ModelONEXContainer`` for HandlerQdrant interface
        compliance.  HandlerQdrant stores the container reference but never
        resolves services from it during ``initialize()`` or ``query_similar()``.
        The connection config is passed directly to ``initialize()``.

        This is confirmed by ``HandlerQdrant.__init__`` (handler_qdrant.py
        lines 122-135): ``self._container = container`` is assigned but only
        retained for future DI-based service resolution.  All current operations
        (``initialize``, ``query_similar``, et al.) use ``self._client`` and
        ``self._config`` exclusively.

        Thread safety:
            ``_qdrant_init_lock`` (an ``asyncio.Lock``) serializes concurrent
            callers.  Without the lock, two ``asyncio.gather()`` tasks could
            both evaluate ``self._qdrant_handler is not None`` as ``False``
            simultaneously, leading to double-initialization and a leaked
            Qdrant connection.  The check-then-act sequence is performed
            atomically while the lock is held, so only the first caller
            creates the handler; all subsequent callers return immediately.

        Returns:
            None.  On return, ``self._qdrant_handler`` is guaranteed to be
            a fully initialized ``HandlerQdrant`` instance.

        Raises:
            InfraConnectionError: If ``HandlerQdrant.initialize()`` cannot
                reach the configured Qdrant URL.
            Exception: Any other exception raised by
                ``HandlerQdrant.initialize()`` is propagated to the caller.
        """
        async with self._qdrant_init_lock:
            if self._qdrant_handler is not None:
                return

            container = ModelONEXContainer()
            handler = HandlerQdrant(container)
            config = ModelVectorConnectionConfig(url=self._qdrant_url)
            await handler.initialize(config)
            self._qdrant_handler = handler


def _format_results_markdown(
    results: list[ModelVectorSearchResult],
) -> str:
    """Format a list of vector search results as Markdown.

    Args:
        results: Non-empty list of ``ModelVectorSearchResult`` instances.

    Returns:
        Markdown string with a section header per result.
    """
    lines: list[str] = ["## Similar Past Sessions", ""]
    for i, result in enumerate(results, start=1):
        lines.append(f"### Session {i} (Score: {result.score:.2f})")
        # Include payload metadata if present.
        if result.metadata:
            for key, schema_value in result.metadata.items():
                value = str(schema_value)
                lines.append(f"- **{key}**: {value}")
        lines.append("")
    return "\n".join(lines).rstrip()


__all__: list[str] = ["AdapterSimilarityEnrichment"]
