# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for AdapterSimilarityEnrichment.

Covers:
- handler_type / handler_category properties
- enrich() with results from Qdrant (top-3)
- enrich() when Qdrant returns no results
- token_count estimation (len(summary) // 4)
- schema_version default "1.0"
- enrichment_type is "similarity"
- protocol compliance with ProtocolContextEnrichment
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.adapters.llm.adapter_similarity_enrichment import (
    _PROMPT_VERSION,
    AdapterSimilarityEnrichment,
)
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_spi.contracts.enrichment.contract_enrichment_result import (
    ContractEnrichmentResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedding_response(vector: list[float] | None = None) -> MagicMock:
    """Build a minimal ModelLlmEmbeddingResponse mock."""
    if vector is None:
        vector = [0.1, 0.2, 0.3, 0.4]
    embedding = MagicMock()
    embedding.vector = vector
    resp = MagicMock()
    resp.embeddings = (embedding,)
    return resp


def _make_search_result(
    result_id: str, score: float, metadata: dict[str, object] | None = None
) -> MagicMock:
    """Build a minimal ModelVectorSearchResult mock."""
    result = MagicMock()
    result.id = result_id
    result.score = score
    result.metadata = metadata or {}
    return result


def _make_search_results(results: list[MagicMock]) -> MagicMock:
    """Build a minimal ModelVectorSearchResults mock."""
    search_results = MagicMock()
    search_results.results = results
    return search_results


def _make_adapter(**kwargs: object) -> AdapterSimilarityEnrichment:
    """Build an AdapterSimilarityEnrichment with mocked handlers.

    Patches HandlerEmbeddingOpenaiCompatible and HandlerQdrant so tests
    do not make real network calls.
    """
    adapter = AdapterSimilarityEnrichment(
        embedding_base_url="http://localhost:8002",
        qdrant_url="http://localhost:6333",
        **kwargs,  # type: ignore[arg-type]
    )
    # Replace the embedding handler with a mock.
    adapter._embedding_handler = AsyncMock()
    return adapter


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterSimilarityEnrichmentProperties:
    """Tests for classification properties."""

    def test_handler_type(self) -> None:
        adapter = _make_adapter()
        assert adapter.handler_type is EnumHandlerType.INFRA_HANDLER

    def test_handler_category(self) -> None:
        adapter = _make_adapter()
        assert adapter.handler_category is EnumHandlerTypeCategory.EFFECT


# ---------------------------------------------------------------------------
# enrich() tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterSimilarityEnrichmentEnrich:
    """Tests for the enrich() method."""

    @pytest.mark.asyncio
    async def test_enrich_with_results(self) -> None:
        """When Qdrant returns results, ContractEnrichmentResult is well-formed."""
        adapter = _make_adapter()
        adapter._embedding_handler.execute = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_embedding_response([0.1, 0.2, 0.3])
        )

        qdrant_results = [
            _make_search_result("session-1", 0.92),
            _make_search_result("session-2", 0.85),
            _make_search_result("session-3", 0.78),
        ]
        mock_qdrant = AsyncMock()
        mock_qdrant.query_similar = AsyncMock(
            return_value=_make_search_results(qdrant_results)
        )
        adapter._qdrant_handler = mock_qdrant

        result = await adapter.enrich(
            prompt="How does registration work?",
            context="",
        )

        assert isinstance(result, ContractEnrichmentResult)
        assert result.enrichment_type == "similarity"
        assert "Similar Past Sessions" in result.summary_markdown
        assert "Session 1" in result.summary_markdown
        assert "Session 2" in result.summary_markdown
        assert "Session 3" in result.summary_markdown
        assert result.relevance_score > 0.0
        assert result.model_used == "gte-qwen2-1.5b"
        assert result.prompt_version == _PROMPT_VERSION
        assert result.latency_ms >= 0.0

        mock_qdrant.query_similar.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_enrich_no_qdrant_results(self) -> None:
        """When Qdrant returns no results, summary contains 'No Similar Sessions Found'."""
        adapter = _make_adapter()
        adapter._embedding_handler.execute = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_embedding_response()
        )

        mock_qdrant = AsyncMock()
        mock_qdrant.query_similar = AsyncMock(return_value=_make_search_results([]))
        adapter._qdrant_handler = mock_qdrant

        result = await adapter.enrich(
            prompt="Something unusual",
            context="",
        )

        assert result.enrichment_type == "similarity"
        assert "No Similar Sessions Found" in result.summary_markdown
        assert result.relevance_score == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_enrich_token_count_estimated(self) -> None:
        """token_count is estimated as len(summary) // 4."""
        adapter = _make_adapter()
        adapter._embedding_handler.execute = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_embedding_response()
        )

        qdrant_results = [_make_search_result("s-1", 0.9)]
        mock_qdrant = AsyncMock()
        mock_qdrant.query_similar = AsyncMock(
            return_value=_make_search_results(qdrant_results)
        )
        adapter._qdrant_handler = mock_qdrant

        result = await adapter.enrich(prompt="Q", context="")

        expected_token_count = max(0, len(result.summary_markdown) // 4)
        assert result.token_count == expected_token_count

    @pytest.mark.asyncio
    async def test_enrich_result_schema_version(self) -> None:
        """schema_version defaults to '1.0'."""
        adapter = _make_adapter()
        adapter._embedding_handler.execute = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_embedding_response()
        )

        mock_qdrant = AsyncMock()
        mock_qdrant.query_similar = AsyncMock(
            return_value=_make_search_results([_make_search_result("s-1", 0.8)])
        )
        adapter._qdrant_handler = mock_qdrant

        result = await adapter.enrich(prompt="Test", context="ignored")

        assert result.schema_version == "1.0"

    @pytest.mark.asyncio
    async def test_enrich_type_is_similarity(self) -> None:
        """enrichment_type is always 'similarity'."""
        adapter = _make_adapter()
        adapter._embedding_handler.execute = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_embedding_response()
        )

        mock_qdrant = AsyncMock()
        mock_qdrant.query_similar = AsyncMock(
            return_value=_make_search_results([_make_search_result("s-1", 0.75)])
        )
        adapter._qdrant_handler = mock_qdrant

        result = await adapter.enrich(prompt="Test", context="")

        assert result.enrichment_type == "similarity"

    @pytest.mark.asyncio
    async def test_enrich_relevance_score_is_average(self) -> None:
        """relevance_score is the average of returned Qdrant scores."""
        adapter = _make_adapter()
        adapter._embedding_handler.execute = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_embedding_response()
        )

        qdrant_results = [
            _make_search_result("s-1", 0.90),
            _make_search_result("s-2", 0.70),
        ]
        mock_qdrant = AsyncMock()
        mock_qdrant.query_similar = AsyncMock(
            return_value=_make_search_results(qdrant_results)
        )
        adapter._qdrant_handler = mock_qdrant

        result = await adapter.enrich(prompt="Test", context="")

        # Average of 0.90 and 0.70 is 0.80
        assert result.relevance_score == pytest.approx(0.80, abs=1e-6)

    @pytest.mark.asyncio
    async def test_enrich_context_is_ignored(self) -> None:
        """The context parameter does not affect similarity results."""
        adapter = _make_adapter()
        adapter._embedding_handler.execute = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_embedding_response()
        )

        qdrant_results = [_make_search_result("s-1", 0.88)]
        mock_qdrant = AsyncMock()
        mock_qdrant.query_similar = AsyncMock(
            return_value=_make_search_results(qdrant_results)
        )
        adapter._qdrant_handler = mock_qdrant

        result_empty_ctx = await adapter.enrich(prompt="Same prompt", context="")

        # Reset mock call count for second call
        adapter._embedding_handler.execute = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_embedding_response()
        )
        mock_qdrant.query_similar = AsyncMock(
            return_value=_make_search_results(qdrant_results)
        )

        result_with_ctx = await adapter.enrich(
            prompt="Same prompt", context="Some context that should be ignored"
        )

        assert result_empty_ctx.enrichment_type == result_with_ctx.enrichment_type
        assert result_empty_ctx.model_used == result_with_ctx.model_used

    @pytest.mark.asyncio
    async def test_enrich_embedding_error_propagates(self) -> None:
        """Errors from the embedding handler propagate out of enrich()."""
        from omnibase_infra.enums import EnumInfraTransportType
        from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext

        adapter = _make_adapter()
        ctx = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.HTTP,
            operation="execute",
        )
        adapter._embedding_handler.execute = AsyncMock(  # type: ignore[method-assign]
            side_effect=InfraConnectionError("Embedding endpoint down", context=ctx)
        )

        with pytest.raises(InfraConnectionError):
            await adapter.enrich(prompt="Test", context="")

    @pytest.mark.asyncio
    async def test_enrich_qdrant_error_propagates(self) -> None:
        """Errors from HandlerQdrant propagate out of enrich()."""
        from omnibase_infra.enums import EnumInfraTransportType
        from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext

        adapter = _make_adapter()
        adapter._embedding_handler.execute = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_embedding_response()
        )

        ctx = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.QDRANT,
            operation="query_similar",
        )
        mock_qdrant = AsyncMock()
        mock_qdrant.query_similar = AsyncMock(
            side_effect=InfraConnectionError("Qdrant unavailable", context=ctx)
        )
        adapter._qdrant_handler = mock_qdrant

        with pytest.raises(InfraConnectionError):
            await adapter.enrich(prompt="Test", context="")

    @pytest.mark.asyncio
    async def test_enrich_scores_in_result_markdown(self) -> None:
        """Scores are included in the Markdown output."""
        adapter = _make_adapter()
        adapter._embedding_handler.execute = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_embedding_response()
        )

        qdrant_results = [
            _make_search_result("s-1", 0.95),
            _make_search_result("s-2", 0.72),
        ]
        mock_qdrant = AsyncMock()
        mock_qdrant.query_similar = AsyncMock(
            return_value=_make_search_results(qdrant_results)
        )
        adapter._qdrant_handler = mock_qdrant

        result = await adapter.enrich(prompt="Test", context="")

        assert "0.95" in result.summary_markdown
        assert "0.72" in result.summary_markdown


# ---------------------------------------------------------------------------
# Lazy Qdrant initialization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdapterSimilarityEnrichmentQdrantInit:
    """Tests for lazy Qdrant initialization."""

    @pytest.mark.asyncio
    async def test_qdrant_initialized_lazily(self) -> None:
        """HandlerQdrant is created and initialized on first enrich() call."""
        adapter = _make_adapter()
        adapter._embedding_handler.execute = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_embedding_response()
        )

        assert adapter._qdrant_handler is None

        mock_handler = AsyncMock()
        mock_handler.query_similar = AsyncMock(return_value=_make_search_results([]))

        with patch(
            "omnibase_infra.adapters.llm.adapter_similarity_enrichment.HandlerQdrant",
            return_value=mock_handler,
        ):
            with patch.object(mock_handler, "initialize", new=AsyncMock()):
                await adapter.enrich(prompt="Test", context="")

        # After the call, _qdrant_handler is set.
        assert adapter._qdrant_handler is not None

    @pytest.mark.asyncio
    async def test_qdrant_not_re_initialized_on_second_call(self) -> None:
        """HandlerQdrant.initialize is called only once across multiple enrich() calls."""
        adapter = _make_adapter()
        adapter._embedding_handler.execute = AsyncMock(  # type: ignore[method-assign]
            return_value=_make_embedding_response()
        )

        mock_qdrant = AsyncMock()
        mock_qdrant.query_similar = AsyncMock(return_value=_make_search_results([]))
        # Pre-set the handler so _ensure_qdrant_initialized is a no-op.
        adapter._qdrant_handler = mock_qdrant

        await adapter.enrich(prompt="First call", context="")
        await adapter.enrich(prompt="Second call", context="")

        # initialize() should never be called since _qdrant_handler was pre-set.
        mock_qdrant.initialize.assert_not_awaited()


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProtocolContextEnrichmentCompliance:
    """Verify AdapterSimilarityEnrichment satisfies ProtocolContextEnrichment."""

    def test_isinstance_check(self) -> None:
        """isinstance() against ProtocolContextEnrichment is True."""
        from omnibase_spi.protocols.intelligence.protocol_context_enrichment import (
            ProtocolContextEnrichment,
        )

        adapter = _make_adapter()
        assert isinstance(adapter, ProtocolContextEnrichment)

    def test_has_enrich_method(self) -> None:
        """enrich() is callable on the adapter."""
        adapter = _make_adapter()
        assert callable(getattr(adapter, "enrich", None))
