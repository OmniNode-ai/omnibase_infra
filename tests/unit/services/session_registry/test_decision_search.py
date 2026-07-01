# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for semantic decision search client.

Part of the Multi-Session Coordination Layer (OMN-6850, Task 14).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.services.session_registry.decision_search import (
    DecisionSearchClient,
    ModelDecisionSearchResult,
)

_TEST_EMBEDDING_URL = "http://test-embed:8100"


@pytest.mark.unit
class TestModelDecisionSearchResult:
    """Tests for the ModelDecisionSearchResult Pydantic model."""

    def test_search_result_model(self) -> None:
        result = ModelDecisionSearchResult(
            task_id="OMN-1234",
            decision_text="Use Postgres for session state",
            context="Multi-session coordination",
            score=0.92,
            timestamp="2026-03-27T14:00:00Z",
        )
        assert result.score == 0.92
        assert result.task_id == "OMN-1234"
        assert result.decision_text == "Use Postgres for session state"
        assert result.context == "Multi-session coordination"
        assert result.timestamp == "2026-03-27T14:00:00Z"

    def test_search_result_frozen(self) -> None:
        result = ModelDecisionSearchResult(
            task_id="OMN-1234",
            decision_text="Use Postgres",
            score=0.5,
        )
        with pytest.raises(Exception):
            result.score = 0.9  # type: ignore[misc]

    def test_search_result_defaults(self) -> None:
        result = ModelDecisionSearchResult(
            task_id="OMN-1234",
            decision_text="Use Postgres",
            score=0.5,
        )
        assert result.context == ""
        assert result.timestamp == ""

    def test_search_result_score_bounds(self) -> None:
        with pytest.raises(Exception):
            ModelDecisionSearchResult(
                task_id="OMN-1234",
                decision_text="Test",
                score=1.5,
            )


@pytest.mark.unit
class TestDecisionSearchClientFormatResults:
    """Tests for DecisionSearchClient.format_results()."""

    def test_format_search_results(self) -> None:
        results = [
            ModelDecisionSearchResult(
                task_id="OMN-1234",
                decision_text="Use Postgres for session state",
                context="coordination design",
                score=0.92,
                timestamp="2026-03-27T14:00:00Z",
            ),
            ModelDecisionSearchResult(
                task_id="OMN-1230",
                decision_text="Use Redis for caching",
                context="performance optimization",
                score=0.71,
                timestamp="2026-03-26T10:00:00Z",
            ),
        ]
        client = DecisionSearchClient(qdrant=None, embedder=None)
        formatted = client.format_results(results)
        assert "OMN-1234" in formatted
        assert "0.92" in formatted
        assert "OMN-1230" in formatted
        assert "0.71" in formatted
        assert "RELATED DECISIONS" in formatted
        assert "Use Postgres for session state" in formatted
        assert "coordination design" in formatted

    def test_format_empty_results(self) -> None:
        client = DecisionSearchClient(qdrant=None, embedder=None)
        formatted = client.format_results([])
        assert formatted == "No related decisions found."

    def test_format_results_without_context(self) -> None:
        results = [
            ModelDecisionSearchResult(
                task_id="OMN-5000",
                decision_text="Skip integration tests",
                score=0.88,
            ),
        ]
        client = DecisionSearchClient(qdrant=None, embedder=None)
        formatted = client.format_results(results)
        assert "OMN-5000" in formatted
        assert "Context:" not in formatted


@pytest.mark.unit
class TestDecisionSearchClientSearch:
    """Tests for DecisionSearchClient.search() with mocked Qdrant."""

    @pytest.mark.asyncio
    async def test_search_returns_empty_without_qdrant(self) -> None:
        client = DecisionSearchClient(qdrant=None, embedder=None)
        results = await client.search("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_calls_qdrant(self) -> None:
        mock_qdrant = MagicMock()
        mock_hit = MagicMock()
        mock_hit.payload = {
            "task_id": "OMN-1234",
            "decision_text": "Use Kafka for events",
            "context": "architecture",
            "timestamp": "2026-03-27T10:00:00Z",
        }
        mock_hit.score = 0.95
        mock_query_result = MagicMock()
        mock_query_result.points = [mock_hit]
        mock_qdrant.query_points.return_value = mock_query_result

        embedder = MagicMock(return_value=[0.1, 0.2, 0.3])

        client = DecisionSearchClient(
            qdrant=mock_qdrant,
            embedder=embedder,
            embedding_base_url=_TEST_EMBEDDING_URL,
        )
        results = await client.search("event bus choice")

        assert len(results) == 1
        assert results[0].task_id == "OMN-1234"
        assert results[0].score == 0.95
        assert results[0].decision_text == "Use Kafka for events"
        mock_qdrant.query_points.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_task_id_filter(self) -> None:
        mock_qdrant = MagicMock()
        mock_query_result = MagicMock()
        mock_query_result.points = []
        mock_qdrant.query_points.return_value = mock_query_result

        embedder = MagicMock(return_value=[0.1, 0.2, 0.3])

        client = DecisionSearchClient(
            qdrant=mock_qdrant,
            embedder=embedder,
            embedding_base_url=_TEST_EMBEDDING_URL,
        )
        await client.search("test", task_id="OMN-5555")

        call_kwargs = mock_qdrant.query_points.call_args
        assert call_kwargs.kwargs.get("query_filter") is not None


@pytest.mark.unit
class TestDecisionSearchClientSearchRelated:
    """Tests for DecisionSearchClient.search_related() with mocked Qdrant."""

    @pytest.mark.asyncio
    async def test_search_related_returns_empty_without_qdrant(self) -> None:
        client = DecisionSearchClient(qdrant=None, embedder=None)
        results = await client.search_related("OMN-1234")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_related_empty_task(self) -> None:
        mock_qdrant = MagicMock()
        mock_qdrant.scroll.return_value = ([], None)

        client = DecisionSearchClient(qdrant=mock_qdrant, embedder=None)
        results = await client.search_related("OMN-9999")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_related_finds_decisions(self) -> None:
        mock_qdrant = MagicMock()

        # Mock scroll to return a decision with a vector
        mock_point = MagicMock()
        mock_point.vector = [0.1, 0.2, 0.3]
        mock_point.payload = {
            "task_id": "OMN-1234",
            "decision_text": "Use Kafka",
        }
        mock_qdrant.scroll.return_value = ([mock_point], None)

        # Mock query_points to return a related decision from another task
        mock_hit = MagicMock()
        mock_hit.payload = {
            "task_id": "OMN-5678",
            "decision_text": "Use RabbitMQ for queues",
            "context": "messaging",
            "timestamp": "2026-03-20T10:00:00Z",
        }
        mock_hit.score = 0.85
        mock_query_result = MagicMock()
        mock_query_result.points = [mock_hit]
        mock_qdrant.query_points.return_value = mock_query_result

        client = DecisionSearchClient(
            qdrant=mock_qdrant,
            embedder=None,
            embedding_base_url=_TEST_EMBEDDING_URL,
        )
        results = await client.search_related("OMN-1234")

        assert len(results) == 1
        assert results[0].task_id == "OMN-5678"
        assert results[0].score == 0.85


@pytest.mark.unit
class TestDecisionSearchClientUrlInjection:
    """Verify that embedding_base_url is injected at construction, not read from env."""

    def test_url_stored_from_constructor(self) -> None:
        """URL passed at construction is stored and not resolved from env."""
        client = DecisionSearchClient(
            embedding_base_url="http://injected-host:9999",
        )
        assert client._embedding_base_url == "http://injected-host:9999"

    def test_model_stored_from_constructor(self) -> None:
        """Embedding model passed at construction is stored."""
        client = DecisionSearchClient(
            embedding_base_url=_TEST_EMBEDDING_URL,
            embedding_model="custom-embed-model",
        )
        assert client._embedding_model == "custom-embed-model"

    def test_no_env_var_read_on_import(self) -> None:
        """LLM_EMBEDDING_URL is never read from env by DecisionSearchClient."""
        with patch("os.getenv", side_effect=AssertionError("env read")):
            client = DecisionSearchClient(
                embedding_base_url=_TEST_EMBEDDING_URL,
            )
        assert client._embedding_base_url == _TEST_EMBEDDING_URL

    @pytest.mark.asyncio
    async def test_embed_uses_injected_url_not_env(self) -> None:
        """_embed sends requests to the injected URL, ignoring any env var."""
        injected_url = "http://injected-embed:1234"
        requested: dict[str, object] = {}

        class _Response:
            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict[str, list[dict[str, list[float]]]]:
                return {"data": [{"embedding": [0.1, 0.2]}]}

        class _AsyncClient:
            def __init__(self, *, timeout: float) -> None:
                requested["timeout"] = timeout

            async def __aenter__(self) -> _AsyncClient:
                return self

            async def __aexit__(
                self,
                exc_type: object,
                exc: object,
                traceback: object,
            ) -> None:
                pass

            async def post(
                self,
                url: str,
                *,
                json: dict[str, str],
            ) -> _Response:
                requested["url"] = url
                requested["json"] = json
                return _Response()

        client = DecisionSearchClient(
            qdrant=MagicMock(),
            embedding_base_url=injected_url,
        )
        with (
            patch("os.getenv", side_effect=AssertionError("env read")),
            patch(
                "omnibase_infra.services.session_registry.decision_search.httpx.AsyncClient",
                _AsyncClient,
            ),
        ):
            result = await client._embed("test text")

        assert result == [0.1, 0.2]
        assert requested["url"] == f"{injected_url}/v1/embeddings"
        assert requested["json"] == {
            "input": "test text",
            "model": "Qwen3-Embedding-8B",
        }

    @pytest.mark.asyncio
    async def test_embed_raises_on_empty_url_without_embedder(self) -> None:
        """_embed fails fast when no custom embedder and URL is empty."""
        client = DecisionSearchClient(qdrant=MagicMock(), embedder=None)
        with pytest.raises(ValueError, match="embedding_base_url is empty"):
            await client._embed("test text")
