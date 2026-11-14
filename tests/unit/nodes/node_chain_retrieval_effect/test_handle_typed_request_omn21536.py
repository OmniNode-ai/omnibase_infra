# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed retrieval request parity, including backend failures and skipped hits."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_infra.nodes.node_chain_orchestrator.models import (
    ModelChainEntry,
    ModelChainRetrievalResult,
)
from omnibase_infra.nodes.node_chain_retrieval_effect.handlers.handler_chain_retrieval import (
    COLLECTION_NAME,
    HandlerChainRetrieval,
)

pytestmark = pytest.mark.unit
HANDLER_MODULE: str = (
    "omnibase_infra.nodes.node_chain_retrieval_effect.handlers.handler_chain_retrieval"
)
PRESERVED_SYMBOLS: tuple[str, ...] = (
    "HandlerChainRetrieval._get_embedding",
    "HandlerChainRetrieval._ensure_collection",
)


class ModelRetrievalScenario(BaseModel):
    """Deterministic embedding/search outcome and request settings."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    seed: int
    branch: Literal[
        "hit", "empty", "embedding_failure", "search_failure", "invalid", "no_client"
    ]
    similarity_threshold: float = 0.85
    limit: int = 5

    @property
    def correlation_id(self) -> UUID:
        return UUID(int=self.seed)


SCENARIOS: tuple[BaseModel, ...] = (
    ModelRetrievalScenario(seed=1, branch="hit", similarity_threshold=0.9, limit=2),
    ModelRetrievalScenario(seed=2, branch="empty"),
    ModelRetrievalScenario(seed=3, branch="embedding_failure"),
    ModelRetrievalScenario(seed=4, branch="search_failure"),
    ModelRetrievalScenario(seed=5, branch="invalid"),
    ModelRetrievalScenario(seed=6, branch="no_client"),
)


async def run_scenario(scenario: ModelRetrievalScenario) -> ModelChainRetrievalResult:
    """Call with one typed request and verify the exact external query."""
    try:
        from omnibase_infra.nodes.node_chain_retrieval_effect.models import (
            ModelChainRetrievalRequest,
        )
    except ImportError as exc:
        pytest.fail(f"the typed retrieval request model is missing: {exc}")
    request = ModelChainRetrievalRequest(
        prompt_text=f"prompt-{scenario.seed}",
        correlation_id=scenario.correlation_id,
        similarity_threshold=scenario.similarity_threshold,
        limit=scenario.limit,
    )
    embedding = [scenario.seed / 10, 0.5]
    embedding_client = AsyncMock()
    embedding_client.get_embedding.return_value = embedding
    client = MagicMock()
    client.collection_exists.return_value = True
    client.search.return_value = []
    if scenario.branch == "embedding_failure":
        embedding_client.get_embedding.side_effect = RuntimeError(
            "embedding unavailable"
        )
    elif scenario.branch == "search_failure":
        client.search.side_effect = ConnectionError("search unavailable")
    elif scenario.branch == "hit":
        entry = ModelChainEntry(
            chain_id=UUID(int=100 + scenario.seed),
            prompt_text=request.prompt_text,
            prompt_hash="hash",
            chain_steps=(),
            contract_hash="contract",
            success_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            workflow_ref="workflow",
        )
        client.search.return_value = [
            SimpleNamespace(
                id=str(entry.chain_id),
                score=score,
                payload=entry.model_dump(mode="json"),
            )
            for score in (0.6, 0.95)
        ]
    elif scenario.branch == "invalid":
        client.search.return_value = [
            SimpleNamespace(id="missing", score=0.99, payload=None),
            SimpleNamespace(id="invalid", score=0.98, payload={"bad": "payload"}),
        ]
    handler = HandlerChainRetrieval(
        embedding_client=embedding_client,
        qdrant_client=None if scenario.branch == "no_client" else client,
        vector_size=2,
    )
    try:
        result = await handler.handle(request)
    except TypeError as exc:
        pytest.fail(f"handle() does not accept the single request model: {exc}")
    embedding_client.get_embedding.assert_awaited_once_with(request.prompt_text)
    if scenario.branch in ("embedding_failure", "no_client"):
        client.collection_exists.assert_not_called()
        client.search.assert_not_called()
    else:
        client.collection_exists.assert_called_once_with(COLLECTION_NAME)
        client.search.assert_called_once_with(
            COLLECTION_NAME,
            query_vector=embedding,
            limit=scenario.limit,
            score_threshold=scenario.similarity_threshold * 0.5,
        )
    assert result.correlation_id == scenario.correlation_id
    return result


class TestRetrievalHandleTypedRequest:
    @pytest.mark.asyncio
    async def test_hits_are_sorted_and_custom_query_settings_forwarded(self) -> None:
        result = await run_scenario(SCENARIOS[0])
        assert result.is_hit is True
        assert result.best_match_similarity == 0.95
        assert tuple(match.similarity_score for match in result.matches) == (0.95, 0.6)
        assert result.matches[0].chain_entry.chain_id == UUID(int=101)
        assert result.query_embedding == [0.1, 0.5]

    @pytest.mark.asyncio
    async def test_empty_search_returns_a_miss(self) -> None:
        result = await run_scenario(SCENARIOS[1])
        assert result.is_hit is False
        assert result.matches == ()
        assert result.best_match_similarity == 0.0
        assert result.query_embedding == [0.2, 0.5]

    @pytest.mark.asyncio
    async def test_embedding_failure_stops_before_search(self) -> None:
        result = await run_scenario(SCENARIOS[2])
        assert result.is_hit is False
        assert result.matches == ()
        assert result.query_embedding == []

    @pytest.mark.asyncio
    async def test_search_failure_retains_embedding(self) -> None:
        result = await run_scenario(SCENARIOS[3])
        assert result.is_hit is False
        assert result.matches == ()
        assert result.query_embedding == [0.4, 0.5]

    @pytest.mark.asyncio
    async def test_malformed_hits_are_skipped(self) -> None:
        result = await run_scenario(SCENARIOS[4])
        assert result.is_hit is False
        assert result.matches == ()
        assert result.best_match_similarity == 0.0

    @pytest.mark.asyncio
    async def test_unconfigured_vector_client_returns_miss(self) -> None:
        result = await run_scenario(SCENARIOS[5])
        assert result.is_hit is False
        assert result.matches == ()
        assert result.query_embedding == [0.6, 0.5]
