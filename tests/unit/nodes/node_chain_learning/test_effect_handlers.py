# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for chain learning EFFECT handlers: retrieval and store.

These handlers perform external I/O and require mocked clients.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_chain_orchestrator.models import (
    ModelChainEntry,
    ModelChainStep,
)
from omnibase_infra.nodes.node_chain_retrieval_effect.handlers.handler_chain_retrieval import (
    COLLECTION_NAME,
    HandlerChainRetrieval,
)
from omnibase_infra.nodes.node_chain_store_effect.handlers.handler_chain_store import (
    HandlerChainStore,
)


def _make_step(index: int = 0) -> ModelChainStep:
    return ModelChainStep(
        step_index=index,
        node_ref="node_test",
        operation="test.op",
        input_hash="abc123",
        output_hash="def456",
        duration_ms=100,
        event_topic="onex.evt.test.v1",
    )


def _make_entry() -> ModelChainEntry:
    return ModelChainEntry(
        chain_id=uuid4(),
        prompt_text="test prompt",
        prompt_hash="sha256_hash",
        chain_steps=(_make_step(0), _make_step(1)),
        contract_hash="contract_sha256",
        success_timestamp=datetime.now(UTC),
        workflow_ref="test_workflow",
    )


def _make_qdrant_hit(entry: ModelChainEntry, score: float = 0.9) -> SimpleNamespace:
    """Create a mock Qdrant search hit."""
    return SimpleNamespace(
        id=str(entry.chain_id),
        score=score,
        payload=entry.model_dump(mode="json"),
    )


# ---- HandlerChainRetrieval ----


@pytest.mark.unit
class TestHandlerChainRetrieval:
    """Tests for the chain retrieval effect handler."""

    def _make_handler(
        self,
        embedding_result: list[float] | Exception | None = None,
        search_result: list[object] | Exception | None = None,
        collection_exists: bool = True,
    ) -> HandlerChainRetrieval:
        embedding_client = AsyncMock()
        if isinstance(embedding_result, Exception):
            embedding_client.get_embedding.side_effect = embedding_result
        else:
            embedding_client.get_embedding.return_value = (
                embedding_result or [0.1] * 128
            )

        qdrant_client = MagicMock()
        qdrant_client.collection_exists.return_value = collection_exists
        if isinstance(search_result, Exception):
            qdrant_client.search.side_effect = search_result
        else:
            qdrant_client.search.return_value = search_result or []

        return HandlerChainRetrieval(
            embedding_client=embedding_client,
            qdrant_client=qdrant_client,
            vector_size=128,
        )

    def test_handler_classification(self) -> None:
        handler = self._make_handler()
        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER
        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT

    @pytest.mark.asyncio
    async def test_no_matches_returns_miss(self) -> None:
        handler = self._make_handler(search_result=[])
        result = await handler.handle(
            prompt_text="hello",
            correlation_id=uuid4(),
            similarity_threshold=0.85,
        )
        assert not result.is_hit
        assert result.matches == ()
        assert result.best_match_similarity == 0.0
        assert len(result.query_embedding) == 128

    @pytest.mark.asyncio
    async def test_hit_above_threshold(self) -> None:
        entry = _make_entry()
        hit = _make_qdrant_hit(entry, score=0.92)
        handler = self._make_handler(search_result=[hit])

        result = await handler.handle(
            prompt_text="test prompt",
            correlation_id=uuid4(),
            similarity_threshold=0.85,
        )
        assert result.is_hit
        assert len(result.matches) == 1
        assert result.best_match_similarity == 0.92
        assert result.matches[0].chain_entry.chain_id == entry.chain_id

    @pytest.mark.asyncio
    async def test_miss_below_threshold(self) -> None:
        entry = _make_entry()
        hit = _make_qdrant_hit(entry, score=0.5)
        handler = self._make_handler(search_result=[hit])

        result = await handler.handle(
            prompt_text="different prompt",
            correlation_id=uuid4(),
            similarity_threshold=0.85,
        )
        assert not result.is_hit
        assert len(result.matches) == 1
        assert result.best_match_similarity == 0.5

    @pytest.mark.asyncio
    async def test_multiple_matches_sorted_by_similarity(self) -> None:
        entry1 = _make_entry()
        entry2 = _make_entry()
        hits = [
            _make_qdrant_hit(entry1, score=0.7),
            _make_qdrant_hit(entry2, score=0.95),
        ]
        handler = self._make_handler(search_result=hits)

        result = await handler.handle(
            prompt_text="test",
            correlation_id=uuid4(),
            similarity_threshold=0.85,
        )
        assert result.is_hit
        assert len(result.matches) == 2
        assert result.matches[0].similarity_score == 0.95
        assert result.matches[1].similarity_score == 0.7

    @pytest.mark.asyncio
    async def test_embedding_failure_returns_miss(self) -> None:
        handler = self._make_handler(
            embedding_result=RuntimeError("embedding endpoint down")
        )
        result = await handler.handle(
            prompt_text="test",
            correlation_id=uuid4(),
        )
        assert not result.is_hit
        assert result.matches == ()
        assert result.query_embedding == []

    @pytest.mark.asyncio
    async def test_qdrant_search_failure_returns_miss(self) -> None:
        handler = self._make_handler(search_result=ConnectionError("qdrant down"))
        result = await handler.handle(
            prompt_text="test",
            correlation_id=uuid4(),
        )
        assert not result.is_hit
        assert result.matches == ()
        assert len(result.query_embedding) == 128  # embedding succeeded

    @pytest.mark.asyncio
    async def test_invalid_payload_skipped(self) -> None:
        """Hits with unparseable payloads are skipped gracefully."""
        valid_entry = _make_entry()
        valid_hit = _make_qdrant_hit(valid_entry, score=0.9)
        invalid_hit = SimpleNamespace(
            id="bad-id",
            score=0.95,
            payload={"invalid": "data"},
        )
        handler = self._make_handler(search_result=[invalid_hit, valid_hit])

        result = await handler.handle(
            prompt_text="test",
            correlation_id=uuid4(),
            similarity_threshold=0.85,
        )
        assert len(result.matches) == 1
        assert result.matches[0].chain_entry.chain_id == valid_entry.chain_id

    @pytest.mark.asyncio
    async def test_collection_created_when_missing(self) -> None:
        handler = self._make_handler(collection_exists=False)
        # Should not raise -- collection creation is best-effort
        result = await handler.handle(
            prompt_text="test",
            correlation_id=uuid4(),
        )
        assert not result.is_hit

    @pytest.mark.asyncio
    async def test_correlation_id_propagated(self) -> None:
        cid = uuid4()
        handler = self._make_handler()
        result = await handler.handle(
            prompt_text="test",
            correlation_id=cid,
        )
        assert result.correlation_id == cid


# ---- HandlerChainStore ----


@pytest.mark.unit
class TestHandlerChainStore:
    """Tests for the chain store effect handler."""

    def _make_handler(
        self,
        upsert_side_effect: Exception | None = None,
    ) -> HandlerChainStore:
        qdrant_client = MagicMock()
        if upsert_side_effect:
            qdrant_client.upsert.side_effect = upsert_side_effect
        return HandlerChainStore(qdrant_client=qdrant_client, vector_size=128)

    def test_handler_classification(self) -> None:
        handler = self._make_handler()
        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER
        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT

    @pytest.mark.asyncio
    async def test_successful_store(self) -> None:
        handler = self._make_handler()
        entry = _make_entry()
        embedding = [0.1] * 128

        result = await handler.handle(
            chain_entry=entry,
            prompt_embedding=embedding,
            correlation_id=uuid4(),
        )
        assert result.success
        assert result.chain_id == entry.chain_id
        assert result.error_message == ""

    @pytest.mark.asyncio
    async def test_qdrant_upsert_failure(self) -> None:
        handler = self._make_handler(
            upsert_side_effect=ConnectionError("qdrant connection refused")
        )
        entry = _make_entry()

        result = await handler.handle(
            chain_entry=entry,
            prompt_embedding=[0.1] * 128,
            correlation_id=uuid4(),
        )
        assert not result.success
        assert result.chain_id == entry.chain_id
        assert "upsert failed" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_correlation_id_propagated(self) -> None:
        handler = self._make_handler()
        cid = uuid4()
        entry = _make_entry()

        result = await handler.handle(
            chain_entry=entry,
            prompt_embedding=[0.1] * 128,
            correlation_id=cid,
        )
        assert result.correlation_id == cid

    @pytest.mark.asyncio
    async def test_upsert_called_with_correct_collection(self) -> None:
        qdrant_client = MagicMock()
        handler = HandlerChainStore(qdrant_client=qdrant_client, vector_size=128)
        entry = _make_entry()

        await handler.handle(
            chain_entry=entry,
            prompt_embedding=[0.1] * 128,
            correlation_id=uuid4(),
        )

        qdrant_client.upsert.assert_called_once()
        call_kwargs = qdrant_client.upsert.call_args
        assert call_kwargs[1]["collection_name"] == COLLECTION_NAME
