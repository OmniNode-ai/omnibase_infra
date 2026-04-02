# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for chain learning handlers."""

from __future__ import annotations

from datetime import UTC, datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_chain_orchestrator.handlers.handler_chain_learn_initiate import (
    HandlerChainLearnInitiate,
)
from omnibase_infra.nodes.node_chain_orchestrator.handlers.handler_chain_replay_complete import (
    HandlerChainReplayComplete,
)
from omnibase_infra.nodes.node_chain_orchestrator.handlers.handler_chain_retrieval_complete import (
    HandlerChainRetrievalComplete,
)
from omnibase_infra.nodes.node_chain_orchestrator.handlers.handler_chain_store_complete import (
    HandlerChainStoreComplete,
)
from omnibase_infra.nodes.node_chain_orchestrator.models import (
    EnumChainVerifyState,
    ModelChainEntry,
    ModelChainMatch,
    ModelChainReplayResult,
    ModelChainRetrievalResult,
    ModelChainStep,
    ModelChainStoreResult,
)
from omnibase_infra.nodes.node_chain_replay_compute.handlers.handler_chain_replay import (
    HandlerChainReplay,
)
from omnibase_infra.nodes.node_chain_verify_reducer.handlers.handler_chain_verify import (
    HandlerChainVerify,
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


# ---- Orchestrator handlers ----


@pytest.mark.unit
class TestHandlerChainLearnInitiate:
    @pytest.mark.asyncio
    async def test_returns_dispatch_params(self) -> None:
        handler = HandlerChainLearnInitiate()
        cid = uuid4()
        result = await handler.handle(
            prompt_text="hello world",
            workflow_ref="test_wf",
            correlation_id=cid,
            similarity_threshold=0.9,
        )
        assert result["prompt_text"] == "hello world"
        assert result["workflow_ref"] == "test_wf"
        assert result["similarity_threshold"] == "0.9"


@pytest.mark.unit
class TestHandlerChainRetrievalComplete:
    @pytest.mark.asyncio
    async def test_hit_path(self) -> None:
        handler = HandlerChainRetrievalComplete()
        entry = _make_entry()
        match = ModelChainMatch(chain_entry=entry, similarity_score=0.92, distance=0.08)
        retrieval = ModelChainRetrievalResult(
            correlation_id=uuid4(),
            matches=(match,),
            best_match_similarity=0.92,
            is_hit=True,
        )
        result = await handler.handle(retrieval, uuid4())
        assert result["path"] == "replay"
        assert result["is_hit"] == "true"

    @pytest.mark.asyncio
    async def test_miss_path(self) -> None:
        handler = HandlerChainRetrievalComplete()
        retrieval = ModelChainRetrievalResult(
            correlation_id=uuid4(),
            is_hit=False,
        )
        result = await handler.handle(retrieval, uuid4())
        assert result["path"] == "explore"
        assert result["is_hit"] == "false"


@pytest.mark.unit
class TestHandlerChainReplayComplete:
    @pytest.mark.asyncio
    async def test_high_confidence_stores(self) -> None:
        handler = HandlerChainReplayComplete()
        replay = ModelChainReplayResult(
            correlation_id=uuid4(),
            adapted_steps=(_make_step(),),
            adaptation_summary="No changes",
            confidence=0.95,
        )
        result = await handler.handle(replay, uuid4())
        assert result["action"] == "store"

    @pytest.mark.asyncio
    async def test_low_confidence_fallback(self) -> None:
        handler = HandlerChainReplayComplete()
        replay = ModelChainReplayResult(
            correlation_id=uuid4(),
            adapted_steps=(_make_step(),),
            adaptation_summary="Many changes",
            confidence=0.3,
        )
        result = await handler.handle(replay, uuid4())
        assert result["action"] == "fallback"


@pytest.mark.unit
class TestHandlerChainStoreComplete:
    @pytest.mark.asyncio
    async def test_success(self) -> None:
        handler = HandlerChainStoreComplete()
        store_result = ModelChainStoreResult(
            correlation_id=uuid4(),
            chain_id=uuid4(),
            success=True,
        )
        result = await handler.handle(store_result, uuid4())
        assert result["status"] == "complete"
        assert result["success"] == "true"

    @pytest.mark.asyncio
    async def test_failure(self) -> None:
        handler = HandlerChainStoreComplete()
        store_result = ModelChainStoreResult(
            correlation_id=uuid4(),
            chain_id=uuid4(),
            success=False,
            error_message="connection refused",
        )
        result = await handler.handle(store_result, uuid4())
        assert result["success"] == "false"


# ---- Compute handler ----


@pytest.mark.unit
class TestHandlerChainReplay:
    @pytest.mark.asyncio
    async def test_exact_replay_high_confidence(self) -> None:
        handler = HandlerChainReplay()
        entry = _make_entry()
        result = await handler.handle(
            cached_chain=entry,
            new_prompt_text="new prompt",
            correlation_id=uuid4(),
        )
        assert result.confidence == 0.95
        assert len(result.adapted_steps) == len(entry.chain_steps)

    @pytest.mark.asyncio
    async def test_context_substitution_lowers_confidence(self) -> None:
        handler = HandlerChainReplay()
        step = ModelChainStep(
            step_index=0,
            node_ref="node_test",
            operation="process_$FILE",
            input_hash="a",
            output_hash="b",
            duration_ms=50,
            event_topic="onex.evt.test.v1",
        )
        entry = ModelChainEntry(
            chain_id=uuid4(),
            prompt_text="original",
            prompt_hash="h",
            chain_steps=(step,),
            contract_hash="c",
            success_timestamp=datetime.now(UTC),
            workflow_ref="w",
        )
        result = await handler.handle(
            cached_chain=entry,
            new_prompt_text="new",
            correlation_id=uuid4(),
            new_context={"$FILE": "readme.md"},
        )
        assert result.confidence < 0.95
        assert "readme.md" in result.adapted_steps[0].operation


# ---- FSM handler ----


@pytest.mark.unit
class TestHandlerChainVerify:
    def test_valid_transitions(self) -> None:
        handler = HandlerChainVerify()

        # pending -> retrieving
        state, intents = handler.delta("pending", "cmd_received")
        assert state == EnumChainVerifyState.RETRIEVING
        assert intents == []

        # retrieving -> replaying (hit)
        state, _ = handler.delta("retrieving", "retrieval_hit")
        assert state == EnumChainVerifyState.REPLAYING

        # retrieving -> exploring (miss)
        state, _ = handler.delta("retrieving", "retrieval_miss")
        assert state == EnumChainVerifyState.EXPLORING

        # replaying -> verifying
        state, _ = handler.delta("replaying", "replay_complete")
        assert state == EnumChainVerifyState.VERIFYING

        # verifying -> complete
        state, _ = handler.delta("verifying", "verify_success")
        assert state == EnumChainVerifyState.COMPLETE

        # verifying -> fallback
        state, _ = handler.delta("verifying", "verify_failed")
        assert state == EnumChainVerifyState.FALLBACK

        # fallback -> exploring
        state, _ = handler.delta("fallback", "fallback_to_explore")
        assert state == EnumChainVerifyState.EXPLORING

    def test_error_from_any_non_terminal(self) -> None:
        handler = HandlerChainVerify()
        for state_name in (
            "retrieving",
            "replaying",
            "exploring",
            "verifying",
            "fallback",
        ):
            new_state, _ = handler.delta(state_name, "error")
            assert new_state == EnumChainVerifyState.FAILED

    def test_invalid_transition_raises(self) -> None:
        handler = HandlerChainVerify()
        with pytest.raises(ValueError, match="Invalid transition"):
            handler.delta("pending", "retrieval_hit")

    def test_terminal_state_raises(self) -> None:
        handler = HandlerChainVerify()
        with pytest.raises(ValueError, match="terminal state"):
            handler.delta("complete", "cmd_received")

        with pytest.raises(ValueError, match="terminal state"):
            handler.delta("failed", "error")
