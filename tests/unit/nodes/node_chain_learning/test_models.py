# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for chain learning models: frozen, extra=forbid, field constraints."""

from __future__ import annotations

from datetime import UTC, datetime, timezone
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_chain_orchestrator.models import (
    EnumChainVerifyState,
    ModelChainEntry,
    ModelChainLearnCommand,
    ModelChainLearnResult,
    ModelChainMatch,
    ModelChainReplayInput,
    ModelChainReplayResult,
    ModelChainRetrievalResult,
    ModelChainStep,
    ModelChainStoreRequest,
    ModelChainStoreResult,
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


@pytest.mark.unit
class TestModelChainStep:
    def test_frozen(self) -> None:
        step = _make_step()
        with pytest.raises(Exception):
            step.step_index = 99  # type: ignore[misc]

    def test_extra_forbid(self) -> None:
        with pytest.raises(Exception):
            ModelChainStep(
                step_index=0,
                node_name="n",
                operation="op",
                input_hash="a",
                output_hash="b",
                duration_ms=10,
                event_topic="t",
                extra_field="boom",  # type: ignore[call-arg]
            )

    def test_negative_step_index_rejected(self) -> None:
        with pytest.raises(Exception):
            ModelChainStep(
                step_index=-1,
                node_name="n",
                operation="op",
                input_hash="a",
                output_hash="b",
                duration_ms=10,
                event_topic="t",
            )


@pytest.mark.unit
class TestModelChainEntry:
    def test_frozen(self) -> None:
        entry = _make_entry()
        with pytest.raises(Exception):
            entry.prompt_text = "changed"  # type: ignore[misc]

    def test_round_trip(self) -> None:
        entry = _make_entry()
        data = entry.model_dump(mode="json")
        restored = ModelChainEntry.model_validate(data)
        assert restored.chain_id == entry.chain_id
        assert len(restored.chain_steps) == 2

    def test_similarity_threshold_bounds(self) -> None:
        with pytest.raises(Exception):
            ModelChainEntry(
                chain_id=uuid4(),
                prompt_text="test",
                prompt_hash="h",
                chain_steps=(),
                contract_hash="c",
                success_timestamp=datetime.now(UTC),
                workflow_ref="w",
                similarity_threshold=1.5,
            )


@pytest.mark.unit
class TestModelChainLearnCommand:
    def test_empty_prompt_rejected(self) -> None:
        with pytest.raises(Exception):
            ModelChainLearnCommand(
                correlation_id=uuid4(),
                prompt_text="",
                workflow_ref="w",
            )

    def test_valid(self) -> None:
        cmd = ModelChainLearnCommand(
            correlation_id=uuid4(),
            prompt_text="test prompt",
            workflow_ref="test_workflow",
        )
        assert cmd.similarity_threshold == 0.85


@pytest.mark.unit
class TestModelChainLearnResult:
    def test_valid_paths(self) -> None:
        for path in ("hit_replay", "miss_explore", "fallback_explore"):
            result = ModelChainLearnResult(
                correlation_id=uuid4(),
                path_taken=path,  # type: ignore[arg-type]
                success=True,
            )
            assert result.path_taken == path

    def test_invalid_path_rejected(self) -> None:
        with pytest.raises(Exception):
            ModelChainLearnResult(
                correlation_id=uuid4(),
                path_taken="invalid",  # type: ignore[arg-type]
                success=True,
            )


@pytest.mark.unit
class TestModelChainRetrievalResult:
    def test_default_miss(self) -> None:
        result = ModelChainRetrievalResult(correlation_id=uuid4())
        assert not result.is_hit
        assert result.best_match_similarity == 0.0
        assert result.matches == ()

    def test_with_matches(self) -> None:
        entry = _make_entry()
        match = ModelChainMatch(
            chain_entry=entry,
            similarity_score=0.9,
            distance=0.1,
        )
        result = ModelChainRetrievalResult(
            correlation_id=uuid4(),
            matches=(match,),
            best_match_similarity=0.9,
            is_hit=True,
        )
        assert result.is_hit
        assert len(result.matches) == 1


@pytest.mark.unit
class TestModelChainReplay:
    def test_replay_input_frozen(self) -> None:
        inp = ModelChainReplayInput(
            correlation_id=uuid4(),
            cached_chain=_make_entry(),
            new_prompt_text="new prompt",
        )
        with pytest.raises(Exception):
            inp.new_prompt_text = "changed"  # type: ignore[misc]

    def test_replay_result(self) -> None:
        result = ModelChainReplayResult(
            correlation_id=uuid4(),
            adapted_steps=(_make_step(),),
            adaptation_summary="No changes",
            confidence=0.95,
        )
        assert result.confidence == 0.95


@pytest.mark.unit
class TestModelChainStore:
    def test_store_request_needs_embedding(self) -> None:
        with pytest.raises(Exception):
            ModelChainStoreRequest(
                correlation_id=uuid4(),
                chain_entry=_make_entry(),
                prompt_embedding=[],
            )

    def test_store_result(self) -> None:
        result = ModelChainStoreResult(
            correlation_id=uuid4(),
            chain_id=uuid4(),
            success=True,
        )
        assert result.success


@pytest.mark.unit
class TestEnumChainVerifyState:
    def test_all_states(self) -> None:
        expected = {
            "pending",
            "retrieving",
            "replaying",
            "exploring",
            "verifying",
            "complete",
            "fallback",
            "failed",
        }
        actual = {s.value for s in EnumChainVerifyState}
        assert actual == expected
