# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerRewardBinder.

Tests:
- objective_fingerprint is deterministic SHA-256 of ModelObjectiveSpec.model_dump_json()
- evidence_refs trace back to ModelEvidenceItem.item_id values in ModelEvidenceBundle
- Three events emitted in correct order: RunEvaluated -> RewardAssigned -> PolicyStateUpdated
- Event structure correct (topic names, field values)
- Kafka publish failure propagates (never swallowed silently)
- Missing inputs raise RuntimeHostError
- No publisher configured raises RuntimeHostError

Ticket: OMN-2552
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.nodes.node_reward_binder_effect.handlers.handler_reward_binder import (
    _TOPIC_POLICY_STATE_UPDATED,
    _TOPIC_REWARD_ASSIGNED,
    _TOPIC_RUN_EVALUATED,
    HandlerRewardBinder,
    _compute_objective_fingerprint,
)
from omnibase_infra.nodes.node_reward_binder_effect.models.model_evaluation_result import (
    ModelEvaluationResult,
)
from omnibase_infra.nodes.node_reward_binder_effect.models.model_evidence_bundle import (
    ModelEvidenceBundle,
)
from omnibase_infra.nodes.node_reward_binder_effect.models.model_evidence_item import (
    ModelEvidenceItem,
)
from omnibase_infra.nodes.node_reward_binder_effect.models.model_objective_spec import (
    ModelObjectiveSpec,
)
from omnibase_infra.nodes.node_reward_binder_effect.models.model_reward_binder_output import (
    ModelRewardBinderOutput,
)
from omnibase_infra.nodes.node_reward_binder_effect.models.model_score_vector import (
    ModelScoreVector,
)

pytestmark = pytest.mark.unit


# ==============================================================================
# Helpers / Fixtures
# ==============================================================================


def _make_objective_spec(name: str = "test-objective") -> ModelObjectiveSpec:
    """Build a minimal ModelObjectiveSpec for testing."""
    return ModelObjectiveSpec(
        objective_id=uuid4(),
        name=name,
        target_types=("tool", "model"),
    )


def _make_evidence_bundle(run_id: UUID) -> ModelEvidenceBundle:
    """Build a ModelEvidenceBundle with two items."""
    return ModelEvidenceBundle(
        run_id=run_id,
        items=(
            ModelEvidenceItem(source="session_log", content="evidence A"),
            ModelEvidenceItem(source="session_log", content="evidence B"),
        ),
    )


def _make_evaluation_result(
    objective_id: UUID | None = None,
    policy_before: dict[str, object] | None = None,
    policy_after: dict[str, object] | None = None,
    num_targets: int = 2,
) -> ModelEvaluationResult:
    """Build a minimal ModelEvaluationResult with N score vectors."""
    run_id = uuid4()
    evidence = _make_evidence_bundle(run_id)
    score_vectors = tuple(
        ModelScoreVector(
            target_id=uuid4(),
            target_type="tool" if i % 2 == 0 else "model",
            dimensions={"accuracy": 0.8 + i * 0.05},
            composite_score=0.75 + i * 0.05,
        )
        for i in range(num_targets)
    )
    return ModelEvaluationResult(
        run_id=run_id,
        objective_id=objective_id or uuid4(),
        score_vectors=score_vectors,
        evidence_bundle=evidence,
        policy_state_before=policy_before or {"version": 1},
        policy_state_after=policy_after or {"version": 2},
    )


class _FakeContainer:
    """Minimal container stub."""


# ==============================================================================
# _compute_objective_fingerprint
# ==============================================================================


class TestComputeObjectiveFingerprint:
    """Tests for the fingerprint helper function."""

    def test_returns_64_char_hex(self) -> None:
        """Fingerprint is exactly 64 hex chars (SHA-256)."""
        spec = _make_objective_spec()
        fp = _compute_objective_fingerprint(spec)
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_deterministic_for_same_spec(self) -> None:
        """Same spec always produces the same fingerprint."""
        spec = ModelObjectiveSpec(
            objective_id=UUID("12345678-1234-5678-1234-567812345678"),
            name="deterministic",
            target_types=("tool",),
        )
        fp1 = _compute_objective_fingerprint(spec)
        fp2 = _compute_objective_fingerprint(spec)
        assert fp1 == fp2

    def test_matches_manual_sha256(self) -> None:
        """Fingerprint matches manual SHA-256 of model_dump_json()."""
        spec = ModelObjectiveSpec(
            objective_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            name="manual-check",
            target_types=("agent",),
        )
        expected = hashlib.sha256(spec.model_dump_json().encode("utf-8")).hexdigest()
        assert _compute_objective_fingerprint(spec) == expected

    def test_different_specs_produce_different_fingerprints(self) -> None:
        """Two different specs produce different fingerprints."""
        spec_a = _make_objective_spec("alpha")
        spec_b = _make_objective_spec("beta")
        assert _compute_objective_fingerprint(spec_a) != _compute_objective_fingerprint(
            spec_b
        )


# ==============================================================================
# HandlerRewardBinder -- execute()
# ==============================================================================


class TestHandlerRewardBinderExecute:
    """Tests for HandlerRewardBinder.execute()."""

    @pytest.fixture
    def publisher(self) -> AsyncMock:
        """AsyncMock publisher that always returns True."""
        mock = AsyncMock(return_value=True)
        return mock

    @pytest.fixture
    def handler(self, publisher: AsyncMock) -> HandlerRewardBinder:
        """Configured handler with mock publisher."""
        return HandlerRewardBinder(
            container=_FakeContainer(),  # type: ignore[arg-type]
            publisher=publisher,
        )

    @pytest.fixture
    def result(self) -> ModelEvaluationResult:
        """Default ModelEvaluationResult with 2 score vectors."""
        return _make_evaluation_result(num_targets=2)

    @pytest.fixture
    def spec(self) -> ModelObjectiveSpec:
        """Default ModelObjectiveSpec."""
        return _make_objective_spec()

    @pytest.mark.asyncio
    async def test_returns_success_output(
        self,
        handler: HandlerRewardBinder,
        result: ModelEvaluationResult,
        spec: ModelObjectiveSpec,
    ) -> None:
        """execute() returns ModelHandlerOutput with success=True."""
        corr_id = uuid4()
        envelope: dict[str, object] = {
            "correlation_id": corr_id,
            "evaluation_result": result,
            "objective_spec": spec,
        }
        await handler.initialize({})
        handler_output = await handler.execute(envelope)

        output = handler_output.result
        assert isinstance(output, ModelRewardBinderOutput)
        assert output.success is True
        assert output.correlation_id == corr_id
        assert output.run_id == result.run_id

    @pytest.mark.asyncio
    async def test_objective_fingerprint_in_output(
        self,
        handler: HandlerRewardBinder,
        result: ModelEvaluationResult,
        spec: ModelObjectiveSpec,
    ) -> None:
        """Output objective_fingerprint matches SHA-256 of spec."""
        expected = _compute_objective_fingerprint(spec)
        envelope: dict[str, object] = {
            "correlation_id": uuid4(),
            "evaluation_result": result,
            "objective_spec": spec,
        }
        await handler.initialize({})
        handler_output = await handler.execute(envelope)

        output = handler_output.result
        assert isinstance(output, ModelRewardBinderOutput)
        assert output.objective_fingerprint == expected
        assert len(output.objective_fingerprint) == 64

    @pytest.mark.asyncio
    async def test_publisher_called_in_order(
        self,
        publisher: AsyncMock,
        handler: HandlerRewardBinder,
        result: ModelEvaluationResult,
        spec: ModelObjectiveSpec,
    ) -> None:
        """Publisher called in order: RunEvaluated -> RewardAssigned x N -> PolicyStateUpdated."""
        num_targets = len(result.score_vectors)
        envelope: dict[str, object] = {
            "correlation_id": uuid4(),
            "evaluation_result": result,
            "objective_spec": spec,
        }
        await handler.initialize({})
        await handler.execute(envelope)

        # Total calls: 1 RunEvaluated + N RewardAssigned + 1 PolicyStateUpdated
        assert publisher.call_count == 1 + num_targets + 1

        calls = publisher.call_args_list
        # First call: RunEvaluated
        assert calls[0].kwargs["topic"] == _TOPIC_RUN_EVALUATED
        assert calls[0].kwargs["event_type"] == "run.evaluated"
        # Middle calls: RewardAssigned
        for i in range(1, num_targets + 1):
            assert calls[i].kwargs["topic"] == _TOPIC_REWARD_ASSIGNED
            assert calls[i].kwargs["event_type"] == "reward.assigned"
        # Last call: PolicyStateUpdated
        assert calls[-1].kwargs["topic"] == _TOPIC_POLICY_STATE_UPDATED
        assert calls[-1].kwargs["event_type"] == "policy.state.updated"

    @pytest.mark.asyncio
    async def test_run_evaluated_contains_fingerprint(
        self,
        publisher: AsyncMock,
        handler: HandlerRewardBinder,
        result: ModelEvaluationResult,
        spec: ModelObjectiveSpec,
    ) -> None:
        """ModelRunEvaluatedEvent payload contains correct objective_fingerprint."""
        expected_fp = _compute_objective_fingerprint(spec)
        envelope: dict[str, object] = {
            "correlation_id": uuid4(),
            "evaluation_result": result,
            "objective_spec": spec,
        }
        await handler.initialize({})
        await handler.execute(envelope)

        # First publisher call is RunEvaluated
        run_eval_payload = publisher.call_args_list[0].kwargs["payload"]
        assert run_eval_payload["objective_fingerprint"] == expected_fp
        assert run_eval_payload["run_id"] == str(result.run_id)

    @pytest.mark.asyncio
    async def test_reward_assigned_evidence_refs_traceable(
        self,
        publisher: AsyncMock,
        handler: HandlerRewardBinder,
        result: ModelEvaluationResult,
        spec: ModelObjectiveSpec,
    ) -> None:
        """ModelRewardAssignedEvent evidence_refs trace to ModelEvidenceItem.item_id values."""
        expected_item_ids = {str(item.item_id) for item in result.evidence_bundle.items}
        envelope: dict[str, object] = {
            "correlation_id": uuid4(),
            "evaluation_result": result,
            "objective_spec": spec,
        }
        await handler.initialize({})
        await handler.execute(envelope)

        num_targets = len(result.score_vectors)
        # Calls 1..N are RewardAssigned
        for i in range(1, num_targets + 1):
            payload = publisher.call_args_list[i].kwargs["payload"]
            refs_in_payload = set(payload["evidence_refs"])
            # All refs must be valid item IDs
            assert refs_in_payload.issubset(expected_item_ids)
            # There must be at least one ref
            assert len(refs_in_payload) > 0

    @pytest.mark.asyncio
    async def test_policy_state_updated_includes_snapshots(
        self,
        publisher: AsyncMock,
        handler: HandlerRewardBinder,
        result: ModelEvaluationResult,
        spec: ModelObjectiveSpec,
    ) -> None:
        """ModelPolicyStateUpdatedEvent payload includes both old_state and new_state."""
        envelope: dict[str, object] = {
            "correlation_id": uuid4(),
            "evaluation_result": result,
            "objective_spec": spec,
        }
        await handler.initialize({})
        await handler.execute(envelope)

        policy_payload = publisher.call_args_list[-1].kwargs["payload"]
        assert "old_state" in policy_payload
        assert "new_state" in policy_payload
        assert policy_payload["old_state"] == result.policy_state_before
        assert policy_payload["new_state"] == result.policy_state_after

    @pytest.mark.asyncio
    async def test_output_reward_event_ids_match_count(
        self,
        handler: HandlerRewardBinder,
        result: ModelEvaluationResult,
        spec: ModelObjectiveSpec,
    ) -> None:
        """Output reward_assigned_event_ids count equals number of score vectors."""
        envelope: dict[str, object] = {
            "correlation_id": uuid4(),
            "evaluation_result": result,
            "objective_spec": spec,
        }
        await handler.initialize({})
        handler_output = await handler.execute(envelope)

        output = handler_output.result
        assert isinstance(output, ModelRewardBinderOutput)
        assert len(output.reward_assigned_event_ids) == len(result.score_vectors)

    @pytest.mark.asyncio
    async def test_output_topics_published(
        self,
        handler: HandlerRewardBinder,
        result: ModelEvaluationResult,
        spec: ModelObjectiveSpec,
    ) -> None:
        """Output topics_published contains all three topic names."""
        envelope: dict[str, object] = {
            "correlation_id": uuid4(),
            "evaluation_result": result,
            "objective_spec": spec,
        }
        await handler.initialize({})
        handler_output = await handler.execute(envelope)

        output = handler_output.result
        assert isinstance(output, ModelRewardBinderOutput)
        assert _TOPIC_RUN_EVALUATED in output.topics_published
        assert _TOPIC_REWARD_ASSIGNED in output.topics_published
        assert _TOPIC_POLICY_STATE_UPDATED in output.topics_published

    @pytest.mark.asyncio
    async def test_kafka_failure_propagates(
        self,
        handler: HandlerRewardBinder,
        result: ModelEvaluationResult,
        spec: ModelObjectiveSpec,
    ) -> None:
        """Kafka publish failure is never swallowed -- it propagates to the caller."""
        handler._publisher = AsyncMock(side_effect=ConnectionError("Kafka down"))

        envelope: dict[str, object] = {
            "correlation_id": uuid4(),
            "evaluation_result": result,
            "objective_spec": spec,
        }
        await handler.initialize({})
        with pytest.raises(ConnectionError, match="Kafka down"):
            await handler.execute(envelope)

    @pytest.mark.asyncio
    async def test_missing_evaluation_result_raises(
        self,
        handler: HandlerRewardBinder,
        spec: ModelObjectiveSpec,
    ) -> None:
        """Missing evaluation_result raises RuntimeHostError."""
        from omnibase_infra.errors import RuntimeHostError

        envelope: dict[str, object] = {
            "correlation_id": uuid4(),
            "objective_spec": spec,
        }
        await handler.initialize({})
        with pytest.raises(RuntimeHostError, match="evaluation_result"):
            await handler.execute(envelope)

    @pytest.mark.asyncio
    async def test_missing_objective_spec_raises(
        self,
        handler: HandlerRewardBinder,
        result: ModelEvaluationResult,
    ) -> None:
        """Missing objective_spec raises RuntimeHostError."""
        from omnibase_infra.errors import RuntimeHostError

        envelope: dict[str, object] = {
            "correlation_id": uuid4(),
            "evaluation_result": result,
        }
        await handler.initialize({})
        with pytest.raises(RuntimeHostError, match="objective_spec"):
            await handler.execute(envelope)

    @pytest.mark.asyncio
    async def test_no_publisher_raises(
        self,
        result: ModelEvaluationResult,
        spec: ModelObjectiveSpec,
    ) -> None:
        """Handler without publisher raises RuntimeHostError on execute()."""
        from omnibase_infra.errors import RuntimeHostError

        handler = HandlerRewardBinder(container=_FakeContainer(), publisher=None)  # type: ignore[arg-type]
        envelope: dict[str, object] = {
            "correlation_id": uuid4(),
            "evaluation_result": result,
            "objective_spec": spec,
        }
        await handler.initialize({})
        with pytest.raises(RuntimeHostError, match="publisher"):
            await handler.execute(envelope)

    @pytest.mark.asyncio
    async def test_single_target_produces_one_reward_event(
        self,
        publisher: AsyncMock,
        spec: ModelObjectiveSpec,
    ) -> None:
        """Single-target result produces exactly one ModelRewardAssignedEvent."""
        result = _make_evaluation_result(num_targets=1)
        handler = HandlerRewardBinder(
            container=_FakeContainer(),  # type: ignore[arg-type]
            publisher=publisher,
        )
        envelope: dict[str, object] = {
            "correlation_id": uuid4(),
            "evaluation_result": result,
            "objective_spec": spec,
        }
        await handler.initialize({})
        await handler.execute(envelope)

        # 1 RunEvaluated + 1 RewardAssigned + 1 PolicyStateUpdated = 3 calls
        assert publisher.call_count == 3
        assert publisher.call_args_list[1].kwargs["topic"] == _TOPIC_REWARD_ASSIGNED

    @pytest.mark.asyncio
    async def test_handler_properties(
        self,
        handler: HandlerRewardBinder,
    ) -> None:
        """Handler exposes correct type and category."""
        from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory

        assert handler.handler_type == EnumHandlerType.NODE_HANDLER
        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT

    @pytest.mark.asyncio
    async def test_initialize_and_shutdown(
        self,
        handler: HandlerRewardBinder,
    ) -> None:
        """initialize() and shutdown() complete without error."""
        await handler.initialize({})
        assert handler._initialized is True
        await handler.shutdown()
        assert handler._initialized is False
