# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Integration tests for NodeRewardBinderEffect against dev Redpanda.

These tests publish actual events to Redpanda and verify receipt.
They require:
  - KAFKA_BOOTSTRAP_SERVERS set in environment
  - A running Redpanda/Kafka instance (dev cluster or local)

Tests are skipped gracefully if Kafka is not available.

Ticket: OMN-2927
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from omnibase_core.models.objective.model_score_vector import ModelScoreVector
from omnibase_infra.nodes.node_reward_binder_effect.handlers.handler_reward_binder import (
    _TOPIC_POLICY_STATE_UPDATED,
    _TOPIC_REWARD_ASSIGNED,
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

# ==============================================================================
# Skip conditions
# ==============================================================================

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_AVAILABLE = KAFKA_BOOTSTRAP_SERVERS is not None

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not KAFKA_AVAILABLE,
        reason="Kafka not available (KAFKA_BOOTSTRAP_SERVERS not set)",
    ),
]


# ==============================================================================
# Helpers
# ==============================================================================


def _make_evaluation_result() -> ModelEvaluationResult:
    """Build a test ModelEvaluationResult with canonical score vector."""
    run_id = uuid4()
    evidence = ModelEvidenceBundle(
        run_id=run_id,
        items=(
            ModelEvidenceItem(source="integration_test", content="evidence_a"),
            ModelEvidenceItem(source="integration_test", content="evidence_b"),
        ),
    )
    return ModelEvaluationResult(
        run_id=run_id,
        objective_id=uuid4(),
        score_vector=ModelScoreVector(
            correctness=0.9,
            safety=0.85,
            cost=0.8,
            latency=0.75,
            maintainability=0.7,
            human_time=0.95,
        ),
        evidence_bundle=evidence,
        policy_state_before={"policy_version": 1},
        policy_state_after={"policy_version": 2},
    )


def _make_objective_spec() -> ModelObjectiveSpec:
    """Build a test ModelObjectiveSpec."""
    return ModelObjectiveSpec(
        objective_id=uuid4(),
        name="integration-test-objective",
        target_types=("tool", "model"),
    )


class _FakeContainer:
    pass


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestRewardBinderKafkaIntegration:
    """Integration tests verifying events land in correct Kafka topics."""

    @pytest.mark.asyncio
    async def test_events_published_to_correct_topics(self) -> None:
        """All three event types reach correct Kafka topics in correct order.

        Uses EventBusKafka to publish and verifies output model is correct.
        """
        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
        from omnibase_infra.runtime.publisher_topic_scoped import PublisherTopicScoped

        bootstrap = os.environ["KAFKA_BOOTSTRAP_SERVERS"]

        result = _make_evaluation_result()
        spec = _make_objective_spec()
        corr_id = uuid4()

        bus = EventBusKafka(
            bootstrap_servers=bootstrap,
            environment="dev",
        )
        await bus.start()

        try:
            publisher = PublisherTopicScoped(
                event_bus=bus,
                allowed_topics={
                    _TOPIC_REWARD_ASSIGNED,
                    _TOPIC_POLICY_STATE_UPDATED,
                },
                environment="dev",
            )

            handler = HandlerRewardBinder(
                container=_FakeContainer(),  # type: ignore[arg-type]
                publisher=publisher.publish,
            )
            await handler.initialize({})

            envelope: dict[str, object] = {
                "correlation_id": corr_id,
                "evaluation_result": result,
                "objective_spec": spec,
            }
            handler_output = await handler.execute(envelope)
            output = handler_output.result

            assert isinstance(output, ModelRewardBinderOutput)
            assert output.success is True
            assert output.run_id == result.run_id

            expected_fp = _compute_objective_fingerprint(spec)
            assert output.objective_fingerprint == expected_fp

            # Verify event counts (one reward event per run)
            # run_evaluated_event_id removed in OMN-2929 (orphan topic retired)
            assert len(output.reward_assigned_event_ids) == 1
            assert output.policy_state_updated_event_id is not None

            # Verify two topics published (run-evaluated topic retired in OMN-2929)
            assert _TOPIC_REWARD_ASSIGNED in output.topics_published
            assert _TOPIC_POLICY_STATE_UPDATED in output.topics_published
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_publish_failure_propagates(self) -> None:
        """Kafka publish failure propagates -- not swallowed silently."""
        broken_publisher = AsyncMock(
            side_effect=ConnectionError("Kafka unavailable for test")
        )
        handler = HandlerRewardBinder(
            container=_FakeContainer(),  # type: ignore[arg-type]
            publisher=broken_publisher,
        )
        result = _make_evaluation_result()
        spec = _make_objective_spec()

        await handler.initialize({})
        with pytest.raises(ConnectionError, match="Kafka unavailable for test"):
            await handler.execute(
                {
                    "correlation_id": uuid4(),
                    "evaluation_result": result,
                    "objective_spec": spec,
                }
            )
