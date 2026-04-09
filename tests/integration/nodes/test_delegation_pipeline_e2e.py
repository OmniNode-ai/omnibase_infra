# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""End-to-end integration test for the delegation pipeline [OMN-7040].

Validates the full delegation flow through all three nodes:
  routing reducer -> orchestrator -> quality gate reducer

Two scenarios:
  1. Happy path: request -> route -> infer (good output) -> gate pass
     -> delegation-completed + baseline intent + backward-compat event
  2. Degraded path: request -> route -> infer (refusal) -> gate fail
     -> delegation-failed + backward-compat event, NO baseline intent

No real Kafka or LLM. The routing reducer reads env vars (monkeypatched),
the quality gate is pure compute, and LLM inference is simulated by
passing a ModelInferenceResponseData directly.

Related:
    - OMN-7040: Node-based delegation pipeline
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_delegation_orchestrator.enums import (
    EnumDelegationState,
)
from omnibase_infra.nodes.node_delegation_orchestrator.handlers.handler_delegation_workflow import (
    HandlerDelegationWorkflow,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_baseline_intent import (
    ModelBaselineIntent,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_event import (
    ModelDelegationEvent,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_request import (
    ModelDelegationRequest,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_inference_intent import (
    ModelInferenceIntent,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_inference_response_data import (
    ModelInferenceResponseData,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_quality_gate_intent import (
    ModelQualityGateIntent,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_routing_intent import (
    ModelRoutingIntent,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_task_delegated_event import (
    ModelTaskDelegatedEvent,
)
from omnibase_infra.nodes.node_delegation_quality_gate_reducer.handlers.handler_quality_gate import (
    delta as quality_gate_delta,
)
from omnibase_infra.nodes.node_delegation_routing_reducer.handlers.handler_delegation_routing import (
    delta as routing_delta,
)


@pytest.fixture(autouse=True)
def _set_llm_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide LLM endpoint env vars required by the routing reducer."""
    import omnibase_infra.nodes.node_delegation_routing_reducer.handlers.handler_delegation_routing as _h

    _h._config = None
    monkeypatch.setenv("LLM_CODER_URL", "http://192.168.86.201:8000")
    monkeypatch.setenv("LLM_CODER_FAST_URL", "http://192.168.86.201:8001")
    monkeypatch.setenv("LLM_DEEPSEEK_R1_URL", "http://192.168.86.200:8101")
    yield
    _h._config = None


def _simulate_inference(
    intent: ModelInferenceIntent,
    content: str,
    latency_ms: int = 800,
    prompt_tokens: int = 120,
    completion_tokens: int = 60,
) -> ModelInferenceResponseData:
    """Simulate an LLM inference response for the given intent."""
    return ModelInferenceResponseData(
        correlation_id=intent.correlation_id,
        content=content,
        model_used=intent.model,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


@pytest.mark.integration
class TestDelegationPipelineHappyPath:
    """Full pipeline: request -> route -> infer -> gate pass -> completed."""

    def test_happy_path_produces_completed_baseline_and_compat(self) -> None:
        cid = uuid4()
        request = ModelDelegationRequest(
            prompt="Write unit tests for verify_registration.py",
            task_type="test",
            correlation_id=cid,
            emitted_at=datetime.now(UTC),
        )

        # Step 1: Routing reducer decides endpoint
        routing_decision = routing_delta(request)
        assert routing_decision.correlation_id == cid
        assert routing_decision.endpoint_url in (
            "http://192.168.86.201:8000",
            "http://192.168.86.201:8001",
        )
        assert routing_decision.cost_tier == "low"

        # Step 2: Orchestrator processes request + routing
        orchestrator = HandlerDelegationWorkflow()
        routing_intents = orchestrator.handle_delegation_request(request)
        assert len(routing_intents) == 1
        assert isinstance(routing_intents[0], ModelRoutingIntent)

        inference_intents = orchestrator.handle_routing_decision(routing_decision)
        assert len(inference_intents) == 1
        assert isinstance(inference_intents[0], ModelInferenceIntent)
        assert inference_intents[0].temperature == 0.3

        # Step 3: Simulate LLM inference with valid test output
        good_output = (
            "import pytest\n\n"
            "@pytest.mark.unit\n"
            "def test_verify_registration_happy_path():\n"
            '    """Test that registration verification returns True."""\n'
            "    result = verify_registration('node-abc')\n"
            "    assert result is True\n\n"
            "def test_verify_registration_missing_node():\n"
            "    with pytest.raises(ValueError):\n"
            "        verify_registration('')\n"
        )
        inference_response = _simulate_inference(
            inference_intents[0],
            content=good_output,
            prompt_tokens=120,
            completion_tokens=80,
        )

        gate_intents = orchestrator.handle_inference_response(inference_response)
        assert len(gate_intents) == 1
        assert isinstance(gate_intents[0], ModelQualityGateIntent)

        # Step 4: Quality gate evaluates LLM output
        gate_result = quality_gate_delta(gate_intents[0].payload)
        assert gate_result.passed is True
        assert gate_result.quality_score >= 0.6

        # Step 5: Orchestrator processes gate result
        final_events = orchestrator.handle_gate_result(gate_result)

        assert len(final_events) == 3
        assert orchestrator.workflows[cid].state == EnumDelegationState.COMPLETED

        # Event 1: delegation-completed
        assert isinstance(final_events[0], ModelDelegationEvent)
        assert (
            final_events[0].topic == "onex.evt.omnibase-infra.delegation-completed.v1"
        )
        result = final_events[0].payload
        assert result.correlation_id == cid
        assert result.quality_passed is True
        assert result.task_type == "test"
        assert result.fallback_to_claude is False
        assert result.prompt_tokens == 120
        assert result.completion_tokens == 80

        # Event 2: baseline intent for savings pipeline
        assert isinstance(final_events[1], ModelBaselineIntent)
        assert final_events[1].correlation_id == cid
        assert final_events[1].baseline_cost_usd > 0
        assert final_events[1].candidate_cost_usd == 0.0

        # Event 3: backward-compat task-delegated event for omnidash
        assert isinstance(final_events[2], ModelTaskDelegatedEvent)
        assert final_events[2].correlation_id == cid
        assert final_events[2].quality_gate_passed is True
        assert final_events[2].delegated_by == "delegation-pipeline"
        assert final_events[2].cost_savings_usd > 0

    def test_happy_path_document_task_routes_to_deepseek(self) -> None:
        """Verify document tasks route to DeepSeek and use temperature 0.5."""
        cid = uuid4()
        request = ModelDelegationRequest(
            prompt="Write docstrings for the ConfigPrefetcher class",
            task_type="document",
            correlation_id=cid,
            emitted_at=datetime.now(UTC),
        )

        routing_decision = routing_delta(request)
        # document tasks route to deepseek-r1-14b (LLM_CODER_FAST_URL) in the local tier
        assert routing_decision.selected_model == "deepseek-r1-14b"
        assert routing_decision.endpoint_url == "http://192.168.86.201:8001"

        orchestrator = HandlerDelegationWorkflow()
        orchestrator.handle_delegation_request(request)
        inference_intents = orchestrator.handle_routing_decision(routing_decision)
        assert inference_intents[0].temperature == 0.5


@pytest.mark.integration
class TestDelegationPipelineDegradedPath:
    """Pipeline with gate failure: no baseline intent, failure event emitted."""

    def test_gate_failure_produces_failed_and_compat_no_baseline(self) -> None:
        cid = uuid4()
        request = ModelDelegationRequest(
            prompt="Write unit tests for auth_handler.py",
            task_type="test",
            correlation_id=cid,
            emitted_at=datetime.now(UTC),
        )

        routing_decision = routing_delta(request)
        orchestrator = HandlerDelegationWorkflow()
        orchestrator.handle_delegation_request(request)
        inference_intents = orchestrator.handle_routing_decision(routing_decision)

        # Simulate LLM returning a refusal
        refusal_output = (
            "I'm sorry, I cannot generate unit tests for this code as it "
            "appears to contain sensitive authentication logic."
        )
        inference_response = _simulate_inference(
            inference_intents[0],
            content=refusal_output,
            prompt_tokens=100,
            completion_tokens=30,
        )

        gate_intents = orchestrator.handle_inference_response(inference_response)
        gate_result = quality_gate_delta(gate_intents[0].payload)

        assert gate_result.passed is False
        assert any("REFUSAL" in r for r in gate_result.failure_reasons)
        assert gate_result.fallback_recommended is True

        final_events = orchestrator.handle_gate_result(gate_result)

        # 2 events: delegation-failed + backward-compat (NO baseline)
        assert len(final_events) == 2
        assert orchestrator.workflows[cid].state == EnumDelegationState.FAILED

        assert isinstance(final_events[0], ModelDelegationEvent)
        assert final_events[0].topic == "onex.evt.omnibase-infra.delegation-failed.v1"
        result = final_events[0].payload
        assert result.quality_passed is False
        assert result.fallback_to_claude is True
        assert "REFUSAL" in result.failure_reason

        assert isinstance(final_events[1], ModelTaskDelegatedEvent)
        assert final_events[1].quality_gate_passed is False

        baseline_events = [
            e for e in final_events if isinstance(e, ModelBaselineIntent)
        ]
        assert len(baseline_events) == 0

    def test_short_output_without_markers_fails_gate(self) -> None:
        """Short output + missing markers -> score below threshold -> fail."""
        cid = uuid4()
        request = ModelDelegationRequest(
            prompt="Write tests for parser.py",
            task_type="test",
            correlation_id=cid,
            emitted_at=datetime.now(UTC),
        )

        routing_decision = routing_delta(request)
        orchestrator = HandlerDelegationWorkflow()
        orchestrator.handle_delegation_request(request)
        inference_intents = orchestrator.handle_routing_decision(routing_decision)

        short_no_marker_output = "Not sure what to do here."
        inference_response = _simulate_inference(
            inference_intents[0],
            content=short_no_marker_output,
        )

        gate_intents = orchestrator.handle_inference_response(inference_response)
        gate_result = quality_gate_delta(gate_intents[0].payload)

        assert gate_result.passed is False
        assert gate_result.quality_score < 0.6
        assert any("WEAK_OUTPUT" in r for r in gate_result.failure_reasons)
        assert any("TASK_MISMATCH" in r for r in gate_result.failure_reasons)
