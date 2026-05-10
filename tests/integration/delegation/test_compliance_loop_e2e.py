# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Proof of Life — end-to-end delegation through the compliance loop (OMN-10796).

This test drives the **full** delegation pipeline with
``output_schema_key="delegation_result"``, simulates an LLM that returns
malformed JSON on the first attempt and valid JSON on the second, and asserts
that the terminal artifacts (``ModelDelegationResult`` + the backward-compat
``ModelTaskDelegatedEvent`` for omnidash) carry ``tokens_to_compliance`` and
``compliance_attempts`` populated from the workflow state.

OMN-10794 already covers the wiring up to ``INFERENCE_COMPLETED``; this test
goes further — through the quality-gate reducer to ``COMPLETED`` — and proves
the compliance counters survive the gate handoff and land on the public
event payloads that downstream projections (omnidash widget, golden chain)
consume.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import NAMESPACE_DNS, UUID, uuid4, uuid5

import pytest
from omnimarket.nodes.node_budget_policy_compute.models.model_budget_limits import (
    ModelBudgetLimits,
)

from omnibase_infra.event_bus.topic_constants import (
    TOPIC_DELEGATION_COMPLETED,
    TOPIC_DELEGATION_TASK_DELEGATED,
)
from omnibase_infra.nodes.node_delegation_orchestrator.enums import EnumDelegationState
from omnibase_infra.nodes.node_delegation_orchestrator.handlers.handler_delegation_workflow import (
    HandlerDelegationWorkflow,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_event import (
    ModelDelegationEvent,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_request import (
    ModelDelegationRequest,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_result import (
    ModelDelegationResult,
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
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_task_delegated_event import (
    ModelTaskDelegatedEvent,
)
from omnibase_infra.nodes.node_delegation_quality_gate_reducer.models.model_quality_gate_result import (
    ModelQualityGateResult,
)
from omnibase_infra.nodes.node_delegation_routing_reducer.models.model_routing_decision import (
    ModelRoutingDecision,
)

pytestmark = pytest.mark.integration


_GENEROUS_LIMITS = ModelBudgetLimits(
    max_tokens=10_000, max_cost_usd=10.0, max_time_s=600.0
)

_ATTEMPT_1_TOKENS = 180
_ATTEMPT_2_TOKENS = 220
_EXPECTED_TOTAL_TOKENS = _ATTEMPT_1_TOKENS + _ATTEMPT_2_TOKENS  # 400


def _routing(cid: UUID) -> ModelRoutingDecision:
    return ModelRoutingDecision(
        correlation_id=cid,
        task_type="test",
        selected_model="qwen3-coder-30b",
        selected_backend_id=uuid5(
            NAMESPACE_DNS, "omninode.ai/backends/qwen3-coder-30b"
        ),
        endpoint_url="http://192.168.86.201:8000",
        cost_tier="low",
        max_context_tokens=65536,
        system_prompt="You are a test generation assistant.",
        rationale="qwen3-coder-30b for task=test",
    )


def _response(
    cid: UUID, content: str, *, total_tokens: int
) -> ModelInferenceResponseData:
    prompt_tokens = total_tokens // 2
    completion_tokens = total_tokens - prompt_tokens
    return ModelInferenceResponseData(
        correlation_id=cid,
        content=content,
        model_used="qwen3-coder-30b",
        latency_ms=100,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _malformed_payload() -> str:
    """Missing every required field on ModelDelegationResult except task_type."""
    return json.dumps({"task_type": "summarize"})


def _valid_payload(cid: UUID) -> str:
    """A schema-compliant ModelDelegationResult payload."""
    return json.dumps(
        {
            "correlation_id": str(cid),
            "task_type": "test",
            "model_used": "qwen3-coder-30b",
            "endpoint_url": "http://192.168.86.201:8000",
            "content": "def test_compliance_loop_e2e(): assert True",
            "quality_passed": True,
            "quality_score": 0.95,
            "latency_ms": 100,
            "prompt_tokens": 110,
            "completion_tokens": 110,
            "total_tokens": _ATTEMPT_2_TOKENS,
            "fallback_to_claude": False,
            "failure_reason": "",
            "tokens_to_compliance": _ATTEMPT_2_TOKENS,
            "compliance_attempts": 1,
        }
    )


class TestComplianceLoopProofOfLife:
    """End-to-end: delegation request → 2-attempt compliance loop → terminal event."""

    def test_full_pipeline_publishes_compliance_counters_on_terminal_event(
        self,
    ) -> None:
        """Drive the workflow from request through to TOPIC_DELEGATION_COMPLETED.

        This is the OMN-10796 Proof of Life. It exercises:

        - request intake + routing decision (state: SUBMITTED → ROUTED)
        - attempt 1: malformed JSON triggers HandlerComplianceLoop, which
          builds a repair prompt and emits a fresh ModelInferenceIntent;
          workflow stays in ROUTED for the self-loop; compliance_attempts=2,
          accumulated_tokens=180.
        - attempt 2: schema-compliant JSON satisfies the compliance loop;
          workflow transitions ROUTED → INFERENCE_COMPLETED and emits a
          ModelQualityGateIntent; accumulated_tokens=400.
        - quality-gate reducer returns a passing verdict.
        - workflow transitions INFERENCE_COMPLETED → GATE_EVALUATED →
          COMPLETED and emits both:
            * a ModelDelegationEvent (topic=TOPIC_DELEGATION_COMPLETED) whose
              payload is a ModelDelegationResult.
            * a ModelTaskDelegatedEvent (topic=TOPIC_DELEGATION_TASK_DELEGATED)
              that the omnidash projection consumes.
        - both terminal artifacts carry tokens_to_compliance=400 and
          compliance_attempts=2 — proving the workflow state survives the
          gate handoff and reaches the projection-facing surface.
        """
        wf = HandlerDelegationWorkflow()
        request = ModelDelegationRequest(
            prompt="Generate a unit test for foo()",
            task_type="test",
            correlation_id=uuid4(),
            emitted_at=datetime.now(UTC),
            output_schema_key="delegation_result",
            compliance_budget=_GENEROUS_LIMITS,
        )
        cid = request.correlation_id

        wf.handle_delegation_request(request)
        wf.handle_routing_decision(_routing(cid))

        # ─── Attempt 1: malformed JSON → repair intent, stay ROUTED ──────
        attempt_1_out = wf.handle_inference_response(
            _response(cid, _malformed_payload(), total_tokens=_ATTEMPT_1_TOKENS)
        )
        assert len(attempt_1_out) == 1
        assert isinstance(attempt_1_out[0], ModelInferenceIntent)
        assert wf.workflows[cid].state == EnumDelegationState.ROUTED
        assert (
            wf.workflows[cid].compliance_attempts == 2
        )  # incremented for next attempt
        assert wf.workflows[cid].accumulated_tokens == _ATTEMPT_1_TOKENS

        # ─── Attempt 2: valid JSON → quality gate intent, INFERENCE_COMPLETED ──
        attempt_2_out = wf.handle_inference_response(
            _response(cid, _valid_payload(cid), total_tokens=_ATTEMPT_2_TOKENS)
        )
        assert len(attempt_2_out) == 1
        assert isinstance(attempt_2_out[0], ModelQualityGateIntent)
        assert wf.workflows[cid].state == EnumDelegationState.INFERENCE_COMPLETED
        assert wf.workflows[cid].compliance_attempts == 2
        assert wf.workflows[cid].accumulated_tokens == _EXPECTED_TOTAL_TOKENS

        # ─── Quality gate verdict (PASS) → terminal events ───────────────
        gate_result = ModelQualityGateResult(
            correlation_id=cid,
            passed=True,
            fail_category="pass",
            quality_score=0.95,
            failure_reasons=(),
            fallback_recommended=False,
        )
        terminal_events = wf.handle_gate_result(gate_result)
        assert wf.workflows[cid].state == EnumDelegationState.COMPLETED

        # Locate the terminal payloads. The handler returns at minimum:
        #   - ModelDelegationEvent(topic=TOPIC_DELEGATION_COMPLETED, payload=ModelDelegationResult)
        #   - ModelTaskDelegatedEvent(topic=TOPIC_DELEGATION_TASK_DELEGATED)
        delegation_events = [
            e
            for e in terminal_events
            if isinstance(e, ModelDelegationEvent)
            and e.topic == TOPIC_DELEGATION_COMPLETED
        ]
        task_delegated_events = [
            e
            for e in terminal_events
            if isinstance(e, ModelTaskDelegatedEvent)
            and e.topic == TOPIC_DELEGATION_TASK_DELEGATED
        ]

        assert len(delegation_events) == 1, (
            f"expected 1 ModelDelegationEvent on TOPIC_DELEGATION_COMPLETED, "
            f"got {[type(e).__name__ for e in terminal_events]}"
        )
        assert len(task_delegated_events) == 1, (
            f"expected 1 ModelTaskDelegatedEvent on TOPIC_DELEGATION_TASK_DELEGATED, "
            f"got {[type(e).__name__ for e in terminal_events]}"
        )

        # ─── Assert: ModelDelegationResult carries the compliance counters ──
        delegation_result = delegation_events[0].payload
        assert isinstance(delegation_result, ModelDelegationResult)
        assert delegation_result.tokens_to_compliance == _EXPECTED_TOTAL_TOKENS, (
            f"tokens_to_compliance must equal sum of both attempts "
            f"({_ATTEMPT_1_TOKENS} + {_ATTEMPT_2_TOKENS} = {_EXPECTED_TOTAL_TOKENS}), "
            f"got {delegation_result.tokens_to_compliance}"
        )
        assert delegation_result.compliance_attempts == 2

        # ─── Assert: ModelTaskDelegatedEvent fields match ────────────────
        task_event = task_delegated_events[0]
        assert task_event.tokens_to_compliance == delegation_result.tokens_to_compliance
        assert task_event.compliance_attempts == delegation_result.compliance_attempts
        assert task_event.tokens_to_compliance == _EXPECTED_TOTAL_TOKENS
        assert task_event.compliance_attempts == 2

    def test_golden_chain_field_presence_on_task_delegated_event(self) -> None:
        """Golden-chain projection check: ModelTaskDelegatedEvent must expose
        ``tokens_to_compliance`` and ``compliance_attempts`` as concrete fields
        so the omnidash quality-gate projection (OMN-10795) can read them off
        the wire without optional-field gymnastics."""
        field_names = set(ModelTaskDelegatedEvent.model_fields.keys())
        assert "tokens_to_compliance" in field_names
        assert "compliance_attempts" in field_names

        # And the same for the typed result that downstream projections receive.
        result_fields = set(ModelDelegationResult.model_fields.keys())
        assert "tokens_to_compliance" in result_fields
        assert "compliance_attempts" in result_fields
