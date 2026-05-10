# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for compliance-loop wiring in HandlerDelegationWorkflow — OMN-10794.

These tests cover the new orchestrator behavior: when a delegation request
opts into the compliance loop via ``output_schema_key``, the workflow
validates each inference attempt, emits repair prompts on validation failure,
and accumulates ``tokens_to_compliance`` + ``compliance_attempts`` onto the
terminal result and event.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_DNS, UUID, uuid4, uuid5

import pytest
from omnimarket.nodes.node_budget_policy_compute.models.model_budget_limits import (
    ModelBudgetLimits,
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

pytestmark = pytest.mark.unit


_GENEROUS_LIMITS = ModelBudgetLimits(
    max_tokens=10_000, max_cost_usd=10.0, max_time_s=600.0
)
_TIGHT_LIMITS = ModelBudgetLimits(max_tokens=300, max_cost_usd=10.0, max_time_s=600.0)


def _valid_delegation_payload() -> str:
    payload: dict[str, Any] = {
        "correlation_id": "00000000-0000-0000-0000-000000000000",
        "task_type": "test",
        "model_used": "qwen3-coder",
        "endpoint_url": "http://192.168.86.201:8000",
        "content": "def test_x(): pass",
        "quality_passed": True,
        "quality_score": 0.95,
        "latency_ms": 100,
        "prompt_tokens": 50,
        "completion_tokens": 20,
        "total_tokens": 70,
        "fallback_to_claude": False,
        "failure_reason": "",
        "tokens_to_compliance": 70,
        "compliance_attempts": 1,
    }
    return json.dumps(payload)


def _make_request(
    *,
    correlation_id: UUID | None = None,
    output_schema_key: str | None = None,
    compliance_budget: ModelBudgetLimits | None = None,
) -> ModelDelegationRequest:
    return ModelDelegationRequest(
        prompt="Generate a unit test for foo()",
        task_type="test",
        correlation_id=correlation_id or uuid4(),
        emitted_at=datetime.now(UTC),
        output_schema_key=output_schema_key,
        compliance_budget=compliance_budget,
    )


def _make_routing(cid: UUID) -> ModelRoutingDecision:
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


def _make_inference_response(
    cid: UUID, content: str, *, total_tokens: int = 100
) -> ModelInferenceResponseData:
    # Split tokens into prompt/completion arbitrarily; the loop only consumes total.
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


def _drive_to_routed(
    workflow: HandlerDelegationWorkflow,
    request: ModelDelegationRequest,
) -> tuple[UUID, list[ModelInferenceIntent]]:
    workflow.handle_delegation_request(request)
    intents = workflow.handle_routing_decision(_make_routing(request.correlation_id))
    return request.correlation_id, intents  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Legacy path — output_schema_key unset, behavior unchanged
# ---------------------------------------------------------------------------


class TestLegacyPathUnchanged:
    """Without output_schema_key, the workflow takes the single-attempt path."""

    def test_legacy_request_skips_compliance_loop(self) -> None:
        wf = HandlerDelegationWorkflow()
        request = _make_request()  # output_schema_key=None
        cid, _ = _drive_to_routed(wf, request)

        out = wf.handle_inference_response(
            _make_inference_response(cid, "any content", total_tokens=120)
        )
        assert len(out) == 1
        assert isinstance(out[0], ModelQualityGateIntent)
        assert wf.workflows[cid].state == EnumDelegationState.INFERENCE_COMPLETED

    def test_legacy_terminal_event_carries_default_compliance_counters(self) -> None:
        wf = HandlerDelegationWorkflow()
        request = _make_request()
        cid, _ = _drive_to_routed(wf, request)

        wf.handle_inference_response(
            _make_inference_response(cid, "x", total_tokens=120)
        )
        events = wf.handle_gate_result(
            ModelQualityGateResult(
                correlation_id=cid,
                passed=True,
                quality_score=0.9,
                failure_reasons=[],
                fallback_recommended=False,
            )
        )
        compat = next(e for e in events if isinstance(e, ModelTaskDelegatedEvent))
        delegation = next(e for e in events if isinstance(e, ModelDelegationEvent))
        # Legacy path: compliance_attempts=1, tokens_to_compliance=total of single attempt.
        assert compat.compliance_attempts == 1
        assert compat.tokens_to_compliance == 120
        from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_result import (
            ModelDelegationResult,
        )

        assert isinstance(delegation.payload, ModelDelegationResult)
        assert delegation.payload.compliance_attempts == 1
        assert delegation.payload.tokens_to_compliance == 120


# ---------------------------------------------------------------------------
# Compliance-loop path — output_schema_key set, retry on validation failure
# ---------------------------------------------------------------------------


class TestComplianceLoopFirstTrySuccess:
    """When the first inference response validates against the schema, the
    workflow accepts immediately and forwards to the quality gate."""

    def test_first_try_compliant_emits_quality_gate_intent(self) -> None:
        wf = HandlerDelegationWorkflow()
        request = _make_request(
            output_schema_key="delegation_result",
            compliance_budget=_GENEROUS_LIMITS,
        )
        cid, _ = _drive_to_routed(wf, request)

        out = wf.handle_inference_response(
            _make_inference_response(cid, _valid_delegation_payload(), total_tokens=200)
        )
        assert len(out) == 1
        assert isinstance(out[0], ModelQualityGateIntent)
        assert wf.workflows[cid].state == EnumDelegationState.INFERENCE_COMPLETED
        assert wf.workflows[cid].compliance_attempts == 1
        assert wf.workflows[cid].accumulated_tokens == 200


class TestComplianceLoopRetry:
    """Validation failure with budget CONTINUE emits a repair-prompt
    ModelInferenceIntent and stays in ROUTED (self-loop)."""

    def test_first_attempt_failure_emits_repair_intent(self) -> None:
        wf = HandlerDelegationWorkflow()
        request = _make_request(
            output_schema_key="delegation_result",
            compliance_budget=_GENEROUS_LIMITS,
        )
        cid, initial = _drive_to_routed(wf, request)
        assert len(initial) == 1

        # Malformed JSON → schema repair triggers
        out = wf.handle_inference_response(
            _make_inference_response(cid, "not valid json {{{", total_tokens=100)
        )
        assert len(out) == 1
        assert isinstance(out[0], ModelInferenceIntent)
        # Repair prompt carries the schema-repair instructions
        assert "schema" in out[0].prompt.lower()
        assert wf.workflows[cid].state == EnumDelegationState.ROUTED  # self-loop
        assert (
            wf.workflows[cid].compliance_attempts == 2
        )  # incremented for next attempt
        assert wf.workflows[cid].accumulated_tokens == 100

    def test_two_attempt_success_accumulates_tokens(self) -> None:
        wf = HandlerDelegationWorkflow()
        request = _make_request(
            output_schema_key="delegation_result",
            compliance_budget=_GENEROUS_LIMITS,
        )
        cid, _ = _drive_to_routed(wf, request)

        # Attempt 1: fail (100 tokens) → repair issued
        attempt1 = wf.handle_inference_response(
            _make_inference_response(cid, "garbage", total_tokens=100)
        )
        assert isinstance(attempt1[0], ModelInferenceIntent)
        assert wf.workflows[cid].state == EnumDelegationState.ROUTED

        # Attempt 2: pass (150 tokens) → accept, forward to gate, total=250
        attempt2 = wf.handle_inference_response(
            _make_inference_response(cid, _valid_delegation_payload(), total_tokens=150)
        )
        assert len(attempt2) == 1
        assert isinstance(attempt2[0], ModelQualityGateIntent)
        assert wf.workflows[cid].state == EnumDelegationState.INFERENCE_COMPLETED
        assert wf.workflows[cid].compliance_attempts == 2
        assert wf.workflows[cid].accumulated_tokens == 250

    def test_two_attempt_terminal_event_carries_running_total(self) -> None:
        wf = HandlerDelegationWorkflow()
        request = _make_request(
            output_schema_key="delegation_result",
            compliance_budget=_GENEROUS_LIMITS,
        )
        cid, _ = _drive_to_routed(wf, request)

        wf.handle_inference_response(
            _make_inference_response(cid, "garbage", total_tokens=100)
        )
        wf.handle_inference_response(
            _make_inference_response(cid, _valid_delegation_payload(), total_tokens=150)
        )
        events = wf.handle_gate_result(
            ModelQualityGateResult(
                correlation_id=cid,
                passed=True,
                quality_score=0.9,
                failure_reasons=[],
                fallback_recommended=False,
            )
        )
        compat = next(e for e in events if isinstance(e, ModelTaskDelegatedEvent))
        assert compat.compliance_attempts == 2
        assert compat.tokens_to_compliance == 250


class TestComplianceLoopBudgetAbort:
    """Budget ABORT terminates the loop without further inference attempts;
    the orchestrator forwards the (non-compliant) attempt to the quality
    gate so the failure reason surfaces in the terminal event."""

    def test_budget_abort_forwards_to_gate(self) -> None:
        wf = HandlerDelegationWorkflow()
        request = _make_request(
            output_schema_key="delegation_result",
            compliance_budget=_TIGHT_LIMITS,  # 300-token ceiling
        )
        cid, _ = _drive_to_routed(wf, request)

        # First attempt: 400 tokens. Ratio 400/300=1.33 > 1.0 → ABORT.
        out = wf.handle_inference_response(
            _make_inference_response(cid, "garbage", total_tokens=400)
        )
        assert len(out) == 1
        # ABORT path forwards to gate, NOT another inference intent.
        assert isinstance(out[0], ModelQualityGateIntent)
        assert wf.workflows[cid].state == EnumDelegationState.INFERENCE_COMPLETED
        # Attempt counter not incremented (no retry happened); tokens recorded.
        assert wf.workflows[cid].compliance_attempts == 1
        assert wf.workflows[cid].accumulated_tokens == 400

    def test_budget_abort_terminal_event_carries_actual_attempts(self) -> None:
        wf = HandlerDelegationWorkflow()
        request = _make_request(
            output_schema_key="delegation_result",
            compliance_budget=_TIGHT_LIMITS,
        )
        cid, _ = _drive_to_routed(wf, request)

        wf.handle_inference_response(
            _make_inference_response(cid, "garbage", total_tokens=400)
        )
        events = wf.handle_gate_result(
            ModelQualityGateResult(
                correlation_id=cid,
                passed=False,
                quality_score=0.1,
                failure_reasons=["malformed json"],
                fallback_recommended=True,
            )
        )
        compat = next(e for e in events if isinstance(e, ModelTaskDelegatedEvent))
        assert compat.compliance_attempts == 1
        assert compat.tokens_to_compliance == 400
        assert compat.quality_gate_passed is False


class TestFsmSelfLoopAllowed:
    """The ROUTED -> ROUTED self-loop must be a legal FSM transition (OMN-10794)."""

    def test_self_loop_does_not_raise(self) -> None:
        wf = HandlerDelegationWorkflow()
        request = _make_request(
            output_schema_key="delegation_result",
            compliance_budget=_GENEROUS_LIMITS,
        )
        cid, _ = _drive_to_routed(wf, request)

        # First failed attempt → self-loop. Must not raise.
        out = wf.handle_inference_response(
            _make_inference_response(cid, "garbage", total_tokens=50)
        )
        assert isinstance(out[0], ModelInferenceIntent)
        assert wf.workflows[cid].state == EnumDelegationState.ROUTED

        # Second failed attempt → another self-loop, also legal.
        out2 = wf.handle_inference_response(
            _make_inference_response(cid, "still garbage", total_tokens=60)
        )
        assert isinstance(out2[0], ModelInferenceIntent)
        assert wf.workflows[cid].state == EnumDelegationState.ROUTED
        assert wf.workflows[cid].compliance_attempts == 3
