# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration test for compliance-loop wiring in HandlerDelegationWorkflow — OMN-10794.

The OMN-10792 unit tests for HandlerComplianceLoop and the OMN-10794 unit tests
for HandlerDelegationWorkflow each cover their own surface, but neither exercises
the full path from a real ModelDelegationRequest through the workflow handler,
into the real omnimarket compute nodes (schema-registry, schema-repair,
budget-policy), and back out as repair intent or terminal event.

This test wires the orchestrator handler against unmocked omnimarket handlers
and asserts the cross-package contract holds end-to-end:

* request -> routing decision -> compliance-eligible response triggers
  HandlerComplianceLoop.evaluate against the registered `delegation_result`
  schema and emits a fresh ModelInferenceIntent carrying the repair prompt
  (workflow remains in ROUTED for the self-loop).
* the second-attempt compliant response transitions ROUTED -> INFERENCE_COMPLETED
  with accumulated tokens equal to attempt 1 + attempt 2.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import NAMESPACE_DNS, uuid4, uuid5

import pytest
from omnimarket.nodes.node_budget_policy_compute.models.model_budget_limits import (
    ModelBudgetLimits,
)
from omnimarket.nodes.node_output_schema_registry_compute.handlers.handler_output_schema_registry import (
    known_schema_keys,
)

from omnibase_infra.nodes.node_delegation_orchestrator.enums import EnumDelegationState
from omnibase_infra.nodes.node_delegation_orchestrator.handlers.handler_delegation_workflow import (
    HandlerDelegationWorkflow,
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
from omnibase_infra.nodes.node_delegation_routing_reducer.models.model_routing_decision import (
    ModelRoutingDecision,
)

pytestmark = pytest.mark.integration


_GENEROUS_LIMITS = ModelBudgetLimits(
    max_tokens=10_000, max_cost_usd=10.0, max_time_s=600.0
)


def _valid_delegation_payload() -> str:
    return json.dumps(
        {
            "correlation_id": "00000000-0000-0000-0000-000000000000",
            "task_type": "test",
            "model_used": "qwen3-coder-30b",
            "endpoint_url": "http://192.168.86.201:8000",
            "content": "def test_compliance_wiring(): assert True",
            "quality_passed": True,
            "quality_score": 0.95,
            "latency_ms": 100,
            "prompt_tokens": 110,
            "completion_tokens": 110,
            "total_tokens": 220,
            "fallback_to_claude": False,
            "failure_reason": "",
            "tokens_to_compliance": 220,
            "compliance_attempts": 1,
        }
    )


def _routing(cid: object) -> ModelRoutingDecision:
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
    cid: object, content: str, *, total_tokens: int
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


class TestComplianceLoopWiringEndToEnd:
    """End-to-end: orchestrator handler + unmocked omnimarket compute nodes."""

    def test_delegation_result_schema_is_registered(self) -> None:
        """Wiring requires the omnimarket schema registry to know
        `delegation_result` — if this regresses the workflow can never opt into
        the compliance loop with a meaningful schema_key."""
        assert "delegation_result" in known_schema_keys()

    def test_first_attempt_invalid_emits_repair_intent_and_stays_routed(self) -> None:
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

        # Attempt 1: missing required fields. Real schema-registry rejects this,
        # real schema-repair builds a repair prompt, real budget-policy says CONTINUE.
        out = wf.handle_inference_response(
            _response(cid, json.dumps({"task_type": "summarize"}), total_tokens=180)
        )

        assert len(out) == 1, f"expected exactly one repair intent, got: {out!r}"
        assert isinstance(out[0], ModelInferenceIntent)
        # Workflow stays in ROUTED (self-loop) so the next inference response
        # for the same correlation_id can re-enter the compliance evaluator.
        assert wf.workflows[cid].state == EnumDelegationState.ROUTED
        # compliance_attempts is incremented for the next attempt, so after
        # attempt 1 fires the counter reads 2.
        assert wf.workflows[cid].compliance_attempts == 2
        assert wf.workflows[cid].accumulated_tokens == 180

    def test_two_attempt_success_transitions_to_completed_with_accumulated_tokens(
        self,
    ) -> None:
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

        # Attempt 1: invalid -> repair intent.
        wf.handle_inference_response(
            _response(cid, json.dumps({"task_type": "summarize"}), total_tokens=180)
        )

        # Attempt 2: valid -> quality gate intent, INFERENCE_COMPLETED, total=400.
        out = wf.handle_inference_response(
            _response(cid, _valid_delegation_payload(), total_tokens=220)
        )

        assert len(out) == 1
        assert isinstance(out[0], ModelQualityGateIntent)
        assert wf.workflows[cid].state == EnumDelegationState.INFERENCE_COMPLETED
        assert wf.workflows[cid].compliance_attempts == 2
        assert wf.workflows[cid].accumulated_tokens == 400
