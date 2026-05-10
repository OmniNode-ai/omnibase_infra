# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration test for HandlerComplianceLoop — OMN-10792.

End-to-end exercise across three real omnimarket pure-compute nodes:

    schema-registry  -> resolves the JSON schema for `delegation_result`
    schema-repair    -> builds a targeted repair prompt from validation errors
    budget-policy    -> classifies cumulative usage as CONTINUE / WARN / ABORT

The unit tests in `tests/unit/...` mock nothing — the handler IS pure compute
and was always tested against the real omnimarket handlers. This file restates
the full retry loop with concrete inputs that match what the delegation
orchestrator (Task 5) will issue, so we can prove that the contract between
omnibase_infra and the omnimarket pure-compute nodes is wired correctly at
the integration boundary.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from omnimarket.nodes.node_budget_policy_compute.models.model_budget_limits import (
    ModelBudgetLimits,
)
from omnimarket.nodes.node_budget_policy_compute.models.model_budget_policy_enums import (
    EnumBudgetAction,
    EnumTaskPriority,
)
from omnimarket.nodes.node_output_schema_registry_compute.handlers.handler_output_schema_registry import (
    known_schema_keys,
)

from omnibase_infra.nodes.node_delegation_orchestrator.handlers.handler_compliance_loop import (
    HandlerComplianceLoop,
)

pytestmark = pytest.mark.integration


def _delegation_payload(**overrides: Any) -> str:
    base: dict[str, Any] = {
        "correlation_id": "11111111-2222-3333-4444-555555555555",
        "task_type": "summarize",
        "model_used": "qwen3-coder-30b-a3b",
        "endpoint_url": "http://192.168.86.201:8000",
        "content": "Compliance loop integration test — payload validates.",
        "quality_passed": True,
        "quality_score": 0.91,
        "latency_ms": 540,
        "prompt_tokens": 200,
        "completion_tokens": 100,
        "total_tokens": 300,
        "fallback_to_claude": False,
        "failure_reason": "",
        "tokens_to_compliance": 300,
        "compliance_attempts": 1,
    }
    base.update(overrides)
    return json.dumps(base)


class TestRegistryWiring:
    """The omnimarket schema registry must expose `delegation_result` so that
    omnibase_infra can find it without holding a hardcoded reference."""

    def test_delegation_result_is_registered(self) -> None:
        keys = known_schema_keys()
        assert "delegation_result" in keys, (
            f"omnimarket schema registry missing 'delegation_result'; got: {keys}"
        )


class TestEndToEndRetryLoop:
    """Drive the loop the way the orchestrator (Task 5) will: attempt 1 fails,
    repair_prompt drives a fresh inference, attempt 2 passes; total tokens are
    the running sum across both attempts."""

    def test_retry_succeeds_and_accumulates_tokens(self) -> None:
        handler = HandlerComplianceLoop()
        limits = ModelBudgetLimits(
            max_tokens=10_000, max_cost_usd=10.0, max_time_s=600.0
        )

        # Attempt 1: missing required fields. Tokens=180. Loop must request retry.
        first = handler.evaluate(
            candidate_output=json.dumps({"task_type": "summarize"}),
            schema_key="delegation_result",
            original_prompt="Summarize the failing tests in 2 sentences",
            attempt_number=1,
            cumulative_tokens=0,
            attempt_tokens=180,
            budget_limits=limits,
        )
        assert first.compliant is False
        assert first.compliance_attempts == 1
        assert first.tokens_to_compliance == 180
        assert first.repair_prompt != ""
        assert first.budget_action == EnumBudgetAction.CONTINUE
        # Repair prompt should mention the original prompt verbatim.
        assert "Summarize the failing tests" in first.repair_prompt

        # Attempt 2: valid output. Tokens=220. Loop must accept; total=400.
        second = handler.evaluate(
            candidate_output=_delegation_payload(),
            schema_key="delegation_result",
            original_prompt=first.repair_prompt,
            attempt_number=2,
            cumulative_tokens=first.tokens_to_compliance,
            attempt_tokens=220,
            budget_limits=limits,
        )
        assert second.compliant is True
        assert second.compliance_attempts == 2
        assert second.tokens_to_compliance == 400
        assert second.repair_prompt == ""


class TestBudgetGate:
    """The budget policy must hard-abort the loop when cumulative tokens cross
    the ratio>1.0 ceiling for NORMAL priority, even when the schema-repair
    handler has built a perfectly valid repair prompt."""

    def test_normal_priority_aborts_above_ceiling(self) -> None:
        handler = HandlerComplianceLoop()
        # Tight budget: 500 tokens. Three attempts already consumed 600 → ratio 1.2 → ABORT.
        limits = ModelBudgetLimits(max_tokens=500, max_cost_usd=10.0, max_time_s=600.0)
        result = handler.evaluate(
            candidate_output="bogus",
            schema_key="delegation_result",
            original_prompt="x",
            attempt_number=4,
            cumulative_tokens=400,
            attempt_tokens=200,  # Bringing us to 600 / 500 = 1.2 → ABORT
            budget_limits=limits,
            task_priority=EnumTaskPriority.NORMAL,
        )
        assert result.compliant is False
        assert result.budget_action == EnumBudgetAction.ABORT
        assert result.repair_prompt == ""
        assert result.tokens_to_compliance == 600
        assert result.compliance_attempts == 4
        assert "tokens" in result.abort_reason.lower()

    def test_critical_priority_continues_when_normal_would_abort(self) -> None:
        handler = HandlerComplianceLoop()
        # Same numbers as above (ratio 1.2) but CRITICAL priority allows up to 1.20.
        limits = ModelBudgetLimits(max_tokens=500, max_cost_usd=10.0, max_time_s=600.0)
        result = handler.evaluate(
            candidate_output="bogus",
            schema_key="delegation_result",
            original_prompt="x",
            attempt_number=4,
            cumulative_tokens=400,
            attempt_tokens=200,
            budget_limits=limits,
            task_priority=EnumTaskPriority.CRITICAL,
        )
        # CRITICAL allows ratio up to 1.20 → 1.20 not > 1.20 → THROTTLE not ABORT.
        # The orchestrator MAY continue (THROTTLE is non-terminal in our policy).
        assert result.budget_action != EnumBudgetAction.ABORT
        assert result.repair_prompt != ""
