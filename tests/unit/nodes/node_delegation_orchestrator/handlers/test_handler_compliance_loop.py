# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for HandlerComplianceLoop — OMN-10792.

The handler evaluates one LLM attempt against an output schema and decides
whether the orchestrator should accept the output (compliant), retry with a
repair prompt (non-compliant + budget CONTINUE), or stop (budget ABORT or
non-repairable errors).
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

from omnibase_infra.nodes.node_delegation_orchestrator.handlers.handler_compliance_loop import (
    HandlerComplianceLoop,
)

pytestmark = pytest.mark.unit


_GENEROUS_LIMITS = ModelBudgetLimits(
    max_tokens=10_000,
    max_cost_usd=10.0,
    max_time_s=600.0,
)
_TIGHT_LIMITS = ModelBudgetLimits(
    max_tokens=300,
    max_cost_usd=10.0,
    max_time_s=600.0,
)


def _valid_delegation_output() -> str:
    payload: dict[str, Any] = {
        "correlation_id": "00000000-0000-0000-0000-000000000000",
        "task_type": "summarize",
        "model_used": "qwen3-coder",
        "endpoint_url": "http://192.168.86.201:8000",
        "content": "All tests pass.",
        "quality_passed": True,
        "quality_score": 0.95,
        "latency_ms": 240,
        "prompt_tokens": 80,
        "completion_tokens": 40,
        "total_tokens": 120,
        "fallback_to_claude": False,
        "failure_reason": "",
        "tokens_to_compliance": 120,
        "compliance_attempts": 1,
    }
    return json.dumps(payload)


class TestFirstTryCompliance:
    """When the first LLM output validates, the loop reports compliant=True
    immediately, with attempt_number=1 and total_tokens equal to that single
    attempt's tokens."""

    def test_first_attempt_passes_returns_compliant(self) -> None:
        handler = HandlerComplianceLoop()
        result = handler.evaluate(
            candidate_output=_valid_delegation_output(),
            schema_key="delegation_result",
            original_prompt="Summarize the tests",
            attempt_number=1,
            cumulative_tokens=0,
            attempt_tokens=120,
            budget_limits=_GENEROUS_LIMITS,
        )
        assert result.compliant is True
        assert result.compliance_attempts == 1
        assert result.tokens_to_compliance == 120
        assert result.repair_prompt == ""
        assert result.budget_action == EnumBudgetAction.CONTINUE
        assert result.validated_output == _valid_delegation_output()


class TestRepairOnValidationFailure:
    """When the LLM output is malformed JSON or missing required fields, the
    loop returns a repair prompt and continues — provided the budget allows."""

    def test_malformed_json_returns_repair_prompt(self) -> None:
        handler = HandlerComplianceLoop()
        result = handler.evaluate(
            candidate_output="not valid json {{{",
            schema_key="delegation_result",
            original_prompt="Summarize",
            attempt_number=1,
            cumulative_tokens=0,
            attempt_tokens=80,
            budget_limits=_GENEROUS_LIMITS,
        )
        assert result.compliant is False
        assert result.compliance_attempts == 1
        assert result.tokens_to_compliance == 80
        assert result.repair_prompt != ""
        assert "schema" in result.repair_prompt.lower()
        assert result.budget_action == EnumBudgetAction.CONTINUE
        assert result.abort_reason == ""

    def test_missing_required_field_returns_repair_prompt(self) -> None:
        handler = HandlerComplianceLoop()
        # Drop several required fields.
        partial = json.dumps({"task_type": "summarize"})
        result = handler.evaluate(
            candidate_output=partial,
            schema_key="delegation_result",
            original_prompt="Summarize",
            attempt_number=1,
            cumulative_tokens=0,
            attempt_tokens=50,
            budget_limits=_GENEROUS_LIMITS,
        )
        assert result.compliant is False
        assert result.repair_prompt != ""
        assert (
            "Field required" in result.repair_prompt
            or "missing" in result.repair_prompt
        )
        assert result.budget_action == EnumBudgetAction.CONTINUE


class TestTwoAttemptCompliance:
    """The orchestrator-driven loop with attempt 1 failing and attempt 2 passing
    accumulates tokens correctly and reports compliance_attempts=2."""

    def test_attempt_2_passes_after_attempt_1_failure(self) -> None:
        handler = HandlerComplianceLoop()

        # Attempt 1: malformed output. Tokens=100. Loop says retry.
        first = handler.evaluate(
            candidate_output="bogus output",
            schema_key="delegation_result",
            original_prompt="Summarize the tests",
            attempt_number=1,
            cumulative_tokens=0,
            attempt_tokens=100,
            budget_limits=_GENEROUS_LIMITS,
        )
        assert first.compliant is False
        assert first.tokens_to_compliance == 100
        assert first.compliance_attempts == 1

        # Attempt 2: valid output. Tokens=150. Loop accepts.
        second = handler.evaluate(
            candidate_output=_valid_delegation_output(),
            schema_key="delegation_result",
            original_prompt=first.repair_prompt,  # orchestrator threads it forward
            attempt_number=2,
            cumulative_tokens=first.tokens_to_compliance,
            attempt_tokens=150,
            budget_limits=_GENEROUS_LIMITS,
        )
        assert second.compliant is True
        assert second.compliance_attempts == 2
        assert second.tokens_to_compliance == 250  # 100 + 150
        assert second.repair_prompt == ""


class TestBudgetAbort:
    """Even when the schema-repair handler can build a repair prompt, the
    budget policy may abort the loop because the cumulative token consumption
    has crossed the hard ceiling."""

    def test_budget_abort_terminates_with_no_repair_prompt(self) -> None:
        handler = HandlerComplianceLoop()
        # tight limits = 300 tokens, attempt brings us to 350 → ABORT (>120%)
        result = handler.evaluate(
            candidate_output="bogus",
            schema_key="delegation_result",
            original_prompt="Summarize",
            attempt_number=2,
            cumulative_tokens=200,
            attempt_tokens=150,  # 200+150=350 > 300*1.2=360? No -> THROTTLE not ABORT
            budget_limits=_TIGHT_LIMITS,
            task_priority=EnumTaskPriority.NORMAL,
        )
        # Normal priority: ABORT only when ratio > 1.0; 350/300 = 1.166 > 1.0 → ABORT
        assert result.budget_action == EnumBudgetAction.ABORT
        assert result.compliant is False
        assert result.repair_prompt == ""
        assert result.abort_reason != ""
        assert result.tokens_to_compliance == 350
        assert result.compliance_attempts == 2


class TestUnknownSchemaKey:
    """An unknown schema key short-circuits the loop with ABORT — there is
    nothing to validate against."""

    def test_unknown_key_aborts(self) -> None:
        handler = HandlerComplianceLoop()
        result = handler.evaluate(
            candidate_output=_valid_delegation_output(),
            schema_key="nonexistent_schema",
            original_prompt="x",
            attempt_number=1,
            cumulative_tokens=0,
            attempt_tokens=10,
            budget_limits=_GENEROUS_LIMITS,
        )
        assert result.compliant is False
        assert result.budget_action == EnumBudgetAction.ABORT
        assert "Unknown output schema key" in result.abort_reason
        assert "nonexistent_schema" in result.abort_reason
        assert result.repair_prompt == ""
        assert result.tokens_to_compliance == 10
        assert result.compliance_attempts == 1


class TestFieldDefaults:
    """The result model carries safe defaults so the orchestrator can rely on
    every field being present even on early-terminating paths."""

    def test_first_try_compliance_has_continue_action_default(self) -> None:
        handler = HandlerComplianceLoop()
        result = handler.evaluate(
            candidate_output=_valid_delegation_output(),
            schema_key="delegation_result",
            original_prompt="x",
            attempt_number=1,
            cumulative_tokens=0,
            attempt_tokens=120,
            budget_limits=_GENEROUS_LIMITS,
        )
        # On compliance we don't actually run budget policy — directive is the
        # default CONTINUE so the orchestrator can uniformly read .budget_action.
        assert result.budget_action == EnumBudgetAction.CONTINUE
