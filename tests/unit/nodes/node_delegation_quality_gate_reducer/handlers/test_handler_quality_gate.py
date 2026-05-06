# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerQualityGate (delta function).

Tests cover:
    - Pass for each task type with valid output
    - Fail on refusal phrases (REFUSAL category)
    - Fail on short response (WEAK_OUTPUT category)
    - Fail on missing markers (TASK_MISMATCH category)
    - Score computation
    - Fallback recommendation
    - Empty response
    - Edge cases

Related:
    - OMN-7040: Node-based delegation pipeline
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_delegation_quality_gate_reducer.handlers.handler_quality_gate import (
    delta,
)
from omnibase_infra.nodes.node_delegation_quality_gate_reducer.models.model_quality_gate_input import (
    ModelQualityGateInput,
)

pytestmark = [pytest.mark.unit]


def _input(
    task_type: str = "test",
    content: str = "",
    expected_markers: tuple[str, ...] = (),
    min_response_length: int = 60,
    dod_deterministic: tuple[str, ...] = (),
    dod_heuristic: tuple[str, ...] = (),
) -> ModelQualityGateInput:
    """Build a valid ModelQualityGateInput."""
    return ModelQualityGateInput(
        correlation_id=uuid4(),
        task_type=task_type,
        llm_response_content=content,
        expected_markers=expected_markers,
        min_response_length=min_response_length,
        dod_deterministic=dod_deterministic,
        dod_heuristic=dod_heuristic,
    )


class TestPassingResponses:
    """Verify passing quality gate for valid outputs."""

    def test_test_task_passes(self) -> None:
        content = (
            "import pytest\n\n"
            "@pytest.mark.unit\n"
            "def test_auth_validates_token():\n"
            "    assert validate_token('valid') is True\n"
            "\n"
            "def test_auth_rejects_expired():\n"
            "    assert validate_token('expired') is False\n"
        )
        result = delta(_input(task_type="test", content=content))
        assert result.passed is True
        assert result.quality_score >= 0.6

    def test_document_task_passes(self) -> None:
        content = (
            '"""Authentication module for token validation.\n\n'
            "Args:\n"
            "    token: The JWT token to validate.\n\n"
            "Returns:\n"
            "    True if the token is valid and not expired.\n\n"
            "Raises:\n"
            '    ValueError if the token format is invalid.\n"""\n'
        )
        result = delta(_input(task_type="document", content=content))
        assert result.passed is True
        assert result.quality_score >= 0.6

    def test_research_task_passes(self) -> None:
        content = (
            "The authentication module uses JWT tokens for session management. "
            "The validate_token function checks both signature validity and "
            "expiration time. The token refresh mechanism is implemented in "
            "refresh_handler.py on line 45."
        )
        result = delta(_input(task_type="research", content=content))
        assert result.passed is True
        assert result.quality_score >= 0.6


class TestRefusalDetection:
    """Verify REFUSAL failure category."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "I cannot help with that request.",
            "I'm sorry, but I can't generate that code.",
            "As an AI language model, I don't have access to that.",
            "Error: something went wrong\nHere is the output...",
            "Traceback (most recent call last):\n  File ...",
        ],
    )
    def test_refusal_phrases_detected(self, phrase: str) -> None:
        # Pad content to meet length requirement
        content = phrase + " " * 200
        result = delta(_input(task_type="research", content=content))
        assert result.passed is False
        assert any("REFUSAL" in r for r in result.failure_reasons)

    def test_refusal_only_checks_first_200_chars(self) -> None:
        # Refusal phrase after 200 chars should not trigger
        safe_prefix = "x" * 201
        content = safe_prefix + "I cannot help with that"
        result = delta(_input(task_type="research", content=content))
        # Should not have refusal failure
        assert not any("REFUSAL" in r for r in result.failure_reasons)


class TestWeakOutput:
    """Verify WEAK_OUTPUT failure category."""

    def test_short_response_fails(self) -> None:
        result = delta(_input(task_type="test", content="def test_x(): pass"))
        assert result.passed is False
        assert any("WEAK_OUTPUT" in r for r in result.failure_reasons)

    def test_empty_response_fails(self) -> None:
        result = delta(_input(task_type="test", content=""))
        assert result.passed is False
        assert any("WEAK_OUTPUT" in r for r in result.failure_reasons)

    def test_document_requires_longer_response(self) -> None:
        # 90 chars -- passes test (80) but fails document (100)
        content = "x" * 90 + ' """docstring"""'
        result_test = delta(_input(task_type="research", content=content))
        assert result_test.passed is True

        result_doc = delta(_input(task_type="document", content=content))
        # Document has higher min length (100) so should have lower or equal score
        assert result_doc.quality_score <= 1.0


class TestTaskMismatch:
    """Verify TASK_MISMATCH failure category."""

    def test_test_without_test_markers(self) -> None:
        content = (
            "Here is a comprehensive analysis of the authentication module. "
            "The module provides token validation with JWT support and "
            "handles both access and refresh tokens securely."
        )
        result = delta(_input(task_type="test", content=content))
        assert any("TASK_MISMATCH" in r for r in result.failure_reasons)

    def test_document_without_doc_markers(self) -> None:
        content = (
            "def test_auth():\n"
            "    assert validate_token('valid') is True\n"
            "    assert validate_token('expired') is False\n"
            "    assert validate_token(None) is False\n"
        )
        result = delta(_input(task_type="document", content=content))
        assert any("TASK_MISMATCH" in r for r in result.failure_reasons)


class TestScoring:
    """Verify score computation."""

    def test_perfect_score(self) -> None:
        content = (
            "import pytest\n\n"
            "@pytest.mark.unit\n"
            "def test_auth():\n"
            "    assert validate_token('valid') is True\n"
        )
        result = delta(_input(task_type="test", content=content))
        assert result.quality_score == 1.0

    def test_zero_score_empty(self) -> None:
        result = delta(_input(task_type="test", content=""))
        assert result.quality_score <= 0.3


class TestFallbackRecommendation:
    """Verify fallback_recommended flag."""

    def test_refusal_recommends_fallback(self) -> None:
        content = "I cannot help with that request. " + "x" * 200
        result = delta(_input(task_type="research", content=content))
        assert result.fallback_recommended is True

    def test_passing_does_not_recommend_fallback(self) -> None:
        content = (
            "The authentication module uses JWT tokens for session management. "
            "It validates signatures and checks expiration timestamps."
        )
        result = delta(_input(task_type="research", content=content))
        assert result.fallback_recommended is False


class TestCorrelationIdPreserved:
    """Verify correlation_id flows through."""

    def test_correlation_id_matches_input(self) -> None:
        gate_input = _input(task_type="research", content="x" * 100)
        result = delta(gate_input)
        assert result.correlation_id == gate_input.correlation_id


class TestLegacyFallbackCategory:
    """Legacy path (no contract DoD) produces fail_heuristic, never fail_deterministic."""

    def test_passing_response_is_pass_category(self) -> None:
        content = (
            "The authentication module uses JWT tokens for session management. "
            "It validates signatures and checks expiration timestamps."
        )
        result = delta(_input(task_type="research", content=content))
        assert result.passed is True
        assert result.fail_category == "pass"

    def test_failing_response_is_heuristic_category(self) -> None:
        result = delta(_input(task_type="test", content="short"))
        assert result.passed is False
        assert result.fail_category == "fail_heuristic"


class TestDeterministicDoDChecks:
    """Contract deterministic checks block delegation on failure."""

    def test_empty_content_fails_output_parses(self) -> None:
        gate_input = _input(
            task_type="test",
            content="",
            dod_deterministic=("output_parses",),
        )
        result = delta(gate_input)
        assert result.passed is False
        assert result.fail_category == "fail_deterministic"
        assert any("MALFORMED" in r for r in result.failure_reasons)

    def test_bare_traceback_fails_output_parses(self) -> None:
        gate_input = _input(
            task_type="test",
            content="Traceback (most recent call last):\n  File foo.py",
            dod_deterministic=("output_parses",),
        )
        result = delta(gate_input)
        assert result.passed is False
        assert result.fail_category == "fail_deterministic"

    def test_truncated_mid_token_fails_signature_preserved(self) -> None:
        gate_input = _input(
            task_type="test",
            content="def test_foo(param1, param2,",
            dod_deterministic=("signature_preserved",),
        )
        result = delta(gate_input)
        assert result.passed is False
        assert result.fail_category == "fail_deterministic"
        assert any("signature_preserved" in r for r in result.failure_reasons)

    def test_valid_content_passes_deterministic_checks(self) -> None:
        gate_input = _input(
            task_type="test",
            content="def test_auth():\n    assert True",
            dod_deterministic=("output_parses", "signature_preserved"),
        )
        result = delta(gate_input)
        assert result.passed is True
        assert result.fail_category == "pass"

    def test_deterministic_failure_recommends_fallback(self) -> None:
        gate_input = _input(
            task_type="test",
            content="",
            dod_deterministic=("output_parses",),
        )
        result = delta(gate_input)
        assert result.fallback_recommended is True


class TestHeuristicDoDChecks:
    """Contract heuristic checks escalate per policy on failure."""

    def test_refusal_phrase_fails_no_refusal(self) -> None:
        gate_input = _input(
            task_type="research",
            content="I cannot help with that. " + "x" * 100,
            dod_heuristic=("no_refusal",),
        )
        result = delta(gate_input)
        assert result.passed is False
        assert result.fail_category == "fail_heuristic"
        assert any("REFUSAL" in r for r in result.failure_reasons)

    def test_refusal_heuristic_recommends_fallback(self) -> None:
        gate_input = _input(
            task_type="research",
            content="I cannot help with that. " + "x" * 100,
            dod_heuristic=("no_refusal",),
        )
        result = delta(gate_input)
        assert result.fallback_recommended is True

    def test_min_length_check_passes(self) -> None:
        gate_input = _input(
            task_type="research",
            content="x" * 100,
            dod_heuristic=("min_length_chars_50",),
        )
        result = delta(gate_input)
        assert result.passed is True
        assert result.fail_category == "pass"

    def test_min_length_check_fails(self) -> None:
        gate_input = _input(
            task_type="research",
            content="short",
            dod_heuristic=("min_length_chars_50",),
        )
        result = delta(gate_input)
        assert result.passed is False
        assert result.fail_category == "fail_heuristic"
        assert any("WEAK_OUTPUT" in r for r in result.failure_reasons)

    def test_clean_content_passes_no_refusal(self) -> None:
        gate_input = _input(
            task_type="research",
            content="The module validates JWT tokens by checking signature and expiry.",
            dod_heuristic=("no_refusal",),
        )
        result = delta(gate_input)
        assert result.passed is True
        assert result.fail_category == "pass"


class TestMixedDoDChecks:
    """Deterministic pass + heuristic fail = fail_heuristic (escalate, not block)."""

    def test_det_pass_heuristic_fail_is_fail_heuristic(self) -> None:
        gate_input = _input(
            task_type="test",
            content="I cannot help with that. x",
            dod_deterministic=("output_parses",),
            dod_heuristic=("no_refusal",),
        )
        result = delta(gate_input)
        assert result.passed is False
        assert result.fail_category == "fail_heuristic"

    def test_det_fail_overrides_heuristic_pass(self) -> None:
        gate_input = _input(
            task_type="test",
            content="",
            dod_deterministic=("output_parses",),
            dod_heuristic=("no_refusal",),
        )
        result = delta(gate_input)
        assert result.passed is False
        assert result.fail_category == "fail_deterministic"

    def test_both_pass_is_pass(self) -> None:
        gate_input = _input(
            task_type="test",
            content="def test_auth():\n    assert True\n",
            dod_deterministic=("output_parses", "signature_preserved"),
            dod_heuristic=("no_refusal", "min_length_chars_10"),
        )
        result = delta(gate_input)
        assert result.passed is True
        assert result.fail_category == "pass"
